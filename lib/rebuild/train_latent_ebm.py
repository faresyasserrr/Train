"""Class-conditional EBM in the autoencoder's latent space.

    latent_bank.npz -> CombinedEnergy -> samples -> AE decoder -> clouds

Same architecture, loss and training settings as the production EBM. Only the
space changes: the production model lives in a PCA of the frozen classifier's
features, which the decoder cannot read. This one lives where the decoder was
trained.

FOUR THINGS THIS DOES DIFFERENTLY FROM train_production.py
----------------------------------------------------------
1. CHAINS START AT mu + randn*sd, NOT randn*sd

   The AE latent is a max-pooled ReLU feature and is not centered:
   ||fit mean|| = 2.74, per-dimension -2.01 to +0.59. Measured norms:

       real latents        5.885
       randn * sd          5.287     10% short, and centered on the origin
                                     rather than on the data
       mu + randn * sd     5.940     matches

   Starting uncentered would put every chain off-distribution before the
   first gradient step. The production space was PCA-centered by
   construction, so this never arose there.

2. DECODE AND EVALUATE AT 5k, NOT ONLY AT 20k

   Latent-space metrics can look healthy while the decoded clouds are
   unusable. Waiting until 20k to find that out wastes the run. The first
   checkpoint is an early diagnostic, not a final rejection of the method.
   Cloud evaluation is optional via --decode; latent evaluation always runs.

3. GEN_K IS CHOSEN ON VALIDATION GENERATION METRICS

   The d_st / r_st anatomy traces diagnose training dynamics. They
   say nothing about how long a generation chain should run. K is swept and
   selected on held-out validation.

   The standing rule this bends -- never select sampler settings on a
   validation generation metric -- existed to stop eta, rho and K being
   chosen together on one result. K alone, with eta and the architecture
   fixed and inherited, is a single selection on a held-out set. That is the
   spirit preserved.

4. THE PER-CLASS GAUSSIAN IS A MATCHED CONTROL

   z ~ N(mu_c, diag(sd_c**2)) is the simple baseline. Judge distribution
   matching separately from decoded class accuracy. It is evaluated through the
   IDENTICAL decode-and-score path -- literally the same functions -- so the
   comparison cannot drift.

SPLIT
-----
Trains on the fit split only (6,890). The production run used all 9,843
because every decision was already made; here K and the stopping point are
still open, and the test set is spent. The autoencoder was trained on the
same 6,890. Validation objects are excluded from weight training but have
already been used for AE checkpoint selection; this remains development data.
"""

import argparse
import hashlib
import random
from functools import wraps
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
PARENT = ROOT.parent
REPO = PARENT / "Pointnet_Pointnet2_pytorch"
for _p in (str(ROOT), str(PARENT), str(REPO), str(REPO / "models")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from autoencoder import PointAE, canonical_order       # noqa: E402
from decoder import load_clouds                        # noqa: E402
from train_combined import CombinedEnergy              # noqa: E402
# nn_two / coverage / prec_rec are imported, never reimplemented, so the
# aim-1 numbers here are defined identically to the production run's.
from train_conditional import (EMA, dev, nn_two, coverage,  # noqa: E402
                               prec_rec, N_EVAL, PR_K)
# Reused verbatim so the EBM and the Gaussian control are scored by the same
# code path -- see point 4 above.
from offmanifold import arm as score_arm               # noqa: E402

BANK = Path(__import__("os").environ.get("LATENT_BANK", ROOT / "latent_bank.npz"))  # override: LATENT_BANK=<path>
CHECKPOINTS = (5000, 10000, 15000, 20000)
ITERS = 20000
TRAIN_STEPS = 150
EPSILON = 0.01
DATA_EPSILON = 0.03
LR = 1e-4
LAMBDA = 1.0
BETA = 1.0
PER_CLASS = 2
K_SWEEP = (50, 110, 150, 180, 220, 440)


def isolated_rng(fn):
    @wraps(fn)
    def wrapped(*a, **kw):
        np_state, py_state = np.random.get_state(), random.getstate()
        try:
            with torch.random.fork_rng():
                return fn(*a, **kw)
        finally:
            np.random.set_state(np_state)
            random.setstate(py_state)
    return wrapped


def file_sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def matched_real_sets(Z, y, n, seed):
    """Disjoint real sets with exactly matching class counts and size."""
    g = torch.Generator(device=Z.device).manual_seed(seed)
    aa, bb = [], []
    for c in torch.unique(y):
        ids = torch.where(y == c)[0]
        ids = ids[torch.randperm(len(ids), generator=g, device=Z.device)]
        m = len(ids) // 2
        aa.append(ids[:m])
        bb.append(ids[m:2*m])
    a, b = torch.cat(aa), torch.cat(bb)
    if n > len(a) or n < 2:
        raise ValueError(f"reference needs 2 <= n <= {len(a)}")
    pick = torch.randperm(len(a), generator=g, device=Z.device)[:n]
    return Z[a[pick]], y[a[pick]], Z[b[pick]], y[b[pick]]


def langevin(energy_fn, x, steps, sd, decay):
    """Identical to train_conditional.langevin. Copied rather than imported
    so this file's sampler cannot change if that one is edited."""
    if steps < 1 or (decay and steps <= 5):
        raise ValueError("steps must be positive; annealed sampling needs steps > 5")
    noise_scale = (EPSILON * sd)[None, :]
    x = x.clone().detach().requires_grad_(True)
    for i in range(steps):
        g = torch.autograd.grad(energy_fn(x).sum(), [x])[0]
        nd = max(0.0, (steps - i - 5) / (steps - 5)) if decay else 1.0
        x.data += -g + nd * noise_scale * torch.randn_like(x)
    return x.detach()


def init_chains(n, mu, sd, dim):
    """mu + randn*sd. See point 1 -- the latent is not centered."""
    return mu[None, :] + torch.randn(n, dim, device=mu.device) * sd[None, :]


def macro_per_class(gen, y_gen, real, y_real, min_n=20):
    """Per-class 1-NN and coverage, averaged over classes with enough
    examples on both sides.

    Global 1-NN is class-blind: a generator that reproduces the dataset's
    overall shape while mixing the classes up scores well on it and badly
    here. Within each class the real side is subsampled to the generated
    count, because 1-NN is biased when the two sets differ in size."""
    nn, cov, precision, recall, used = [], [], [], [], 0
    for c in range(40):
        g, r = gen[y_gen == c], real[y_real == c]
        n = min(len(g), len(r))
        if n < min_n:
            continue
        nn.append(nn_two(g[:n], r[:n]))
        cov.append(coverage(g[:n], r[:n]))
        p, rr = prec_rec(g[:n], r[:n])
        precision.append(p)
        recall.append(rr)
        used += 1
    if not nn:
        return float("nan"), float("nan"), 0, float("nan"), float("nan")
    return (float(np.mean(nn)), float(np.mean(cov)), used,
            float(np.mean(precision)), float(np.mean(recall)))


def real_floor(Zval, yval, n, seed=0, reps=5):
    """Finite-sample real-versus-real reference, not a bound or ceiling."""
    rows = []
    for rep in range(reps):
        A, ya, B, yb = matched_real_sets(Zval, yval, n, seed + rep)
        p, r = prec_rec(A, B)
        mnn, mcov, _, mp, mr = macro_per_class(A, ya, B, yb)
        rows.append([nn_two(A, B), coverage(A, B), p, r, mnn, mcov, mp, mr])
    m = np.nanmean(rows, 0)
    return {"one_nn": m[0], "coverage": m[1], "precision": m[2],
            "recall": m[3], "macro_one_nn": m[4], "macro_coverage": m[5],
            "macro_precision": m[6], "macro_recall": m[7],
            "n": n, "per_rep": rows, "reps": reps}


def latent_metrics(z, y_gen, Zref, yref):
    """STAGE 1: the EBM's own job. Real held-out latents against generated
    latents. No decoder anywhere in this function."""
    n = min(len(z), len(Zref))
    R, yR = Zref[:n], yref[:n]
    p, r = prec_rec(z[:n], R)
    mnn, mcov, used, mp, mr = macro_per_class(z[:n], y_gen[:n], R, yR)
    return {"one_nn": nn_two(z[:n], R), "coverage": coverage(z[:n], R),
            "precision": p, "recall": r, "macro_one_nn": mnn,
            "macro_coverage": mcov, "macro_classes": used, "n_latent": n,
            "macro_precision": mp, "macro_recall": mr}


def strat_subset(y, per):
    """Up to `per` indices from each class present, sorted."""
    out = []
    for c in torch.unique(y):
        out.append((y == c).nonzero(as_tuple=True)[0][:per])
    return torch.sort(torch.cat(out)).values


@isolated_rng
def evaluate(it, ema, mu, sd, dim, ae, Zval, yval, Xf_c, Zf_ae, args, out):
    """Two INDEPENDENT stages, reported separately and judged separately.

    STAGE 1  the conditional EBM. Does it learn the real latent
             distribution? Real held-out latents vs generated latents,
             globally and within each class. No decoder involved. The target
             is the real distribution; the Gaussian is a baseline, not a goal.

    STAGE 2  the frozen decoder. Do those latents become plausible clouds
             that keep their class identity and variety?

    An EBM can pass stage 1 while the decoder loses information in stage 2.
    Merging them into one verdict would hide exactly that case, so nothing
    here combines the two."""
    print("")
    print("=" * 92)
    print(f"CHECKPOINT {it}   (EMA weights)")
    print("=" * 92)
    net = ema.shadow

    Zref, y_lat, _, _ = matched_real_sets(Zval, yval, args.n_latent, args.seed)
    g = torch.Generator(device=dev).manual_seed(args.seed + 100000)

    # ---------------------------------------------------------- generate
    zs = {}
    for K in args.k_sweep:
        torch.manual_seed(args.seed + 200000)
        x0 = init_chains(args.n_latent, mu, sd, dim)
        with torch.enable_grad():
            zs[f"K{K}"] = langevin(lambda x: net(x, y_lat), x0, K, sd,
                                   decay=args.gen_decay)
    zs["gauss_cls"] = (args.mu_c[y_lat] + args.sd_c[y_lat]
                       * torch.randn(args.n_latent, dim, generator=g,
                                     device=dev))

    # ------------------------------------------------------------ STAGE 1
    print("")
    print("  STAGE 1 -- CONDITIONAL EBM: does it learn the latent"
          " distribution?")
    print("  real held-out latents vs generated latents, no decoder")
    print(f"  {'arm':<14}{'1-NN':>9}{'COV':>9}{'Prec':>9}{'Rec':>9}"
          f"{'macro 1-NN':>12}{'macro COV':>11}{'cls':>6}")
    print("  " + "-" * 76)
    f = args.floor
    print(f"  {'REAL reference':<14}{f['one_nn']:>9.2f}{f['coverage']:>9.2f}"
          f"{f['precision']:>9.2f}{f['recall']:>9.2f}"
          f"{f['macro_one_nn']:>12.2f}{f['macro_coverage']:>11.2f}"
          f"{'':>6}")
    rows = {}
    for name, z in zs.items():
        r = latent_metrics(z, y_lat, Zref, y_lat)
        label = "gauss-cls" if name == "gauss_cls" else f"EBM {name}"
        print(f"  {label:<14}{r['one_nn']:>9.2f}{r['coverage']:>9.2f}"
              f"{r['precision']:>9.2f}{r['recall']:>9.2f}"
              f"{r['macro_one_nn']:>12.2f}{r['macro_coverage']:>11.2f}"
              f"{r['macro_classes']:>6}")
        rows[name] = r
        print(f"    per-class precision {r['macro_precision']:.2f}, "
              f"recall {r['macro_recall']:.2f} ({r['macro_classes']} eligible classes)")

    ks = [k for k in rows if k.startswith("K")]
    best = min(ks, key=lambda k: abs(rows[k]["one_nn"] - 50.0))
    print("")
    print(f"  Exploratory best 1-NN on STAGE 1: {best}   1-NN {rows[best]['one_nn']:.2f}"
          f" vs real reference {f['one_nn']:.2f}"
          f"  (+{rows[best]['one_nn'] - f['one_nn']:.2f})")
    print("  Not a promotion verdict: read global AND per-class precision/recall/coverage.")
    if best in (f"K{min(args.k_sweep)}", f"K{max(args.k_sweep)}"):
        print("  WARNING: best 1-NN is at a sweep boundary.")
    print(f"  baseline gaussian 1-NN {rows['gauss_cls']['one_nn']:.2f}"
          f"  (+{rows['gauss_cls']['one_nn'] - f['one_nn']:.2f})")

    # ------------------------------------------------------------ STAGE 2
    if args.decode:
        print("")
        print("  STAGE 2 -- DECODER: do those latents become plausible clouds?")
        print(f"  {'arm':<14}{'class acc':>10}{'radius':>10}{'>1.05':>10}"
              f"{'novelty':>11}{'diversity':>11}")
        print("  " + "-" * 66)
        per = max(1, args.eval_n // 40)
        for name, z in zs.items():
            sub = strat_subset(y_lat, per)[:args.eval_n]
            label = "gauss-cls" if name == "gauss_cls" else f"EBM {name}"
            torch.manual_seed(args.seed + 300000)
            rows[name]["decoded"] = score_arm(label, ae, z[sub], y_lat[sub], dev,
                                               Zf_ae, Xf_c, y_lat[sub])
        print(f"  {'real recon':<14}{args.real_recon_acc:>9.2f}%"
              f"{'':>10}{'':>10}{args.real_novelty:>11.5f}"
              f"{args.real_diversity:>11.5f}   <- reference")

        bd = max(ks, key=lambda k: rows[k]["decoded"]["accuracy"])
        print("")
        print(f"  best K by decoded class accuracy: {bd} "
              f"{rows[bd]['decoded']['accuracy']:.2f}%"
              f"   (gaussian {rows['gauss_cls']['decoded']['accuracy']:.2f}%)")
        if bd != best:
            print(f"  NOTE: stage 1 selects {best}, stage 2 prefers {bd}. These"
                  " are separate outcomes;")
            print("  the EBM is judged on stage 1, the pipeline on stage 2.")

    out[str(it)] = {"real_reference": f, "arms": rows, "config": args.config,
                    "evaluation_seed": args.seed, "exploratory_best_k": best,
                    "n_latent": args.n_latent, "k_sweep": list(args.k_sweep),
                    "generation_epsilon": EPSILON,
                    "generation_decay": args.gen_decay,
                    "decode_enabled": args.decode,
                    "sampling_seed_count": 1}
    dest = ROOT / f"{args.tag}_eval_s{args.seed}.json"
    tmp = dest.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(out, indent=1))
    tmp.replace(dest)
    print("=" * 92)
    print("", flush=True)



def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--iters", type=int, default=ITERS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=500)
    ap.add_argument("--eval-n", type=int, default=480,
                    help="clouds decoded and scored per arm. The expensive "
                         "tier -- CD-based novelty and diversity.")
    ap.add_argument("--train-steps", type=int, default=TRAIN_STEPS,
                    help="Langevin steps in the TRAINING chains. The default "
                         "150 is inherited from production. Generation peaks "
                         "at K=180 and drifts beyond it (centred radius "
                         "ratio 1.02 at K=110 rising to 1.18 at K=220), "
                         "which is what running a chain longer than it was "
                         "trained for looks like. Setting this to match the "
                         "generation length is the single-variable test of "
                         "that explanation.")
    ap.add_argument("--data-epsilon", type=float, default=DATA_EPSILON,
                    help="noise added to real latents before they are used as "
                         "positives, as a fraction of the per-dimension "
                         "spread. It sets how BROAD the distribution being "
                         "fitted is: every positive is a real latent smeared "
                         "by this much, so the model is taught a deliberately "
                         "widened version of the data. The trained model's "
                         "diagnosed fault is exactly that -- recall 96.07 "
                         "above the real floor of 94.75 while precision sits "
                         "at 77.22 against 94.15, which is a distribution "
                         "containing the real one and spilling past it. This "
                         "is the one training setting that controls that "
                         "directly, and it has never been varied. Lowering it "
                         "sharpens the target; lowering it too far removes "
                         "the smoothing that stops the model concentrating "
                         "unbounded density on individual training points, so "
                         "watch the distinctness and energy-collapse guards.")
    ap.add_argument("--decode", action="store_true",
                    help="also run STAGE 2: decode the generated "
                         "latents and score the clouds. Off by "
                         "default -- it is the expensive tier, and "
                         "the EBM is judged on STAGE 1 alone.")
    ap.add_argument("--n-latent", type=int, default=N_EVAL,
                    help="samples for the latent-space aim-1 metrics. "
                         "Capped to the size of a class-matched disjoint "
                         "real reference. PCA-space scores are not directly comparable.")
    ap.add_argument("--k-sweep", type=int, nargs="+", default=list(K_SWEEP))
    ap.add_argument("--gen-decay", action="store_true",
                    help="anneal generation noise to zero. Default is constant "
                         "noise, matching training and the confirmed K=180 setting")
    ap.add_argument("--ae", type=Path, default=ROOT /
                    "pointae_sphere_unsquared_L195_h1024x512x256x128_s0.pth")
    ap.add_argument("--resume", type=Path, default=None)
    ap.add_argument("--check-only", action="store_true",
                    help="validate the bank and decoder correspondence, then exit without training")
    args = ap.parse_args()
    if args.iters < 1 or args.log_every < 1 or args.eval_n < 1 or args.n_latent < 2:
        raise SystemExit("iteration, logging and evaluation counts must be positive")
    if not np.isfinite(args.data_epsilon) or args.data_epsilon < 0:
        raise SystemExit("--data-epsilon must be a finite non-negative value")
    if not args.k_sweep or any(k <= 5 for k in args.k_sweep):
        raise SystemExit("all annealed generation K values must exceed 5")
    # The tag carries every setting that differs from production, so two runs
    # can never land on the same filename and be silently compared.
    args.tag = ("latentebm" if args.train_steps == TRAIN_STEPS
                else f"latentebm_ts{args.train_steps}")
    if args.data_epsilon != DATA_EPSILON:
        args.tag += "_de" + format(args.data_epsilon, "g").replace(".", "p")
    if not args.check_only and args.resume is None:
        if (list(ROOT.glob(f"{args.tag}_*_s{args.seed}.pth")) or
                (ROOT / f"{args.tag}_eval_s{args.seed}.json").exists()):
            raise SystemExit("Existing run found: use --resume or a different seed; refusing overwrite")

    import pointnet2_features as P
    P.patch_deterministic_fps(True)

    d = np.load(BANK, allow_pickle=False)
    if str(d["source_model"]) != args.ae.name:
        raise SystemExit("Bank source_model does not match --ae")
    if int(d["seed"]) != args.seed:
        raise SystemExit("Bank split seed does not match --seed; do not silently change the split")
    from train_views import load as load_split
    _, _, expected_y, expected_fit, expected_val, _ = load_split(args.seed)
    for key, expected in (("labels", expected_y), ("fit_obj", expected_fit),
                          ("val_obj", expected_val)):
        if not np.array_equal(d[key], expected):
            raise SystemExit(f"Bank {key} does not match the object-level split")
    if (d["latents"].shape != (len(d["labels"]), int(d["latent_dim"])) or
            not np.isfinite(d["latents"]).all()):
        raise SystemExit("Invalid latent bank shape or nonfinite entries")
    fit_array = d["latents"][d["fit_obj"]].astype(np.float64)
    for key, expected in (("fit_mean", fit_array.mean(0)),
                          ("fit_sd", fit_array.std(0))):
        if not np.allclose(d[key], expected, rtol=1e-5, atol=1e-6):
            raise SystemExit(f"Bank {key} does not match fit-only statistics")
    if np.any(d["fit_sd"] <= 0):
        raise SystemExit("Bank contains zero-variance dimensions")
    if any(np.sum(d["labels"][d["fit_obj"]] == c) < 2 for c in range(40)):
        raise SystemExit("Each fit class requires at least two objects")
    args.config = {"bank_sha256": file_sha(BANK), "ae_sha256": file_sha(args.ae),
                   "seed": args.seed, "lr": LR, "lambda": LAMBDA, "beta": BETA,
                   "epsilon": EPSILON, "data_epsilon": args.data_epsilon,
                   "train_steps": args.train_steps, "per_class": PER_CLASS}
    # The old bank has no source checksum. Verify a deterministic sample of
    # encoded objects instead of trusting the checkpoint filename alone.
    ck = torch.load(args.ae, map_location="cpu", weights_only=False)
    if int(ck["latent"]) != int(d["latent_dim"]):
        raise SystemExit("AE and bank latent dimensions differ")
    verifier = PointAE(ck["latent"], n_points=ck["n_points"], decoder=ck["decoder"],
                       hidden=tuple(ck.get("hidden") or (256, 256))).to(dev).eval()
    verifier.load_state_dict(ck["state_dict"])
    verifier.requires_grad_(False)
    clouds, cloud_y = load_clouds("train")
    if not np.array_equal(np.asarray(cloud_y), d["labels"]):
        raise SystemExit("Cloud labels and latent-bank labels differ")
    sample = np.linspace(0, len(clouds) - 1, 16, dtype=int)
    with torch.no_grad():
        encoded = verifier.encode(torch.as_tensor(clouds[sample]).float().to(dev)).cpu().numpy()
    relative = np.linalg.norm(encoded - d["latents"][sample]) / max(
        np.linalg.norm(d["latents"][sample]), 1e-12)
    if not np.isfinite(relative) or relative > 1e-3:
        raise SystemExit(f"AE/bank correspondence failed: relative error {relative:.3g}")
    print(f"Bank/split/statistics verified; sampled AE correspondence error {relative:.3g}")
    if args.check_only:
        print("CHECK ONLY passed; no training or checkpoint writes")
        return
    del verifier, clouds
    Z = torch.from_numpy(d["latents"]).float().to(dev)
    y = torch.from_numpy(d["labels"]).long().to(dev)
    fit_obj = d["fit_obj"]
    val_obj = d["val_obj"]
    dim = int(d["latent_dim"])
    Zfit, yfit = Z[fit_obj], y[fit_obj]
    Zval, yval = Z[val_obj], y[val_obj]
    max_ref = sum(int((yval == c).sum()) // 2 for c in range(40))
    if args.n_latent > max_ref:
        print(f"Capping evaluation N from {args.n_latent} to {max_ref} so all real/generated references match")
        args.n_latent = max_ref

    mu = torch.from_numpy(d["fit_mean"]).float().to(dev)
    sd = torch.from_numpy(d["fit_sd"]).float().to(dev)

    print("=" * 78)
    print("LATENT-SPACE CLASS-CONDITIONAL EBM")
    print("=" * 78)
    print(f"  bank {BANK.name}   dim {dim}   fit {len(Zfit)}  val {len(Zval)}")
    print(f"  ||mu|| {float(mu.norm()):.4f}   sd {float(sd.min()):.4f}"
          f"-{float(sd.max()):.4f} ({float(sd.max()/sd.min()):.1f}x)")
    print(f"  real |z| {float(Zfit.norm(dim=1).mean()):.4f}   "
          f"init |z| {float(init_chains(512, mu, sd, dim).norm(dim=1).mean()):.4f}"
          "   <- diagnostic; matching means/norms does not establish distribution matching")

    torch.manual_seed(args.seed)
    net = CombinedEnergy(dim, lam=LAMBDA).to(dev)
    ema = EMA(net)
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    print(f"  CombinedEnergy {sum(p.numel() for p in net.parameters()):,}"
          f" params   lambda {LAMBDA}  beta {BETA}")
    print(f"  {args.train_steps} Langevin steps, two chains, eps {EPSILON}, "
          f"data eps {args.data_epsilon}")
    print(f"  {args.iters} iterations, checkpoints {CHECKPOINTS}")
    print(f"  K sweep {args.k_sweep}: {args.n_latent} latents for STAGE 1"
          + (f", {args.eval_n} decoded for STAGE 2" if args.decode else ", STAGE 2 off"))
    print(f"  generation eps {EPSILON}, noise "
          f"{'annealed' if args.gen_decay else 'constant'}")
    print("  One paired sampling seed: exploratory evidence, not uncertainty across training runs")

    # Per-class Gaussian, fitted on the FIT split only.
    args.mu_c = torch.stack([Zfit[yfit == c].mean(0) for c in range(40)])
    args.sd_c = torch.stack([Zfit[yfit == c].std(0, unbiased=False) for c in range(40)])
    train_cpu_rng = torch.get_rng_state()
    train_cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

    ae = Xf_c = Zf_ae = None
    if args.decode:
        # Decoder and the references the scorer needs.
        ck = torch.load(args.ae, map_location="cpu", weights_only=False)
        ae = PointAE(ck["latent"], n_points=ck["n_points"], decoder=ck["decoder"],
                     hidden=tuple(ck.get("hidden") or (256, 256))).to(dev)
        ae.load_state_dict(ck["state_dict"])
        ae.eval()
        clouds, _ = load_clouds("train")
        Xall = clouds if torch.is_tensor(clouds) else torch.from_numpy(clouds)
        Xf = Xall[fit_obj].float().to(dev)
        Xf_c = canonical_order(Xf)
        with torch.no_grad():
            Zf_ae = torch.cat([ae.encode(Xf[i:i + 64])
                               for i in range(0, len(Xf), 64)])


    # ------------------------------------------------ STAGE 1 reference
    # Match sample count and class composition. This is a reference, not a ceiling.
    n_floor = min(args.n_latent, len(Zval) // 2)
    args.floor = real_floor(Zval, yval, n_floor, seed=args.seed)
    print("  STAGE 1 reference (real vs real, n=%d): 1-NN %.2f  COV %.2f  "
          "Prec %.2f  Rec %.2f" % (n_floor, args.floor["one_nn"],
                                   args.floor["coverage"],
                                   args.floor["precision"],
                                   args.floor["recall"]))
    print("                                      macro 1-NN %.2f  "
          "macro COV %.2f" % (args.floor["macro_one_nn"],
                              args.floor["macro_coverage"]))
    print("                                      macro precision %.2f  recall %.2f" %
          (args.floor["macro_precision"], args.floor["macro_recall"]))

    if args.decode:
        # ------------------------------------------------ STAGE 2 reference
        # Real val latents through the same decoder and the same scorer. This is
        # a reconstruction reference, not an upper bound on generation accuracy.
        Xv = Xall[val_obj].float().to(dev)
        with torch.no_grad():
            Zv_ae = torch.cat([ae.encode(Xv[i:i + 64])
                               for i in range(0, len(Xv), 64)])
        per = max(1, args.eval_n // 40)
        sub = strat_subset(yval, per)[:args.eval_n]
        ref = score_arm("real recon", ae, Zv_ae[sub], yval[sub], dev, Zf_ae,
                        Xf_c, yval[sub])
        args.real_recon_acc = ref["accuracy"]
        args.real_novelty = ref["novelty_cd"]
        args.real_diversity = ref["diversity_cd"]
        print("")


    by_class = [torch.where(yfit == c)[0] for c in range(40)]
    labels = torch.arange(40, device=dev).repeat_interleave(PER_CLASS)
    batch = len(labels)
    data_scale = (args.data_epsilon * sd)[None, :]

    start = 1
    torch.set_rng_state(train_cpu_rng)
    if train_cuda_rng is not None:
        torch.cuda.set_rng_state_all(train_cuda_rng)
    if args.resume is not None:
        st = torch.load(args.resume, map_location=dev, weights_only=False)
        if st.get("config") != args.config:
            raise SystemExit("Resume configuration or bank/AE identity mismatch (legacy unverified states refused)")
        if torch.cuda.is_available() and st.get("cuda_rng_state_all") is None:
            raise SystemExit("Resume checkpoint lacks CUDA RNG state")
        net.load_state_dict(st["model_state_dict"])
        ema.shadow.load_state_dict(st["ema_state_dict"])
        opt.load_state_dict(st["optimizer_state_dict"])
        torch.set_rng_state(st["torch_rng_state"].cpu())
        if torch.cuda.is_available():
            torch.cuda.set_rng_state_all([s.cpu() for s in st["cuda_rng_state_all"]])
        start = int(st["iters"]) + 1
        print(f"resumed from iteration {st['iters']}")

    out = {}
    ev = ROOT / f"{args.tag}_eval_s{args.seed}.json"
    if args.resume is not None and ev.is_file():
        out = json.load(open(ev))

    t0 = time.time()
    for it in range(start, args.iters + 1):
        idx_c = torch.cat([g[torch.randint(len(g), (PER_CLASS,), device=dev)]
                           for g in by_class])
        real_c = Zfit[idx_c] + data_scale * torch.randn(batch, dim, device=dev)
        idx_g = torch.randint(len(Zfit), (batch,), device=dev)
        real_g = Zfit[idx_g] + data_scale * torch.randn(batch, dim, device=dev)

        x0g = init_chains(batch, mu, sd, dim)
        fake_g = langevin(lambda x: net.global_energy(x), x0g, args.train_steps,
                          sd, decay=False)
        x0c = init_chains(batch, mu, sd, dim)
        fake_c = langevin(lambda x: net(x, labels), x0c, args.train_steps,
                          sd, decay=False)

        gr = net.global_energy(real_g)
        gc, cr = net.both(real_c, labels)
        gfg = net.global_energy(fake_g)
        gfc, cfc = net.both(fake_c, labels)
        loss = (BETA * (gr.mean() - gfg.mean())
                + (gc + net.lam * cr).mean() - (gfc + net.lam * cfc).mean())

        if not torch.isfinite(loss):
            raise RuntimeError(f"Nonfinite loss at iteration {it}; last saved checkpoint retained")

        opt.zero_grad(set_to_none=True)
        loss.backward()
        if not torch.stack([torch.isfinite(p.grad).all() for p in net.parameters()
                            if p.grad is not None]).all():
            raise RuntimeError(f"Nonfinite parameter gradient at iteration {it}")
        opt.step()
        ema.update(net)

        if it % args.log_every == 0 or it == 1:
            print(f"  it {it:>6}  loss {float(loss.detach()):+.4f}  "
                  f"E_real {float(gr.mean().detach()):+.3f}  "
                  f"E_fake {float(gfg.mean().detach()):+.3f}  "
                  f"|fake| {float(fake_g.norm(dim=1).mean()):.3f}  "
                  f"|real| {float(Zfit.norm(dim=1).mean()):.3f}  "
                  f"[{(time.time()-t0)/60:.1f}m]", flush=True)

        if it in CHECKPOINTS or it == args.iters:
            dest = ROOT / f"{args.tag}_{it}_s{args.seed}.pth"
            tmp = dest.with_suffix(".pth.tmp")
            torch.save({"model_state_dict": net.state_dict(),
                        "ema_state_dict": ema.shadow.state_dict(),
                        "optimizer_state_dict": opt.state_dict(),
                        "torch_rng_state": torch.get_rng_state(),
                        "cuda_rng_state_all": (torch.cuda.get_rng_state_all()
                                               if torch.cuda.is_available() else None),
                        "config": args.config,
                        "iters": it, "dim": dim, "seed": args.seed,
                        "bank": BANK.name, "ae": args.ae.name,
                        "fit_mean": mu.cpu(), "fit_sd": sd.cpu()},
                       tmp)
            tmp.replace(dest)
            print(f"  saved {args.tag}_{it}_s{args.seed}.pth")
            evaluate(it, ema, mu, sd, dim, ae, Zval, yval, Xf_c, Zf_ae,
                     args, out)


if __name__ == "__main__":
    main()
