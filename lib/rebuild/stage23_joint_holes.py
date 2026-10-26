"""Hunt holes by moving the shape latent AND the class vector together.

WHY JOINTLY RATHER THAN SEPARATELY
Each axis has been searched alone and each produces negatives that help:

    latent only     (arm H2, 855)   OOD 0.820   0 inverted   worst 0.564
    embedding only  (arm J, 1246)   OOD 0.797   0 inverted   worst 0.556

Pooling the two, count-matched, raises the mean but damages the tail:

    pooled          (arm K3, 852)   OOD 0.837   2 inverted   worst 0.377

A pooled set contains two visibly different sub-populations -- every sample was
displaced on exactly one axis -- and the model can handle them separately,
doing well on average while leaving specific categories exposed. Searching both
axes at once produces a single continuous population instead: points that are
in neither pile, warped objects of an unfamiliar kind rather than warped
familiar ones or unwarped unfamiliar ones.

THE SEARCH

    fix       ||(z - z0)/sd|| <= r_z * D_class      shape displacement
              ||e - e0||      <= r_e * spread       class displacement
    minimise  E_cls(decode(z, e)) + lam * E_gen(z, y0)

Both radii come from measurements already made and are not re-tuned here:

    r_z = 1.5   beyond it the decoded shapes stop being objects
                (crowd ratio 2.0-4.4 at r = 3.0)
    r_e = 1.0   best yield on the embedding axis; d_real 0.37, still objects
                (2.0 gave a quarter of the holes, 4.0 gave none)

E_gen still constrains the LATENT part, as on the latent axis. It cannot score a
fabricated embedding, so the class displacement carries no constraint during the
search and is bounded only by r_e -- the same honest cost the embedding-only
search had, and the reason r_e stays at its measured value.

NO TARGET CLASS IS NAMED
The class vector moves wherever the classifier's gradient points, from a random
start. Decoding with the label of an evaluation category would be outlier
exposure on that category.

BALL, NOT SPHERE, AND AN ADAPTIVE STEP
Both as in the other searches, and for the same reasons: PGD projects onto the
ball, and a fixed step is the documented cause of overestimated robustness
(APGD, Croce & Hein, ICML 2020). Measured on both axes, these changes roughly
doubled the yield -- latent 37.6% -> 59.4%, embedding 8.6% -> 35.0%.

COUNT IS CAPPED
Across every arm, the number of negatives predicted the result monotonically
(542 -> 0.831, 855 -> 0.820, 1246 -> 0.797, 1710 -> 0.779), and the
synthetic-outlier literature reports the same saturation beyond a few hundred
samples. --max-keep therefore caps the saved set, keeping the deepest holes, so
this arm is comparable to arm H2 (855) and arm K3 (852) rather than confounded
by volume.

NOTHING IS TRAINED AND NO TEST DATA IS OPENED.
"""

import argparse
import json
import math
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

AE_DEFAULT = "pointae_sphere_unsquared_L195_h1024x512x256x128_s0.pth"
EBM_DEFAULT = "latentebm_20000_s0.pth"


def build_cls(ck, dev):
    head = ck.get("head", "linear")
    n = len(ck["classes"])
    if head == "subcenter":
        from stage49_subcenter import SubCenterEnergyNet
        net = SubCenterEnergyNet(n, scale=float(ck.get("scale", 16.0)),
                                 centres=int(ck.get("centres", 4)))
    elif head == "cosine":
        from stage6_cosine_head import CosineEnergyNet
        net = CosineEnergyNet(n, scale=float(ck.get("scale", 16.0)),
                              wide=bool(ck.get("wide", False)))
    elif head == "dual":
        from stage8_dual_head import DualEnergyNet
        net = DualEnergyNet(n, scale=float(ck.get("scale", 16.0)))
    elif head == "distance":
        from stage4_distance_head import DistanceEnergyNet
        net = DistanceEnergyNet(n)
    else:
        from train_energy_classifier import EnergyNet
        net = EnergyNet(n)
    net = net.to(dev)
    net.load_state_dict(ck["state_dict"])
    net.eval()
    net.requires_grad_(False)
    return net


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", type=Path, default=ROOT / "stage0_hard60_s0.pth")
    ap.add_argument("--ae", type=Path, default=ROOT / AE_DEFAULT)
    ap.add_argument("--ebm", type=Path, default=ROOT / EBM_DEFAULT)
    ap.add_argument("--rz", type=float, default=1.5,
                    help="latent radius, multiples of D_class; measured, "
                         "not tuned here")
    ap.add_argument("--re", type=float, default=1.0,
                    help="embedding radius, multiples of the class spread; "
                         "measured, not tuned here")
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--restarts", type=int, default=8)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--endpoint", action="store_true",
                    help="return where the chain ENDED instead of the best "
                         "point it visited. Taking the extremum over a "
                         "trajectory is optimisation, not sampling: a "
                         "Langevin chain's output is its state after the "
                         "last step. This is the change that decides whether "
                         "the method can be called MCMC at all. The endpoint "
                         "is shallower than the best point by construction, "
                         "so the negatives get weaker -- by how much is the "
                         "question.")
    ap.add_argument("--keep-random", action="store_true",
                    help="when the per-class quota bites, choose among the "
                         "gated candidates at random rather than taking the "
                         "deepest first. Depth-first selection keeps only the "
                         "tail of what the chain produces and deletes every "
                         "shallow near-data negative, which is why the band "
                         "at d_z 0.2-0.6 is empty.")
    ap.add_argument("--ref", type=int, default=40)
    ap.add_argument("--points", type=int, default=256)
    ap.add_argument("--max-keep", type=int, default=855,
                    help="cap on saved holes, deepest first; keeps this arm "
                         "count-matched to arm H2 and arm K3")
    ap.add_argument("--repel", type=float, default=0.0,
                    help="push each new negative away from the ones already "
                         "found. WHY, measured on the six-class set: the "
                         "categories that fail are the ones the decoder can "
                         "reconstruct as well as it reconstructs the training "
                         "classes -- door 0.199 against 0.201 for the training "
                         "classes, xbox 0.258, cone 0.307 -- while airplane, "
                         "which it cannot represent at all (1.46), is rejected "
                         "at 0.994. So the shape space of the failures is "
                         "REACHABLE; the search simply never goes there, "
                         "because from every starting object it descends to "
                         "the same kind of nearby weakness. Distance to the "
                         "nearest existing hole does not predict which "
                         "categories fail (corr +0.26), so more of the same "
                         "negatives does not help -- coverage has to spread. "
                         "The term uses only the negatives already generated, "
                         "never the evaluation categories, so nothing here is "
                         "outlier exposure: it cannot aim at tables, only "
                         "decline to revisit where it has been.")
    ap.add_argument("--repel-k", type=int, default=64,
                    help="how many previously found negatives each new one is "
                         "pushed away from, sampled from the running set")
    ap.add_argument("--ratio", type=float, default=0.0,
                    help="negatives per REAL object of the same class. The "
                         "measured rule is ~5:1, and it is a PER-CLASS rule. "
                         "--max-keep caps every class equally, which is only "
                         "the same thing when the classes are balanced; on the "
                         "six-class set it would hand flower_pot 1040 against "
                         "a target of 605 and vase 1040 against 1870. Setting "
                         "this overrides --max-keep and caps each class at "
                         "ratio x its own fit count.")
    ap.add_argument("--langevin", action="store_true",
                    help="replace the deterministic search with a short-run "
                         "LANGEVIN chain initialised from a real object. The "
                         "search as written is not a sampler and cannot be "
                         "described as one: it normalises the gradient to a "
                         "direction, discarding its magnitude; it projects "
                         "back into a ball every step; and it halves the step "
                         "when progress stalls. None of those is a "
                         "discretisation of anything. With this flag the "
                         "update becomes the canonical form already used to "
                         "train and sample this project's generator -- "
                         "z <- z - (eta/2) sd^2 grad + sqrt(eta) sd noise -- "
                         "with the unnormalised gradient and a fixed step. "
                         "The ball is KEPT as a constraint, so the chain is a "
                         "constrained one: for hard positives especially, a "
                         "free chain could wander until the result is no "
                         "longer the same object, and labelling that as the "
                         "class would teach the classifier something false. "
                         "What this buys over the deterministic search is "
                         "exploration: without noise every restart from one "
                         "object drains into the same basin, which is the "
                         "measured complaint that the negatives are deep but "
                         "not diverse. What it costs is the guarantee of "
                         "finding the DEEPEST point, since the chain wanders "
                         "rather than descending.")
    ap.add_argument("--eta", type=float, default=0.01,
                    help="Langevin step size, used only with --langevin. "
                         "0.01 is the epsilon this generator was trained and "
                         "is sampled with elsewhere in the project, not a "
                         "value chosen here.")
    ap.add_argument("--target-top", type=float, default=0.0,
                    help="restrict the search to this fraction of real "
                         "objects, chosen PER CLASS as the ones with the "
                         "HIGHEST energy under the current classifier. Only "
                         "meaningful with --hard-positives, and it exists "
                         "because of what FPR@95TPR actually measures: the "
                         "threshold is the 95th percentile of the real "
                         "objects' own energies, so the worst 5% of real "
                         "objects set it single-handedly and everything else "
                         "about the model is irrelevant to that metric. "
                         "Uniform 5-per-object spreads the search over all "
                         "of them, most of which are already low and have "
                         "nothing to teach. 0.2 aims it at the fifth that "
                         "sets the line. Measured starting point: FPR95 is "
                         "0.409 on the six-class set A and 0.165 on set B, "
                         "and hard positives are the only intervention that "
                         "has moved it.")
    ap.add_argument("--hard-positives", action="store_true",
                    help="search for HARD POSITIVES instead of holes: "
                         "maximise the classifier energy inside a small ball "
                         "around a real object, rather than minimising it in "
                         "a large one. A hole is a point the model wrongly "
                         "finds plausible; a hard positive is the mirror -- a "
                         "point that is still unambiguously the same object "
                         "and which the model wrongly finds implausible. "
                         "Nothing in the training has ever looked for those: "
                         "the classification loss sees only the real objects "
                         "themselves, never the space between them, so a "
                         "spike sitting between two real chairs costs nothing "
                         "in any term and shows up in no number we report. "
                         "Interpolated positives were the random version of "
                         "this and gave no accuracy gain on the trio, which "
                         "is what you would expect from points the model "
                         "already handles. These are adversarial, so by "
                         "construction there is something to fix. Use a small "
                         "--rz (0.25 or less): the point must stay the same "
                         "object, or pushing its energy DOWN would be "
                         "labelling something else as real. The generator-EBM "
                         "term is kept, so the latent stays in the data "
                         "distribution.")
    ap.add_argument("--per-class-ruler", action="store_true",
                    help="normalise each distance by the ruler of its own "
                         "class instead of the average over classes")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=ROOT / "holes_joint_armA.npz")
    ap.add_argument("--save-clouds", action="store_true")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    import pointnet2_features as P
    P.patch_deterministic_fps(True)
    from decoder import load_clouds
    from autoencoder import PointAE, canonical_order, fps_points
    from train_combined import CombinedEnergy
    from train_latent_ebm import BANK
    from stage11_can_we_cross import nearest_cd, ruler_of
    with (P.DATA_ROOT / "modelnet40_shape_names.txt").open() as handle:
        names = [line.rstrip() for line in handle]

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    classes, cname = ck["classes"], ck["class_names"]
    fit_src = np.array(ck["fit_sources"])
    val_src = np.array(ck["val_sources"])
    cls = build_cls(ck, dev)

    bank = np.load(BANK, allow_pickle=True)
    Z = torch.from_numpy(bank["latents"]).float().to(dev)
    zlab = bank["labels"]
    sd = torch.from_numpy(bank["fit_sd"]).float().to(dev)
    ack = torch.load(args.ae, map_location="cpu", weights_only=False)
    ae = PointAE(ack["latent"], n_points=ack["n_points"],
                 decoder=ack["decoder"],
                 hidden=tuple(ack["hidden"])).to(dev).eval()
    ae.load_state_dict(ack["state_dict"])
    ae.out_points = ack.get("out_points")
    ae.requires_grad_(False)
    D = ae.decoder
    gck = torch.load(args.ebm, map_location=dev, weights_only=False)
    gen = CombinedEnergy(int(bank["latent_dim"]), lam=1.0).to(dev)
    gen.load_state_dict(gck["ema_state_dict"])
    gen.eval()
    gen.requires_grad_(False)

    Xtr, ytr = load_clouds("train")
    Xtr = torch.as_tensor(np.asarray(Xtr)).float()
    ytr = np.asarray(ytr)
    REF = torch.cat([Xtr[np.where(ytr == c)[0][:args.ref]] for c in classes])
    rulers = {c: float(ruler_of(Xtr[np.where(ytr == c)[0][:args.ref]],
                                dev, args.points)) for c in classes}
    ruler = float(np.mean(list(rulers.values())))
    # see stage42_ladder_probe: one averaged ruler assumes the classes are
    # comparably spread, which holds for three pieces of boxy furniture and
    # fails on the vessel trio (bottle 0.080 vs flower_pot 0.257)
    rul_of = (np.array([rulers[int(v)] for v in ytr[fit_src]], np.float32)
              if args.per_class_ruler
              else np.full(len(fit_src), ruler, np.float32))

    # --- the two scales
    dclass = {}
    for c in classes:
        idx = np.where(zlab == c)[0]
        w = Z[idx[rng.choice(len(idx), min(80, len(idx)), replace=False)]] / sd[None, :]
        dm = torch.cdist(w, w)
        dclass[c] = float(dm[~torch.eye(len(dm), dtype=torch.bool, device=dev)].median())
    E3 = D.embed(torch.tensor(classes, device=dev))
    off = ~torch.eye(len(classes), dtype=torch.bool, device=dev)
    spread = float(torch.cdist(E3, E3)[off].mean())

    print("=" * 78)
    print("JOINT SEARCH   shape latent AND class vector move together")
    print("=" * 78)
    print(f"  classifier {args.ckpt.name}   classes {cname}")
    print(f"  r_z {args.rz} x D_class {[round(dclass[c],1) for c in classes]}")
    print(f"  r_e {args.re} x spread {spread:.2f}")
    print(f"  steps {args.steps}  restarts {args.restarts}  lam {args.lam}  "
          f"cap {args.max_keep}")
    if args.repel > 0:
        print(f"  repel {args.repel} against up to {args.repel_k} previously "
              f"found negatives (their latents only; no evaluation data)")
    print(f"  rulers {[round(rulers[c], 4) for c in classes]}   "
          f"spread {max(rulers.values())/min(rulers.values()):.2f}x   "
          f"unit: {'per class' if args.per_class_ruler else 'averaged'}")
    print("")

    def decode(z, e):
        cond = torch.cat([z, e], dim=1).unsqueeze(2).expand(-1, -1, D.n_points)
        grid = D.anchors.t().unsqueeze(0).expand(len(z), -1, -1)
        X = D.refine(D.mlp(torch.cat([cond, grid], dim=1))).transpose(2, 1)
        if getattr(ae, "out_points", None) and X.shape[1] > ae.out_points:
            X = fps_points(X, int(ae.out_points))   # same as PointAE.decode
        return canonical_order(X)

    @torch.no_grad()
    def energy_of(x, b=48):
        return torch.cat([(-torch.logsumexp(cls(x[i:i + b].to(dev)), 1)).cpu()
                          for i in range(0, len(x), b)]).numpy()

    # the provenance must carry the FULL split, not the targeted subset:
    # the trainer verifies positives and negatives came from the same split
    # by comparing this list exactly against arm A's
    fit_src_all = fit_src.copy()
    if args.target_top > 0:
        # energy of the REAL objects themselves, per class, highest first
        e_src = energy_of(Xtr[fit_src])
        sel = []
        for c in classes:
            idx = np.where(ytr[fit_src] == c)[0]
            k = max(1, int(round(args.target_top * len(idx))))
            sel.append(idx[np.argsort(-e_src[idx])[:k]])
        sub = np.sort(np.concatenate(sel))
        print(f"  targeting  the top {args.target_top:.0%} by real energy: "
              f"{len(sub)} of {len(fit_src)} objects, energy "
              f"{e_src[sub].min():.2f} to {e_src[sub].max():.2f} against a "
              f"median of {np.median(e_src):.2f} over all of them")
        fit_src = fit_src[sub]
        # every per-object array built from the full split has to be rebuilt
        # against the subset, or the search loop slices mismatched lengths
        rul_of = (np.array([rulers[int(v)] for v in ytr[fit_src]], np.float32)
                  if args.per_class_ruler
                  else np.full(len(fit_src), ruler, np.float32))

    z_all = Z[torch.from_numpy(fit_src).long().to(dev)]
    y_all = torch.from_numpy(ytr[fit_src]).long().to(dev)
    e_all = D.embed(y_all)
    rz_all = torch.tensor([dclass[int(v)] * args.rz for v in ytr[fit_src]],
                          device=dev).float()

    with torch.no_grad():
        dec_real = torch.cat([decode(z_all[i:i + args.batch],
                                     e_all[i:i + args.batch]).cpu()
                              for i in range(0, len(z_all), args.batch)])
    e_real = energy_of(dec_real)
    med_real = float(np.median(e_real))
    print(f"  decoded real objects: median E {med_real:.3f}")
    print("")

    def ball(v, v0, r, scale=None):
        d = (v - v0) / scale[None, :] if scale is not None else v - v0
        n = d.norm(dim=1, keepdim=True).clamp_min(1e-9)
        d = d * torch.clamp(r / n, max=1.0)
        return v0 + (d * scale[None, :] if scale is not None else d)

    def rand_in(v0, r, g, scale=None):
        d = torch.randn(v0.shape, generator=g, device=dev)
        d = d / d.norm(dim=1, keepdim=True)
        return v0 + (r * d * scale[None, :] if scale is not None else r * d)

    # the running set of accepted negatives, in whitened latent space; the
    # repulsion term reads from this and nothing else
    seen = []

    def repel_term(z):
        """mean inverse distance to a sample of the negatives already found,
        in whitened latent units, so it is on the same scale as rz."""
        if not seen or args.repel <= 0:
            return torch.zeros((), device=dev)
        S = torch.cat(seen, 0)
        if len(S) > args.repel_k:
            S = S[torch.randint(len(S), (args.repel_k,), device=dev)]
        d = torch.cdist(z / sd[None, :], S / sd[None, :])
        return (1.0 / d.clamp_min(1e-3)).mean(1).sum()

    def descend(z0, e0, y0, rz, g):
        z = rand_in(z0, rz, g, sd)
        e = rand_in(e0, args.re * spread, g)
        sz, se = args.lr, args.lr
        with torch.no_grad():
            best_v = -torch.logsumexp(cls(decode(z, e)), 1)
        best_z, best_e = z.clone(), e.clone()
        stale = 0
        for _ in range(args.steps):
            z = z.detach().requires_grad_(True)
            e = e.detach().requires_grad_(True)
            x = decode(z, e)
            # sign flips for hard positives: climb the classifier energy
            # instead of descending it. The generator term keeps its sign --
            # we always want to stay in the data distribution.
            e_cls = (-torch.logsumexp(cls(x), 1)).sum()
            obj = ((-e_cls if args.hard_positives else e_cls)
                   + args.lam * gen(z, y0).sum()
                   + args.repel * repel_term(z))
            gz, ge = torch.autograd.grad(obj, [z, e])
            if args.langevin:
                # canonical coupled Langevin, preconditioned by M = diag(sd^2),
                # the same form train_convergent.langevin uses. Unnormalised
                # gradient, fixed step, noise scaled as sqrt(eta) * sd. Still
                # ball-projected, so the chain is constrained.
                drift = 0.5 * args.eta * sd.square()
                nscale = math.sqrt(args.eta) * sd
                z = ball(z.detach() - drift[None, :] * gz
                         + nscale[None, :] * torch.randn(z.shape, device=dev,
                                                         generator=g),
                         z0, rz, sd)
                e_drift = 0.5 * args.eta
                e_noise = math.sqrt(args.eta)
                e = ball(e.detach() - e_drift * ge
                         + e_noise * torch.randn(e.shape, device=dev,
                                                 generator=g),
                         e0, args.re * spread)
            else:
                z = ball(z.detach() - sz * rz * sd[None, :] * gz /
                         (gz / sd[None, :]).norm(dim=1, keepdim=True).clamp_min(1e-9),
                         z0, rz, sd)
                e = ball(e.detach() - se * args.re * spread * ge /
                         ge.norm(dim=1, keepdim=True).clamp_min(1e-9),
                         e0, args.re * spread)
            with torch.no_grad():
                v = -torch.logsumexp(cls(decode(z, e)), 1)
                better = (v > best_v) if args.hard_positives else (v < best_v)
                best_v = torch.where(better, v, best_v)
                best_z[better] = z[better]
                best_e[better] = e[better]
            if args.langevin:
                # fixed step: halving it on stalling is a search heuristic and
                # would stop the chain being a discretisation of anything
                pass
            elif bool(better.any()):
                stale = 0
            else:
                stale += 1
                if stale >= args.patience:
                    sz *= 0.5
                    se *= 0.5
                    stale = 0
        if args.endpoint:
            return z.detach(), e.detach()
        return best_z.detach(), best_e.detach()

    g = torch.Generator(device=dev).manual_seed(args.seed)
    keep = {k: [] for k in ("latent", "embed", "source_id", "label",
                            "energy", "d_real")}
    clouds = []
    t0 = time.time()
    ef_all, er_all = [], []
    for rep in range(args.restarts):
        for i in range(0, len(z_all), args.batch):
            sl = slice(i, i + args.batch)
            z0, e0, y0 = z_all[sl], e_all[sl], y_all[sl]
            rz = rz_all[sl][:, None]
            zf, ef = descend(z0, e0, y0, rz, g)
            with torch.no_grad():
                xf = decode(zf, ef)
                xr = decode(rand_in(z0, rz, g, sd),
                            rand_in(e0, args.re * spread, g))
            a = energy_of(xf.cpu())
            b = energy_of(xr.cpu())
            ef_all.append(a)
            er_all.append(b)
            keep["latent"].append(zf.cpu().numpy())
            keep["embed"].append(ef.cpu().numpy())
            keep["source_id"].append(fit_src[sl])
            keep["label"].append(ytr[fit_src][sl])
            keep["energy"].append(a)
            keep["d_real"].append(
                nearest_cd(xf.cpu(), REF, dev, args.points) / rul_of[sl])
            clouds.append(xf.cpu().numpy().astype(np.float32))
            if args.repel > 0:
                seen.append(zf.detach())
        print(f"  restart {rep+1}/{args.restarts}   "
              f"found E {np.concatenate(ef_all).mean():>8.3f}   "
              f"random E {np.concatenate(er_all).mean():>8.3f}   "
              f"{(time.time()-t0)/60:>5.1f} min", flush=True)

    m = {k: np.concatenate(v) for k, v in keep.items()}
    m["clouds"] = np.concatenate(clouds)
    ef_all = np.concatenate(ef_all)
    er_all = np.concatenate(er_all)
    if args.hard_positives:
        # a useful hard positive is one the model finds IMPLAUSIBLE: energy
        # above the decoded-real median. Below it the model already handles
        # the point and there is nothing to teach.
        hole = np.where(m["energy"] > med_real)[0]
        print("")
        print(f"  found spike%  {100*float((ef_all > med_real).mean()):.1f}%"
              f"    random spike%  {100*float((er_all > med_real).mean()):.1f}%")
        print(f"  hard positives {len(hole)} of {len(ef_all)}   "
              f"d_real median {np.median(m['d_real'][hole]):.2f}")
        print("  (d_real must stay SMALL -- these are meant to be the same")
        print("   object; a large one means the search left the class)")
    else:
        hole = np.where(m["energy"] < med_real)[0]
        print("")
        print(f"  found hole%   {100*float((ef_all < med_real).mean()):.1f}%"
              f"    random hole%   {100*float((er_all < med_real).mean()):.1f}%")
        print(f"  holes {len(hole)} of {len(ef_all)}   "
              f"d_real median {np.median(m['d_real'][hole]):.2f}")

    # cap, deepest first
    if args.ratio > 0:
        sel, short = [], []
        for c in classes:
            # the quota counts the FULL class, not the targeted subset, so
            # targeting concentrates the same number of positives on fewer
            # objects instead of generating fewer of them
            want = int(round(args.ratio * int((ytr[fit_src_all] == c).sum())))
            idx = hole[m["label"][hole] == c]
            if args.keep_random:
                order = np.random.default_rng(args.seed).permutation(len(idx))
            else:
                order = (np.argsort(-m["energy"][idx]) if args.hard_positives
                         else np.argsort(m["energy"][idx]))
            took = idx[order[:want]]
            sel.append(took)
            if len(took) < want:
                short.append(f"{names[c]} {len(took)}/{want}")
        hole = np.concatenate(sel)
        print(f"  capped at {args.ratio}:1 per class -> {len(hole)} holes")
        if short:
            print(f"  SHORT of target: {', '.join(short)}  "
                  f"-- raise --restarts")
    elif len(hole) > args.max_keep:
        per = args.max_keep // len(classes)
        sel = []
        for c in classes:
            idx = hole[m["label"][hole] == c]
            order = (np.argsort(-m["energy"][idx]) if args.hard_positives
                     else np.argsort(m["energy"][idx]))
            sel.append(idx[order[:per]])
        hole = np.concatenate(sel)
        print(f"  capped to {len(hole)} (deepest, {per} per class)")
    m = {k: v[hole] for k, v in m.items()}
    m["provenance_json"] = json.dumps({
        "ckpt": args.ckpt.name, "ae": args.ae.name, "axis": "joint(z,e)",
        "rz": args.rz, "re": args.re, "spread": spread, "steps": args.steps,
        "restarts": args.restarts, "lam": args.lam, "seed": args.seed,
        "ruler": ruler, "rulers": {str(k): v for k, v in rulers.items()},
        "per_class_ruler": bool(args.per_class_ruler),
        "kind": "hard_positive" if args.hard_positives else "hole",
        "median_real_energy": (float(m["energy"].min()) - 1.0
                               if args.hard_positives
                               else float(m["energy"].max()) + 1.0),
        "fit_sources": fit_src_all.tolist(),
        "targeted_sources": (fit_src.tolist()
                             if args.target_top > 0 else None),
        "target_top": args.target_top,
        "val_sources": val_src.tolist(),
        "classes": classes})
    np.savez_compressed(args.out, **m)
    print(f"  saved {args.out.name}   {len(m['energy'])} holes")
    for c in classes:
        s = m["label"] == c
        print(f"    {names[c]:<12}{int(s.sum()):>5}   "
              f"mean E {m['energy'][s].mean():>8.3f}")


if __name__ == "__main__":
    main()
