"""What value of lam puts the two terms of the search objective in balance?

THE PARAMETER
Every chain -- negatives, positives and ladders -- descends or climbs

    E_cls(decode(z, e))  +  lam * E_gen(z, y)

The first term is the classifier's own energy, which is what makes the search
seek the classifier's weak spots rather than wander. The second is the frozen
generative EBM, which keeps the sample inside the data distribution. lam is
the exchange rate between them and has been 1.0 throughout, swept but never
derived.

THE MECHANISM THAT FIXES IT
The two terms are different quantities in different units -- a logsumexp over
six class scores against a latent-space energy -- so 1.0 has no meaning a
priori. What is comparable is how far each one MOVES the chain. The update is
preconditioned by M = diag(sd^2), so the displacement a term contributes is

    0.5 * eta * sd^2 * dTerm/dz

and the ratio of the two displacement norms is the only scale-free statement
one can make about their relative strength. The value of lam that equalises
them is

    lam* = norm(0.5 eta sd^2 dE_cls/dz) / norm(0.5 eta sd^2 dE_gen/dz)

measured along real chains rather than at the starting point, because the
balance can drift as the chain leaves the data.

WHAT IS MEASURED
Chains are run exactly as the negatives are generated -- same r_z, r_e, eta,
steps, same ball constraint -- and at every step both gradients are taken
separately and their preconditioned norms recorded. Reported as the median
ratio over all steps and all chains, with the trajectory split into thirds so
a drift is visible, and with the share of steps where the generator term is
already the larger of the two.

HOW TO READ IT
    lam* near 1.0        the current value already balances the terms and
                         1.0 is derived rather than inherited
    lam* far from 1.0    at lam = 1.0 one term dominates the search by that
                         factor. The derived value is lam*, and whether it
                         helps is a separate measurement
    lam* drifts a lot    no single constant balances the chain, and the
                         honest report is the range with the choice stated

The generating classifier is used, which is the one every sample was searched
against. Nothing is trained, no unseen category is touched, and the TEST
split is not opened.
"""

import argparse
import math
import sys
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


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", type=Path, default=ROOT / "stage62_armA_m10.pth")
    ap.add_argument("--ae", type=Path, default=ROOT / AE_DEFAULT)
    ap.add_argument("--ebm", type=Path, default=ROOT / EBM_DEFAULT)
    ap.add_argument("--rz", type=float, default=1.5)
    ap.add_argument("--re", type=float, default=1.0)
    ap.add_argument("--eta", type=float, default=0.1)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--lam", type=float, default=1.0,
                    help="the value the chain actually runs at, so the "
                         "trajectory is the production one")
    ap.add_argument("--hard", action="store_true",
                    help="climb the classifier energy, as the positives do")
    ap.add_argument("--chains", type=int, default=48)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(args.seed)
    g = torch.Generator(device=dev).manual_seed(args.seed)
    import pointnet2_features as P
    P.patch_deterministic_fps(True)
    from decoder import load_clouds
    from autoencoder import PointAE, canonical_order, fps_points
    from train_combined import CombinedEnergy
    from train_latent_ebm import BANK
    from stage23_joint_holes import build_cls

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    classes = ck["classes"]
    fit_src = np.array(ck["fit_sources"])
    cls = build_cls(ck, dev)

    bank = np.load(BANK, allow_pickle=True)
    Z = torch.from_numpy(bank["latents"]).float().to(dev)
    zlab = np.asarray(bank["labels"])
    sd = torch.from_numpy(bank["fit_sd"]).float().to(dev)

    ack = torch.load(args.ae, map_location="cpu", weights_only=False)
    ae = PointAE(ack["latent"], n_points=ack["n_points"],
                 decoder=ack["decoder"], hidden=tuple(ack["hidden"])
                 ).to(dev).eval()
    ae.load_state_dict(ack["state_dict"])
    ae.out_points = ack.get("out_points")
    ae.requires_grad_(False)
    D = ae.decoder
    gck = torch.load(args.ebm, map_location=dev, weights_only=False)
    gen = CombinedEnergy(int(bank["latent_dim"]), lam=1.0).to(dev)
    gen.load_state_dict(gck["ema_state_dict"])
    gen.eval()
    gen.requires_grad_(False)

    _, ytr = load_clouds("train")
    ytr = np.asarray(ytr)

    dclass = {}
    for c in classes:
        idx = np.where(zlab == c)[0]
        w = Z[idx[rng.choice(len(idx), min(80, len(idx)), replace=False)]] \
            / sd[None, :]
        dm = torch.cdist(w, w)
        dclass[c] = float(
            dm[~torch.eye(len(dm), dtype=torch.bool, device=dev)].median())
    E3 = D.embed(torch.tensor(classes, device=dev))
    off = ~torch.eye(len(classes), dtype=torch.bool, device=dev)
    spread = float(torch.cdist(E3, E3)[off].mean())

    def decode(z, e):
        cond = torch.cat([z, e], 1).unsqueeze(2).expand(-1, -1, D.n_points)
        grid = D.anchors.t().unsqueeze(0).expand(len(z), -1, -1)
        X = D.refine(D.mlp(torch.cat([cond, grid], 1))).transpose(2, 1)
        if getattr(ae, "out_points", None) and X.shape[1] > ae.out_points:
            X = fps_points(X, int(ae.out_points))   # same as PointAE.decode
        return canonical_order(X)

    def ball(v, v0, r, scale=None):
        d = (v - v0) / scale[None, :] if scale is not None else v - v0
        n = d.norm(dim=1, keepdim=True).clamp_min(1e-9)
        d = d * torch.clamp(r / n, max=1.0)
        return v0 + (d * scale[None, :] if scale is not None else d)

    def rand_in(v0, r, scale=None):
        u = torch.randn(v0.shape, device=dev, generator=g)
        u = u / u.norm(dim=1, keepdim=True).clamp_min(1e-9)
        rad = r * torch.rand((len(v0), 1), device=dev,
                            generator=g).pow(1.0 / v0.shape[1])
        d = u * rad
        return v0 + (d * scale[None, :] if scale is not None else d)

    src = rng.choice(fit_src, min(args.chains, len(fit_src)), replace=False)
    z0 = Z[torch.from_numpy(src).long().to(dev)]
    y0 = torch.from_numpy(ytr[src]).long().to(dev)
    e0 = D.embed(y0)
    # one radius per chain, in whitened units, exactly as the generator sets it
    rz = torch.tensor([args.rz * dclass[int(c)] for c in ytr[src]],
                      device=dev)

    z = rand_in(z0, 1.0, sd)
    z = ball(z, z0, rz.median(), sd)
    e = rand_in(e0, args.re * spread)
    drift = 0.5 * args.eta * sd.square()

    rc, rg, ratio, step_of = [], [], [], []
    for t in range(args.steps):
        z = z.detach().requires_grad_(True)
        ec = (-torch.logsumexp(cls(decode(z, e.detach())), 1)).sum()
        gc, = torch.autograd.grad(-ec if args.hard else ec, [z])
        z2 = z.detach().requires_grad_(True)
        eg = gen(z2, y0).sum()
        gg, = torch.autograd.grad(eg, [z2])
        nc = (drift[None, :] * gc).norm(dim=1)
        ng = (drift[None, :] * gg).norm(dim=1).clamp_min(1e-12)
        rc.append(nc.detach().cpu().numpy())
        rg.append(ng.detach().cpu().numpy())
        ratio.append((nc / ng).detach().cpu().numpy())
        step_of.append(np.full(len(nc), t))
        with torch.no_grad():
            tot = gc + args.lam * gg
            z = ball(z.detach() - drift[None, :] * tot
                     + math.sqrt(args.eta) * sd[None, :]
                     * torch.randn(z.shape, device=dev, generator=g),
                     z0, rz.median(), sd)

    rc = np.concatenate(rc)
    rg = np.concatenate(rg)
    ratio = np.concatenate(ratio)
    step_of = np.concatenate(step_of)

    print("=" * 78)
    print("LAM BALANCE   what exchange rate puts the two terms level?")
    print("=" * 78)
    kind = "CLIMBING (positives)" if args.hard else "DESCENDING (negatives)"
    print(f"  {args.chains} chains x {args.steps} steps, {kind}, "
          f"run at lam {args.lam}")
    print(f"  r_z {args.rz}  r_e {args.re}  eta {args.eta}")
    print("  displacement norms are preconditioned by M = diag(sd^2), the")
    print("  same form the chain uses, so the two are directly comparable")
    print("")
    print(f"  {'portion of chain':<22}{'d_cls':>11}{'d_gen':>11}"
          f"{'lam*':>10}{'gen larger':>13}")
    print("  " + "-" * 67)
    thirds = [("first third", step_of < args.steps / 3),
              ("middle third", (step_of >= args.steps / 3)
               & (step_of < 2 * args.steps / 3)),
              ("last third", step_of >= 2 * args.steps / 3),
              ("whole chain", np.ones(len(ratio), bool))]
    for tag, m in thirds:
        print(f"  {tag:<22}{np.median(rc[m]):>11.4f}"
              f"{np.median(rg[m]):>11.4f}{np.median(ratio[m]):>10.3f}"
              f"{(ratio[m] < 1).mean():>12.0%}")
    print("")
    print("  lam* is the median of d_cls / d_gen -- the weight that would")
    print("  make the generator term shift the chain as far as the classifier")
    print("  term does. gen larger is the share of steps where it already")
    print("  does.")
    print("")
    print("  Nothing trained, no unseen category, TEST not opened.")


if __name__ == "__main__":
    main()
