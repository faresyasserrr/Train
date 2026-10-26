"""MEASUREMENT ONLY: does a walk away from a real object leave the data steadily?

THE QUESTION
A monotonicity loss would say: energy must rise at every step of a walk away
from a real object. That is only a correct thing to ask if the walk actually
leaves the real data at a steady pace. If the walk stays among real objects for
most of its length and only exits at the very end, then raising the energy of
the early steps would be pushing up genuine data neighbourhoods, and only the
final step deserves a penalty -- which is the wall loss that already exists.

So before writing any loss, measure the geometry.

THE WALK
For each real object, run the same joint (z, e) descent that produces the hard
negatives, to get an endpoint (z*, e*). The walk is the straight segment from
the real object to that endpoint, sampled at

    t = 0, 0.25, 0.50, 0.75, 1.00     x_k = decode(z0 + t_k dz, e0 + t_k de)

The straight segment is used rather than the descent trajectory itself because
the descent is re-projected onto the ball at every step, so its distance from
the real object is not monotone in the step index. Along the segment the
displacement is exactly t_k, which is the ordering the loss would need.

WHAT IS PRINTED
Per rung: the displacement actually travelled, the Chamfer distance to the
nearest real object in class rulers, and the energy. The Chamfer column is the
one that decides the question. Nothing is trained and no test data is opened.
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


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", type=Path, default=ROOT / "stage0_hard60_s0.pth")
    ap.add_argument("--ae", type=Path, default=ROOT / AE_DEFAULT)
    ap.add_argument("--ebm", type=Path, default=ROOT / EBM_DEFAULT)
    ap.add_argument("--rz", type=float, default=1.5)
    ap.add_argument("--re", type=float, default=1.0)
    ap.add_argument("--langevin", action="store_true",
                    help="find the ladder's far endpoint with a constrained "
                         "Langevin chain instead of the PGD-style descent. "
                         "The rungs are training material for the ordering "
                         "loss, so if the method's adversarial data must come "
                         "from MCMC, this search has to change with the "
                         "others -- the endpoint is found the same way a "
                         "negative is. Update, ball and filters are otherwise "
                         "identical to stage23_joint_holes.py --langevin.")
    ap.add_argument("--eta", type=float, default=0.01,
                    help="Langevin step size, used only with --langevin; the "
                         "epsilon this generator is trained and sampled with")
    ap.add_argument("--rungs", type=int, default=4)
    ap.add_argument("--t-start", type=float, default=0.0,
                    help="where the first rung sits along the walk, as a "
                         "fraction. The default 0.0 puts it on the decoded "
                         "real object. The trainer then discards every rung "
                         "closer than min_d_rel x the class floor, which on "
                         "the current file is 0.33-0.47 rulers and lands at "
                         "t = 0.375-0.5 -- so the inner rungs are generated "
                         "and then thrown away (1540 of 3240 kept, 47.5%). "
                         "Starting at 0.5 raises that to 65.5% for the same "
                         "compute, and is the furthest the start can move "
                         "before the span of distances the ladder covers "
                         "begins to shrink, which is the ladder's purpose.")
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--ref", type=int, default=40)
    ap.add_argument("--points", type=int, default=256)
    ap.add_argument("--per-class-ruler", action="store_true",
                    help="normalise each distance by the ruler of its own "
                         "class instead of the average over classes")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=ROOT / "ladders_armA.npz")
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
    from stage23_joint_holes import build_cls
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
    # A ruler is how far apart two real objects of ONE class are. Averaging the
    # three into a single unit assumes the classes are comparably spread. On
    # three pieces of boxy furniture they are (rung x1 sat at 0.277 / 0.269 /
    # 0.297). On the vessel trio they are not: bottle 0.080 against flower_pot
    # 0.257, a factor of three, so the same numeric threshold means something
    # different per class. --per-class-ruler divides each object by the ruler
    # of the class it came from, which makes the unit mean the same thing
    # everywhere and leaves any threshold expressed in it a single number.
    rul_of = (np.array([rulers[int(v)] for v in ytr[fit_src]], np.float32)
              if args.per_class_ruler
              else np.full(len(fit_src), ruler, np.float32))

    dclass = {}
    for c in classes:
        idx = np.where(zlab == c)[0]
        w = Z[idx[rng.choice(len(idx), min(80, len(idx)), replace=False)]] / sd[None, :]
        dm = torch.cdist(w, w)
        dclass[c] = float(dm[~torch.eye(len(dm), dtype=torch.bool, device=dev)].median())
    E3 = D.embed(torch.tensor(classes, device=dev))
    off = ~torch.eye(len(classes), dtype=torch.bool, device=dev)
    spread = float(torch.cdist(E3, E3)[off].mean())

    t0 = args.t_start
    ts = [t0 + (1.0 - t0) * k / args.rungs
          for k in range(args.rungs + 1)]
    print("=" * 78)
    print("LADDER PROBE   does the walk leave the data steadily?")
    print("=" * 78)
    print(f"  classifier {args.ckpt.name}   classes {cname}")
    print(f"  r_z {args.rz} x D_class {[round(dclass[c], 1) for c in classes]}")
    print(f"  r_e {args.re} x spread {spread:.2f}")
    print(f"  rungs t = {[round(t, 2) for t in ts]}   steps {args.steps}")
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

    z_all = Z[torch.from_numpy(fit_src).long().to(dev)]
    y_all = torch.from_numpy(ytr[fit_src]).long().to(dev)
    e_all = D.embed(y_all)
    rz_all = torch.tensor([dclass[int(v)] * args.rz for v in ytr[fit_src]],
                          device=dev).float()

    def ball(v, v0, r, scale=None):
        d = (v - v0) / scale[None, :] if scale is not None else v - v0
        n = d.norm(dim=1, keepdim=True).clamp_min(1e-9)
        d = d * torch.clamp(r / n, max=1.0)
        return v0 + (d * scale[None, :] if scale is not None else d)

    def rand_in(v0, r, g, scale=None):
        d = torch.randn(v0.shape, generator=g, device=dev)
        d = d / d.norm(dim=1, keepdim=True)
        return v0 + (r * d * scale[None, :] if scale is not None else r * d)

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
            obj = (-torch.logsumexp(cls(x), 1)).sum() + args.lam * gen(z, y0).sum()
            gz, ge = torch.autograd.grad(obj, [z, e])
            if args.langevin:
                drift = 0.5 * args.eta * sd.square()
                nscale = math.sqrt(args.eta) * sd
                z = ball(z.detach() - drift[None, :] * gz
                         + nscale[None, :] * torch.randn(z.shape, device=dev,
                                                         generator=g),
                         z0, rz, sd)
                e = ball(e.detach() - 0.5 * args.eta * ge
                         + math.sqrt(args.eta) * torch.randn(e.shape,
                                                             device=dev,
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
                better = v < best_v
                best_v = torch.where(better, v, best_v)
                best_z[better] = z[better]
                best_e[better] = e[better]
            if args.langevin:
                pass                      # fixed step: a chain, not a search
            elif bool(better.any()):
                stale = 0
            else:
                stale += 1
                if stale >= args.patience:
                    sz *= 0.5
                    se *= 0.5
                    stale = 0
        return best_z.detach(), best_e.detach()

    g = torch.Generator(device=dev).manual_seed(args.seed)
    n, R = len(z_all), args.rungs + 1
    lat = np.zeros((n, R, z_all.shape[1]), np.float32)
    emb = np.zeros((n, R, e_all.shape[1]), np.float32)
    eng = np.zeros((n, R), np.float32)
    dzr = np.zeros((n, R), np.float32)
    der = np.zeros((n, R), np.float32)
    dre = np.zeros((n, R), np.float32)
    clouds = np.zeros((n, R, D.n_points, 3), np.float32)
    t0 = time.time()

    for i in range(0, n, args.batch):
        sl = slice(i, i + args.batch)
        z0, e0, y0 = z_all[sl], e_all[sl], y_all[sl]
        rz = rz_all[sl][:, None]
        zf, ef = descend(z0, e0, y0, rz, g)
        dz, de = zf - z0, ef - e0
        dcl = np.array([dclass[int(v)] for v in ytr[fit_src][sl]], np.float32)
        for k, t in enumerate(ts):
            zk, ek = z0 + t * dz, e0 + t * de
            with torch.no_grad():
                xk = decode(zk, ek).cpu()
            lat[sl, k] = zk.cpu().numpy()
            emb[sl, k] = ek.cpu().numpy()
            eng[sl, k] = energy_of(xk)
            dzr[sl, k] = ((t * dz / sd[None, :]).norm(dim=1).cpu().numpy() / dcl)
            der[sl, k] = (t * de).norm(dim=1).cpu().numpy() / spread
            dre[sl, k] = (nearest_cd(xk, REF, dev, args.points)
                          / rul_of[sl])
            clouds[sl, k] = xk.numpy()
        print(f"  {min(i + args.batch, n):>5}/{n}   "
              f"E0 {eng[sl, 0].mean():>7.3f}  E{args.rungs} "
              f"{eng[sl, -1].mean():>7.3f}   "
              f"{(time.time() - t0) / 60:>5.1f} min", flush=True)

    print("")
    print("  THE TABLE THAT DECIDES IT")
    print("  rung       t    ||dz||/D   ||de||/sp   d_real   d_real step   mean E")
    prev = None
    for k, t in enumerate(ts):
        d = float(np.median(dre[:, k]))
        stp = "      -" if prev is None else f"{d - prev:>7.3f}"
        print(f"  x{k}     {t:>5.2f}   {dzr[:, k].mean():>8.2f}   "
              f"{der[:, k].mean():>9.2f}   {d:>6.3f}   {stp}      "
              f"{eng[:, k].mean():>7.3f}")
        prev = d
    med = np.median(dre, axis=0)
    gaps = np.diff(med)
    print("")
    print(f"  steady?  largest d_real step / smallest = "
          f"{gaps.max() / max(gaps.min(), 1e-6):.2f}"
          f"   (1.0 = perfectly even, large = the walk only exits at the end)")
    print(f"  fraction of the total d_real travelled in the LAST rung  "
          f"{gaps[-1] / gaps.sum():.2f}   (0.25 = even)")

    de_ = np.diff(eng, axis=1)
    print("")
    print("  and the energy, for reference (not the deciding column)")
    print(f"    fully monotone ladders  {100 * float((de_ > 0).all(1).mean()):.1f}%")
    print(f"    individual rises        {100 * float((de_ > 0).mean()):.1f}%")
    print("    per-step change        " +
          "  ".join(f"{v:>7.3f}" for v in de_.mean(0)))

    np.savez_compressed(
        args.out, latent=lat, embed=emb, energy=eng, dz_ruler=dzr,
        de_ruler=der, d_real=dre, clouds=clouds, source_id=fit_src,
        label=ytr[fit_src], t=np.array(ts, np.float32),
        provenance_json=json.dumps({
            "ckpt": args.ckpt.name, "ae": args.ae.name, "kind": "ladder",
            "axis": "joint(z,e) segment", "rz": args.rz, "re": args.re,
            "rungs": args.rungs, "t_start": args.t_start, "langevin": bool(args.langevin), "spread": spread, "steps": args.steps,
            "lam": args.lam, "seed": args.seed, "ruler": ruler,
            "rulers": {str(k): v for k, v in rulers.items()},
            "per_class_ruler": bool(args.per_class_ruler),
            "fit_sources": fit_src.tolist(), "val_sources": val_src.tolist(),
            "classes": classes}))
    print(f"\n  saved {args.out.name}   {n} ladders x {R} rungs")
    for c in classes:
        s = ytr[fit_src] == c
        print(f"    {names[c]:<12}{int(s.sum()):>5}   d_real "
              f"{np.median(dre[s, 1]):.3f} -> {np.median(dre[s, -1]):.3f}")


if __name__ == "__main__":
    main()
