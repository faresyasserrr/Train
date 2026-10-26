"""Arm H: arm D, with decoded real objects added as POSITIVES.

THE CONFOUND THIS REMOVES
Measured on arm D, passing a real desk through the frozen autoencoder moves its
energy by +10:

    real desk/dresser/night_stand      raw -5.49      through the AE  +4.50
    real laptop                        raw -6.87      through the AE  +5.36

The classifier separates reconstructions from raw clouds trivially, regardless
of shape. And every negative it has ever been trained on is a decoded cloud,
while every positive is raw:

    positives   raw real objects
    negatives   decoded holes        -> "decoded" is a perfect cue for "fake"

So part of what the wall has learned may be "reconstructions are fake" rather
than "non-desks are fake". Arm B's gain did transfer to raw unseen categories
(0.619 -> 0.762), so it is not ONLY reconstruction detection -- but the cue is
available and free, and a network will use a free cue.

THE FIX
The real pool becomes the raw fit objects AND the same objects passed through
the autoencoder, both labelled with their true class:

    positives   raw real objects  +  the same objects decoded
    negatives   decoded holes

Reconstruction signature then carries no information about the label, and the
only way to separate positives from negatives is shape. Batch size, step count,
split and loss are unchanged, so compute matches arm D exactly; the only
difference is what a "real" example can look like.

Everything below is arm D.

--- arm D's original description follows ---

Arm D: arm B with a scale-invariant (cosine) energy head.

THE DIAGNOSIS THIS ACTS ON
Arm B's energy is linear in the 256-D feature, so

    W_y . h = ||W_y|| * ||h|| * cos(theta_y)

and the readout multiplies two signals that both carry information but mix
badly. Measured on arm B, post hoc, over the 37 unseen categories:

    energy as used                0.762
    energy with norms equalised   0.845
    feature norm alone            0.782
    best cosine alone             0.706

Equalising the norms -- keeping direction, discarding length -- gains 0.083
with nothing retrained. A long feature vector buys low energy regardless of
where it points, which is why bottle (0.630 -> 0.762) and wardrobe (0.627 ->
0.739) recover once the length is removed.

WHY TRAIN WITH IT RATHER THAN PATCH AFTERWARDS
The post-hoc version trains under one geometry and reads out under another: the
features are shaped to exploit magnitude, and then the magnitude is deleted. It
also breaks the energy interpretation -- rescaling features before the readout
means -logsumexp is no longer the marginal of the joint density that the JEM
formulation defines, so it becomes a score rather than an energy.

Training with the cosine head removes both problems. The network learns to put
its information in the DIRECTION, because direction is all the readout can see,
and -logsumexp remains the marginal of the model actually being trained.

THE HEAD

    f_y(x) = s * cos(h, W_y) + b_y        E(x) = -logsumexp_y f_y(x)

s is a fixed scale, not learned and not tuned: a bare cosine lies in [-1, 1],
which is too small a range for cross-entropy to produce confident predictions.
s = 16 is the standard value used with cosine classifiers. It is not selected
from any result here.

NO NEW HYPER-PARAMETER IS CHOSEN FROM DATA
This head was preferred over an additive combination of norm and direction for
exactly that reason: an additive form needs a weight balancing the two terms,
and every category available to choose it on has already been inspected during
development. The cosine head introduces nothing to select.

EVERYTHING ELSE IS ARM B, UNCHANGED
Same 180 real objects, same 542 holes, same split, same loss, same 2,000 steps,
same batch composition. The head is the only difference.

    L_cls    cross-entropy over -E(x, .) on the real clouds
    L_wall   relu(margin*sd - (E(hole) - E_real)) / sd, measured against the
             real clouds in the SAME batch. The scale-free form matters here
             for the same reason it mattered for the quadratic head: an
             absolute margin means different things under different readouts.

SELECTION DISCIPLINE
The 0.845 that motivates this head was measured on the 37 unseen categories,
all of which have been examined during development. A method chosen because it
helps on categories one has looked at is selected on the evaluation set even
when no OOD sample enters training. The a priori argument -- a scale-invariant
readout for a quantity that should not depend on feature magnitude -- stands on
its own, but the validation numbers below cannot settle it. The clean number is
the official ModelNet TEST split, which no stage of this project has opened.
This script does not open it.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
PARENT = ROOT.parent
REPO = PARENT / "Pointnet_Pointnet2_pytorch"
for _p in (str(ROOT), str(PARENT), str(REPO), str(REPO / "models")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from stage0_real_only import auroc                               # noqa: E402


class CosineEnergyNet(nn.Module):
    """PN++ trunk with a scale-invariant energy head.

    forward(x) returns f_y(x) = -E(x, y), matching EnergyNet's contract, so
    argmax classifies and -logsumexp gives E(x) exactly as before.
    """

    def __init__(self, n_classes, dim=256, scale=16.0, wide=False):
        super().__init__()
        from pointnet2_cls_ssg import get_model
        self.b = get_model(num_class=n_classes, normal_channel=False)
        # wide: read the trunk one layer earlier, at fc1's 512-D output,
        # instead of fc2's 256-D. Measured motivation: the correlation between
        # OOD and the distance ordering is +0.99/+0.98/+0.84 on the three
        # class trios and -0.56 on the six-class set -- the two properties stop
        # reinforcing and start competing. Both are read out of the same
        # feature space, which also has to hold twice as many class
        # directions, so the natural reading is that the space ran out of room.
        # This adds no architecture: fc1's 512-D output already exists and is
        # currently discarded by fc2.
        self.wide = wide
        self.w = nn.Parameter(torch.randn(n_classes, 512 if wide else dim)
                              * 0.05)
        self.bias = nn.Parameter(torch.zeros(n_classes))
        self.scale = scale

    def features(self, x):
        b = self.b
        xyz = x.transpose(2, 1)
        l1_xyz, l1_points = b.sa1(xyz, None)
        l2_xyz, l2_points = b.sa2(l1_xyz, l1_points)
        _, l3_points = b.sa3(l2_xyz, l2_points)
        h = l3_points.view(len(x), 1024)
        h = b.drop1(F.relu(b.bn1(b.fc1(h))))
        if self.wide:
            return h
        return b.drop2(F.relu(b.bn2(b.fc2(h))))

    def forward(self, x):
        h = F.normalize(self.features(x), dim=1)
        w = F.normalize(self.w, dim=1)
        return self.scale * (h @ w.T) + self.bias


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--holes", type=Path, default=ROOT / "holes_r15_s0.npz")
    ap.add_argument("--arm-a", type=Path, default=ROOT / "stage0_hard60_s0.pth",
                    help="supplies the split, shared by every arm")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--hole-slots", type=int, default=8)
    ap.add_argument("--scale", type=float, default=16.0,
                    help="standard cosine-classifier scale; not tuned here")
    ap.add_argument("--ae", type=Path,
                    default=ROOT / "pointae_sphere_unsquared_L195_"
                                   "h1024x512x256x128_s0.pth",
                    help="the frozen autoencoder the holes came from; its "
                         "reconstructions become positives here")
    ap.add_argument("--margin", type=float, default=2.0,
                    help="in standard deviations of the real energies")
    ap.add_argument("--lam-wall", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--ood-cap", type=int, default=120)
    ap.add_argument("--out", type=Path, default=ROOT / "stage13_armh_s0.pth")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if args.out.exists() and not args.overwrite:
        raise SystemExit(f"refusing to overwrite {args.out}; pass --overwrite")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    import pointnet2_features as P
    P.patch_deterministic_fps(True)
    from decoder import load_clouds
    with (P.DATA_ROOT / "modelnet40_shape_names.txt").open() as handle:
        names = [line.rstrip() for line in handle]

    a = torch.load(args.arm_a, map_location="cpu", weights_only=False)
    classes = a["classes"]
    cname = a["class_names"]
    C = len(classes)
    remap = {c: i for i, c in enumerate(classes)}
    fit_src = np.array(a["fit_sources"])
    val_src = np.array(a["val_sources"])

    d = np.load(args.holes, allow_pickle=True)
    prov = json.loads(str(d["provenance_json"]))
    if prov["fit_sources"] != fit_src.tolist():
        raise SystemExit("holes come from a different split than arm A")
    keep = np.where(d["energy"] < float(prov["median_real_energy"]))[0]
    H = torch.from_numpy(d["clouds"][keep]).float()

    Xtr, ytr = load_clouds("train")
    Xtr = torch.as_tensor(np.asarray(Xtr)).float()
    ytr = np.asarray(ytr)
    y_fit = np.array([remap[int(v)] for v in ytr[fit_src]])
    y_val = np.array([remap[int(v)] for v in ytr[val_src]])
    X_val = Xtr[val_src]

    # ---- decoded copies of the real fit objects, used as POSITIVES
    from autoencoder import PointAE, canonical_order
    ack = torch.load(args.ae, map_location="cpu", weights_only=False)
    ae = PointAE(ack["latent"], n_points=ack["n_points"],
                 decoder=ack["decoder"],
                 hidden=tuple(ack["hidden"])).to(dev).eval()
    ae.load_state_dict(ack["state_dict"])
    ae.requires_grad_(False)
    dec = []
    with torch.no_grad():
        for i in range(0, len(fit_src), 48):
            sl = fit_src[i:i + 48]
            xb = Xtr[sl].to(dev)
            yb = torch.from_numpy(ytr[sl]).long().to(dev)
            dec.append(canonical_order(ae.decode(ae.encode(xb), yb)).cpu())
    X_dec = torch.cat(dec)
    y_dec = y_fit.copy()
    unseen = [c for c in sorted(set(int(v) for v in ytr))
              if c not in classes and (ytr == c).sum() >= 40]
    n_real = args.batch - args.hole_slots

    bpath = ROOT / f"stage2_armb_s{args.seed}.pth"
    bl = (torch.load(bpath, map_location="cpu", weights_only=False)["log"][-1]
          if bpath.is_file() else None)

    print("=" * 78)
    print("STAGE 13   ARM H   arm D + decoded real objects as positives")
    print("=" * 78)
    print(f"  classes    {cname}")
    print(f"  real       {len(fit_src)} fit objects, holes {len(H)}")
    print(f"  head       f_y(x) = {args.scale:g} * cos(h, W_y) + b_y")
    print(f"  batch      {n_real} real + {args.hole_slots} holes   "
          f"{args.steps} steps, lr {args.lr}, seed {args.seed}")
    print(f"  positives  {len(fit_src)} raw + {len(fit_src)} decoded")
    print("  identical to arm D except that real objects also appear decoded")
    if bl:
        print("")
        print(f"  ARM B was   acc {bl['val_acc']:.1f}%   OOD {bl['ood']:.3f}"
              f"   inverted {bl['inverted']}")
        print("  arm B with norms stripped post hoc reached OOD 0.845")
    print("")

    counts = np.array([(y_fit == i).sum() for i in range(C)], dtype=np.float64)
    w = torch.tensor(counts.sum() / (C * np.maximum(counts, 1)),
                     dtype=torch.float32, device=dev)

    net = CosineEnergyNet(C, scale=args.scale).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)

    @torch.no_grad()
    def fwd(X, b=48):
        return torch.cat([net(X[i:i + b].to(dev)).cpu()
                          for i in range(0, len(X), b)])

    def evaluate():
        net.eval()
        fv = fwd(X_val)
        acc = float((fv.argmax(1).numpy() == y_val).mean()) * 100
        e_in = (-torch.logsumexp(fv, 1)).numpy()
        per, pool = {}, []
        for c in unseen:
            idx = np.where(ytr == c)[0][:args.ood_cap]
            eo = (-torch.logsumexp(fwd(Xtr[idx]), 1)).numpy()
            per[names[c]] = auroc(e_in, eo)
            pool.append(eo)
        eh = float((-torch.logsumexp(fwd(H[:200]), 1)).mean())
        net.train()
        return acc, auroc(e_in, np.concatenate(pool)), per, eh

    g = torch.Generator().manual_seed(args.seed)
    rpool = torch.from_numpy(fit_src)
    t0, log = time.time(), []
    print(f"  {'step':>7}{'cls':>9}{'wall':>9}{'val acc':>10}{'OOD':>8}"
          f"{'inv':>6}{'E hole':>9}{'min':>7}")
    print("  " + "-" * 65)
    rc = rw = 0.0
    for st in range(1, args.steps + 1):
        # half the real slots raw, half decoded: the reconstruction cue then
        # appears on both sides of the label and carries no information
        n_raw = n_real // 2
        n_dec = n_real - n_raw
        ri = rpool[torch.randint(len(rpool), (n_raw,), generator=g)]
        di = torch.randint(len(X_dec), (n_dec,), generator=g)
        hi = torch.randint(len(H), (args.hole_slots,), generator=g)
        xb = torch.cat([Xtr[ri], X_dec[di], H[hi]]).to(dev)
        f = net(xb)
        e = -torch.logsumexp(f, dim=1)
        yb = torch.from_numpy(np.concatenate([
            y_fit[np.searchsorted(fit_src, ri.numpy())],
            y_dec[di.numpy()]])).to(dev)
        l_cls = F.cross_entropy(f[:n_real], yb, weight=w)
        e_real = e[:n_real]
        mu_r = e_real.mean().detach()
        sd_r = e_real.std().detach().clamp_min(1e-6)
        l_wall = F.relu(args.margin * sd_r - (e[n_real:] - mu_r)).mean() / sd_r
        loss = l_cls + args.lam_wall * l_wall
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        rc += float(l_cls.detach())
        rw += float(l_wall.detach())
        if st % args.eval_every == 0 or st == args.steps:
            acc, ood, per, eh = evaluate()
            inv = sum(1 for v in per.values() if v < 0.5)
            mins = (time.time() - t0) / 60
            log.append({"step": st, "cls": rc / args.eval_every,
                        "wall": rw / args.eval_every, "val_acc": acc,
                        "ood": ood, "inverted": inv, "e_hole": eh,
                        "per_category": per, "minutes": mins})
            print(f"  {st:>7}{rc/args.eval_every:>9.4f}"
                  f"{rw/args.eval_every:>9.4f}{acc:>9.1f}%{ood:>8.3f}"
                  f"{inv:>6}{eh:>9.2f}{mins:>7.1f}", flush=True)
            rc = rw = 0.0
            torch.save({"state_dict": net.state_dict(), "classes": classes,
                        "class_names": cname, "val_sources": val_src.tolist(),
                        "fit_sources": fit_src.tolist(),
                        "head": "cosine", "scale": args.scale,
                        "holes": args.holes.name,
                        "config": {k: str(v) for k, v in vars(args).items()},
                        "log": log}, args.out)

    r = log[-1]
    print("")
    print(f"  saved {args.out.name}")
    if bl:
        print("")
        print("  " + "=" * 62)
        print(f"  {'':<14}{'ARM B':>12}{'ARM D':>12}{'change':>12}")
        print("  " + "-" * 62)
        print(f"  {'val accuracy':<14}{bl['val_acc']:>11.1f}%"
              f"{r['val_acc']:>11.1f}%{r['val_acc']-bl['val_acc']:>+12.1f}")
        print(f"  {'OOD AUROC':<14}{bl['ood']:>12.3f}{r['ood']:>12.3f}"
              f"{r['ood']-bl['ood']:>+12.3f}")
        print(f"  {'inverted':<14}{bl['inverted']:>12}{r['inverted']:>12}"
              f"{r['inverted']-bl['inverted']:>+12}")
        print("  " + "=" * 62)
        print("")
        print("  reference: arm B with norms stripped post hoc scored 0.845;")
        print("  arm D should reach at least that if training under the same")
        print("  geometry helps rather than merely reproducing the patch.")
        print("")
        print("  the categories where arm B was weakest")
        print(f"    {'category':<14}{'arm B':>9}{'arm D':>9}{'change':>10}")
        for nm in sorted(bl["per_category"],
                         key=lambda z: bl["per_category"][z])[:8]:
            print(f"    {nm:<14}{bl['per_category'][nm]:>9.3f}"
                  f"{r['per_category'][nm]:>9.3f}"
                  f"{r['per_category'][nm]-bl['per_category'][nm]:>+10.3f}")
    print("")
    print("  Validation numbers. Every unseen category has been examined")
    print("  during development, so these cannot settle the method. The clean")
    print("  number comes from the official TEST split, still unopened.")


if __name__ == "__main__":
    main()
