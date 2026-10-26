"""Arm E: an energy that reads both the separating direction and the class centre.

THE DIAGNOSIS THIS ACTS ON
Measured on arm D (stage7_weight_vs_centre.py), the learned weight vectors and
the class feature means point in almost perpendicular directions:

    class          cos(W_y, mu_y)   angle
    desk                    0.387     67 degrees
    dresser                 0.328     71
    night_stand             0.446     63

They are different objects. W_y is whatever direction separates desk from
dresser from night_stand -- the only job cross-entropy ever asked for. mu_y is
where the class actually sits. Nothing in the training objective ties them
together.

Our energy reads W_y alone, so it answers "how strongly does the separating
rule fire?" rather than "is this near anything real?". A laptop fires the rule
harder than a real desk does:

    category      cos(h,W)  x real   cos(h,mu)  x real   AUROC W   AUROC mu
    laptop           0.432    1.25       0.903    0.96     0.193     0.655
    glass_box        0.420    1.22       0.917    0.98     0.346     0.591
    REAL             0.345    1.00       0.939    1.00

Used alone as OOD scores over all 37 categories: weights 0.810, centres 0.856.

THE HEAD

    f_y(x) = s * [ cos(h, W_y) + cos(h, mu_y) ]      E(x) = -logsumexp_y f_y(x)

Both terms, because each does the job the other is bad at. W_y keeps the
classification accuracy -- it is the direction that genuinely separates the
classes. mu_y supplies the rejection. Replacing W_y outright would have risked
the accuracy, since a direction chosen to DESCRIBE a class is not necessarily
the best one to SEPARATE it.

mu_y IS NOT TRAINED BY GRADIENT DESCENT
It is an exponential moving average of the features of real class-y objects,
updated from the batch. If it were a free parameter, cross-entropy would pull
it toward a separating direction exactly as it pulled W_y, and the second term
would collapse into a copy of the first. Keeping it a running statistic pins it
to where the data is, by construction.

Only real clouds update it. Holes are not members of any class and must not
move a class centre.

NO WEIGHT IS SELECTED FROM DATA
The two terms are added with equal weight, fixed at 1. A free balance
coefficient would have to be chosen, and every unseen category available to
choose it on has already been inspected during development. Equal weight is the
neutral default -- the same reasoning as HEAT's beta = 0 -- and introduces
nothing tuned. The EMA momentum is 0.9, a standard value, not swept.

EVERYTHING ELSE IS ARM D, UNCHANGED
Same 180 real objects, same 542 holes, same split, same loss, same 2,000 steps,
same batch composition, same scale s = 16. The head is the only difference.

SELECTION DISCIPLINE
0.856 was measured on the 37 unseen categories, all of which have been examined
during development, so it cannot settle the method -- it motivates it. The
argument that an energy should be anchored to where a class is rather than to
how it is separated stands on its own. The clean number is the official
ModelNet TEST split, which no stage of this project has opened, and this script
does not open it either.
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


class DualEnergyNet(nn.Module):
    """PN++ trunk, energy from the separating direction AND the class centre.

    forward(x) returns f_y(x) = -E(x, y), so argmax classifies and -logsumexp
    gives E(x), exactly as in every other arm.
    """

    def __init__(self, n_classes, dim=256, scale=16.0, momentum=0.9):
        super().__init__()
        from pointnet2_cls_ssg import get_model
        self.b = get_model(num_class=n_classes, normal_channel=False)
        self.w = nn.Parameter(torch.randn(n_classes, dim) * 0.05)
        self.bias = nn.Parameter(torch.zeros(n_classes))
        # a buffer, not a parameter: no gradient ever reaches it
        self.register_buffer("mu", torch.zeros(n_classes, dim))
        self.register_buffer("mu_seen", torch.zeros(n_classes))
        self.scale = scale
        self.momentum = momentum

    def features(self, x):
        b = self.b
        xyz = x.transpose(2, 1)
        l1_xyz, l1_points = b.sa1(xyz, None)
        l2_xyz, l2_points = b.sa2(l1_xyz, l1_points)
        _, l3_points = b.sa3(l2_xyz, l2_points)
        h = l3_points.view(len(x), 1024)
        h = b.drop1(F.relu(b.bn1(b.fc1(h))))
        return b.drop2(F.relu(b.bn2(b.fc2(h))))

    @torch.no_grad()
    def update_centres(self, h, y):
        """EMA of the real class features. Call with REAL clouds only."""
        for c in y.unique():
            m = h[y == c].mean(0)
            if self.mu_seen[c] == 0:
                self.mu[c] = m
                self.mu_seen[c] = 1
            else:
                self.mu[c].mul_(self.momentum).add_(m, alpha=1 - self.momentum)

    def forward(self, x, h=None):
        if h is None:
            h = self.features(x)
        hn = F.normalize(h, dim=1)
        cw = hn @ F.normalize(self.w, dim=1).T
        # before the first update the centres are zero; normalize would give
        # nan, so fall back to the weight term alone for those steps
        cm = hn @ F.normalize(self.mu + 1e-8, dim=1).T
        cm = cm * self.mu_seen[None]
        return self.scale * (cw + cm) + self.bias


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--holes", type=Path, default=ROOT / "holes_r15_s0.npz")
    ap.add_argument("--arm-a", type=Path, default=ROOT / "stage0_hard60_s0.pth")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--hole-slots", type=int, default=8)
    ap.add_argument("--scale", type=float, default=16.0)
    ap.add_argument("--momentum", type=float, default=0.9,
                    help="EMA for the class centres; standard, not swept")
    ap.add_argument("--margin", type=float, default=2.0,
                    help="in standard deviations of the real energies")
    ap.add_argument("--lam-wall", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--ood-cap", type=int, default=120)
    ap.add_argument("--out", type=Path, default=ROOT / "stage8_arme_s0.pth")
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
    classes, cname = a["classes"], a["class_names"]
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
    unseen = [c for c in sorted(set(int(v) for v in ytr))
              if c not in classes and (ytr == c).sum() >= 40]
    n_real = args.batch - args.hole_slots

    prev = {}
    for tag, p in (("ARM B", ROOT / f"stage2_armb_s{args.seed}.pth"),
                   ("ARM D", ROOT / f"stage6_armd_s{args.seed}.pth")):
        if p.is_file():
            prev[tag] = torch.load(p, map_location="cpu",
                                   weights_only=False)["log"][-1]

    print("=" * 78)
    print("STAGE 8   ARM E   energy = separating direction + class centre")
    print("=" * 78)
    print(f"  classes    {cname}")
    print(f"  real       {len(fit_src)} fit objects, holes {len(H)}")
    print(f"  head       f_y(x) = {args.scale:g} * "
          f"[cos(h,W_y) + cos(h,mu_y)] + b_y")
    print(f"  mu_y       EMA of real class features, momentum "
          f"{args.momentum}, no gradient")
    print(f"  batch      {n_real} real + {args.hole_slots} holes   "
          f"{args.steps} steps, seed {args.seed}")
    print("  identical to arm D except the head")
    print("")
    for tag, l in prev.items():
        print(f"  {tag} was   acc {l['val_acc']:.1f}%   OOD {l['ood']:.3f}"
              f"   inverted {l['inverted']}")
    print("  post hoc on arm D: centres alone scored 0.856, weights 0.810")
    print("")

    counts = np.array([(y_fit == i).sum() for i in range(C)], dtype=np.float64)
    w = torch.tensor(counts.sum() / (C * np.maximum(counts, 1)),
                     dtype=torch.float32, device=dev)

    net = DualEnergyNet(C, scale=args.scale, momentum=args.momentum).to(dev)
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
        ri = rpool[torch.randint(len(rpool), (n_real,), generator=g)]
        hi = torch.randint(len(H), (args.hole_slots,), generator=g)
        xb = torch.cat([Xtr[ri], H[hi]]).to(dev)
        yb = torch.from_numpy(
            y_fit[np.searchsorted(fit_src, ri.numpy())]).to(dev)

        h = net.features(xb)
        # centres update from the REAL half only, before the readout, so the
        # loss sees the same centres the evaluation will
        net.update_centres(h[:n_real].detach(), yb)
        f = net(None, h=h)
        e = -torch.logsumexp(f, dim=1)

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
                        "head": "dual", "scale": args.scale,
                        "momentum": args.momentum, "holes": args.holes.name,
                        "config": {k: str(v) for k, v in vars(args).items()},
                        "log": log}, args.out)

    r = log[-1]
    print("")
    print(f"  saved {args.out.name}")
    print("")
    print("  " + "=" * 66)
    cols = list(prev) + ["ARM E"]
    print(f"  {'':<14}" + "".join(f"{c:>12}" for c in cols))
    print("  " + "-" * 66)
    for lab, key, fmt in (("val accuracy", "val_acc", "{:>11.1f}%"),
                          ("OOD AUROC", "ood", "{:>12.3f}"),
                          ("inverted", "inverted", "{:>12}")):
        vals = [prev[c][key] for c in prev] + [r[key]]
        print(f"  {lab:<14}" + "".join(fmt.format(v) for v in vals))
    print("  " + "=" * 66)
    print("")
    if "ARM D" in prev:
        pd = prev["ARM D"]["per_category"]
        print("  the categories arm D got backwards")
        print(f"    {'category':<14}{'arm D':>9}{'arm E':>9}{'change':>10}")
        for nm in sorted(pd, key=lambda z: pd[z])[:8]:
            print(f"    {nm:<14}{pd[nm]:>9.3f}{r['per_category'][nm]:>9.3f}"
                  f"{r['per_category'][nm]-pd[nm]:>+10.3f}")
    print("")
    print("  Validation numbers. Every unseen category has been examined")
    print("  during development, so these motivate rather than settle. The")
    print("  clean number is the official TEST split, still unopened.")


if __name__ == "__main__":
    main()
