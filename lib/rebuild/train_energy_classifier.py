"""Energy-based classifier: one conditioned energy field per class.

    cloud -> PN++ -> C outputs -> f_y(x) = -E(x,y),  E(x) = -logsumexp_y f_y(x)

    classify   argmin_y E(x, y)
    reject     E(x) above a threshold set on validation

WHY RAW OUTPUTS AND NOT log_softmax
The stock pointnet2_cls_ssg.get_model ends in log_softmax, which normalises the
outputs and throws away their overall magnitude -- and that magnitude is the
entire OOD signal:

    logits [ 8,  2,  1]  -> softmax [0.99, 0.006, 0.002]
    logits [ 1, -5, -6]  -> softmax [0.99, 0.006, 0.002]   identical

Same class, same confidence, but the second cloud is far less real. logsumexp
keeps that (-8.0 vs -1.0); softmax discards it. This model therefore reuses the
exact SA layers and fc head of get_model but returns fc3 directly. Softmax
appears only inside cross-entropy during training, never at inference.

THE TRAINING DATA IS A LADDER, NOT TWO PILES
From make_ladder.py, every sample carries (source_id, class, direction,
distance):

    FLOOR   real + descend 0.05/0.10/0.15    valid objects of their class
    WALL    ascend 0.05 .. 0.20              progressively off-support

Descend samples sit on the FLOOR, not on a slope: the crowd test measured them
at 0.95-1.09x the nearest-neighbour spacing of real objects, i.e. exactly where
real objects sit. They widen the floor rather than climbing it.

Ascend rungs are NOT pre-labelled negative. The ordering loss uses the distance
tag directly, so where a rung lands on the slope is measured, not decided in
advance. The usable outer limit is then read off the trained model rather than
fixed by a threshold chosen beforehand.

THE THREE LOSS TERMS
    L_cls    CE over -E(x,.) on FLOOR only. Pulls the true class energy down
             and pushes every other class up in one expression:
                 CE = E(x,y) - (-logsumexp_c -E(x,c))
             Its gradient is (p_c - 1[c=y]), so wrong classes the model
             currently finds plausible are pushed hardest -- hard negatives are
             weighted automatically. Ascend samples are excluded: they are not
             members of any class, so demanding a label from them is wrong.

    L_order  marginal energy rises with distance, each riser pinned to
             m * (distance gap) from BOTH sides with slack `tol`. A one-sided
             hinge would only enforce a MINIMUM slope and would be satisfied
             just as well by one large jump as by an even staircase -- which
             is what the first version of this file did, and the measured
             risers came out flattening (+1.03, +0.79, +0.50) rather than
             even. Applied to per-rung batch MEANS, not individual pairs,
             because samples within a rung genuinely vary.

    L_dE     max(||E(far) - E(floor)|| - gamma, 0). L_order pushes energies
             apart; this caps the spread. Together they give a bounded slope.
             From Delta Energy Regularizer, arXiv 2505.22486, which finds
             unbounded delta-energy growth is the signature of catastrophic
             overfitting in adversarially trained energy models.

NO LANGEVIN IN THIS LOOP
Standard EBM classifier training runs an MCMC chain every step. Here the
negatives are pre-generated in latent space and decoded, so this is ordinary
supervised training. The justification is measured rather than assumed: our
Table 1 found a cloud-space EBM (GPointNet, COV-CD 30.63) well behind the same
model sampled in latent space (45.64), so sampling in 3072 coordinate
dimensions is the harder problem.

SPLIT
Sources are split by source_id: an object and every sample derived from it stay
on the same side, or a perturbed copy of a training object leaks into
validation. The official test split is not touched here.
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


class EnergyNet(nn.Module):
    """PN++ with the log_softmax removed, so fc3 is -E(x, y) directly."""

    def __init__(self, n_classes):
        super().__init__()
        from pointnet2_cls_ssg import get_model
        self.b = get_model(num_class=n_classes, normal_channel=False)

    def forward(self, x):
        """x: (B, N, 3) -> (B, C) of f_y(x) = -E(x, y).
        With SHUFFLE_POINTS=1 the points are put in one fixed random order
        first: PointNet++'s ball query keeps the first 32 points in a ball, so
        sorted (canonically ordered) clouds bias the groups once the clouds are
        dense (2,048 points). Default off: unchanged."""
        import os
        if os.environ.get("SHUFFLE_POINTS"):
            n = x.shape[1]
            if getattr(self, "_perm", None) is None or len(self._perm) != n:
                self._perm = torch.randperm(n, generator=torch.Generator().manual_seed(0))
            x = x[:, self._perm.to(x.device)]
        b = self.b
        xyz = x.transpose(2, 1)
        l1_xyz, l1_points = b.sa1(xyz, None)
        l2_xyz, l2_points = b.sa2(l1_xyz, l1_points)
        _, l3_points = b.sa3(l2_xyz, l2_points)
        h = l3_points.view(len(x), 1024)
        h = b.drop1(F.relu(b.bn1(b.fc1(h))))
        h = b.drop2(F.relu(b.bn2(b.fc2(h))))
        return b.fc3(h)

    def energy(self, x):
        """Marginal energy E(x) = -logsumexp_y f_y(x). Low = looks real."""
        return -torch.logsumexp(self(x), dim=1)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ladder", type=Path,
                    default=ROOT / "ladder_easy5_fit_s0_annotated.npz")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--sources-per-batch", type=int, default=8,
                    help="batches are grouped by source object so every rung "
                         "of a source appears together; the ordering loss "
                         "needs them side by side")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--margin", type=float, default=2.0,
                    help="m: energy per unit of normalised distance, the slope")
    ap.add_argument("--tol", type=float, default=0.05,
                    help="slack on each riser before the two-sided ordering "
                         "term activates. MUST be smaller than the intended "
                         "rise m*dd, or the constraint is vacuous in the "
                         "direction that matters: with m=2, dd=0.05 the target "
                         "rise is 0.10, so tol=0.15 would leave relu(|got-0.10| "
                         "-0.15) at zero for got as low as -0.05 -- an energy "
                         "DROP costing nothing. tol=0.05 pins each riser to "
                         "[0.05, 0.15], strictly positive.")
    ap.add_argument("--gamma", type=float, default=8.0,
                    help="allowed total spread between floor and the far rung")
    ap.add_argument("--lam-order", type=float, default=1.0)
    ap.add_argument("--lam-de", type=float, default=1.0)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    out = args.out or (ROOT / f"energycls_s{args.seed}.pth")
    if out.exists() and not args.overwrite:
        raise SystemExit(f"refusing to overwrite {out}; pass --overwrite")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    import pointnet2_features as P
    with (P.DATA_ROOT / "modelnet40_shape_names.txt").open() as handle:
        names = [line.rstrip() for line in handle]

    d = np.load(args.ladder, allow_pickle=True)
    if "clouds" not in d:
        raise SystemExit(f"{args.ladder.name} holds latents; regenerate with "
                         "--store clouds (the classifier consumes clouds)")
    X = torch.from_numpy(d["clouds"]).float()
    y40 = d["labels"]
    src = d["source_id"]
    direction = d["direction"]
    dist = d["distance"].astype(np.float32)

    classes = sorted(set(int(v) for v in y40))
    remap = {c: i for i, c in enumerate(classes)}
    y = np.array([remap[int(v)] for v in y40], dtype=np.int64)
    C = len(classes)

    # ROLE, NOT DIRECTION. annotate_ladder.py assigns each rung floor or wall
    # from its MEASURED crowd ratio, because direction is not a reliable proxy:
    # on the easy-5 ladder monitor's descend rungs leave the crowd (1.20-1.64x,
    # further out than chair's ascend at the same rung) while bed's near ascend
    # rungs are still inside it (1.09-1.17x). Treating monitor's descend as
    # valid class members would teach the model that off-support monitors are
    # real. Falls back to direction if the ladder has not been annotated.
    if "role" in d:
        role = d["role"].astype(str)
        floor = role == "floor"
        print("  roles read from the ladder annotation (measured crowd ratio)")
    else:
        floor = (direction == "real") | (direction == "descend")
        print("  WARNING: ladder is not annotated; falling back to direction, "
              "which mislabelled 8 of 45 rungs on the easy-5 set. "
              "Run annotate_ladder.py first.")
    # Wall rungs keep their distance tag; the ordering term uses it directly.
    rungs = sorted(set(float(v) for v in dist[~floor]))

    # ---- split by SOURCE, never by row
    uniq = np.array(sorted(set(int(v) for v in src)))
    rng.shuffle(uniq)
    n_val = max(1, int(len(uniq) * args.val_frac))
    val_src = set(uniq[:n_val].tolist())
    is_val = np.array([int(v) in val_src for v in src])

    print("=" * 84)
    print("ENERGY CLASSIFIER   f_y(x) = -E(x,y),  E(x) = -logsumexp_y f_y(x)")
    print("=" * 84)
    print(f"  {args.ladder.name}: {len(X):,} clouds, {C} classes")
    for c in classes:
        m = y40 == c
        print(f"    {names[c]:<14}{int(m.sum()):>6} rows, "
              f"{len(set(int(v) for v in src[m])):>5} source objects")
    print(f"  floor rows {int(floor.sum()):,}   wall rows "
          f"{int((~floor).sum()):,}")
    print(f"  wall rungs {[round(r, 2) for r in rungs]}")
    print(f"  split by source: {len(uniq) - n_val} train / {n_val} val objects")
    print(f"  margin m={args.margin}  gamma={args.gamma}  "
          f"lam_order={args.lam_order}  lam_dE={args.lam_de}")
    print("")

    # Class weights: object counts are very uneven (chair 622 vs desk 140),
    # so without this the larger classes dominate every field.
    counts = np.array([ (y[floor & ~is_val] == i).sum() for i in range(C) ],
                      dtype=np.float64)
    w = torch.tensor(counts.sum() / (C * np.maximum(counts, 1)),
                     dtype=torch.float32, device=dev)

    by_src = {}
    for i, s in enumerate(src):
        by_src.setdefault(int(s), []).append(i)
    train_src = [s for s in uniq if s not in val_src]
    val_rows = np.where(is_val)[0]

    net = EnergyNet(C).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    print(f"  {sum(p.numel() for p in net.parameters()):,} parameters")
    print("")

    def batch_rows(sources):
        return np.concatenate([by_src[int(s)] for s in sources])

    @torch.no_grad()
    def evaluate():
        net.eval()
        acc_n = acc_ok = 0
        E_by = {("floor", 0.0): []}
        for t in rungs:
            E_by[("ascend", t)] = []
        for i in range(0, len(val_rows), 64):
            idx = val_rows[i:i + 64]
            xb = X[idx].to(dev)
            f = net(xb)
            e = -torch.logsumexp(f, dim=1)
            fl = floor[idx]
            if fl.any():
                pred = f[torch.from_numpy(fl).to(dev)].argmax(1).cpu().numpy()
                acc_ok += int((pred == y[idx][fl]).sum())
                acc_n += int(fl.sum())
                E_by[("floor", 0.0)].append(e[torch.from_numpy(fl).to(dev)].cpu())
            for t in rungs:
                m = (~floor[idx]) & (np.abs(dist[idx] - t) < 1e-6)
                if m.any():
                    E_by[("ascend", t)].append(e[torch.from_numpy(m).to(dev)].cpu())
        net.train()
        means = {k: float(torch.cat(v).mean()) if v else float("nan")
                 for k, v in E_by.items()}
        return (acc_ok / max(acc_n, 1)) * 100, means

    log, t0 = [], time.time()
    steps = max(1, len(train_src) // args.sources_per_batch)
    print(f"  {'epoch':>6}{'loss':>9}{'cls':>9}{'order':>9}{'dE':>8}"
          f"{'val acc':>9}{'E floor':>9}{'E far':>8}{'min':>7}")
    print("  " + "-" * 76)

    for ep in range(1, args.epochs + 1):
        perm = rng.permutation(len(train_src))
        tot = {"loss": 0.0, "cls": 0.0, "order": 0.0, "de": 0.0}
        for st in range(steps):
            sel = [train_src[i] for i in
                   perm[st * args.sources_per_batch:
                        (st + 1) * args.sources_per_batch]]
            if not sel:
                continue
            idx = batch_rows(sel)
            xb = X[idx].to(dev)
            yb = torch.from_numpy(y[idx]).to(dev)
            f = net(xb)
            e = -torch.logsumexp(f, dim=1)

            fl = torch.from_numpy(floor[idx]).to(dev)
            l_cls = (F.cross_entropy(f[fl], yb[fl], weight=w)
                     if fl.any() else f.sum() * 0)

            # --- ordering on per-rung means
            e_floor = e[fl].mean() if fl.any() else e.mean() * 0
            rung_means, rung_d = [], []
            for t in rungs:
                m = torch.from_numpy(
                    (~floor[idx]) & (np.abs(dist[idx] - t) < 1e-6)).to(dev)
                if m.any():
                    rung_means.append(e[m].mean())
                    rung_d.append(t)
            # TWO-SIDED. A one-sided hinge penalises only a rise SMALLER
            # than m*dd and is indifferent to a larger one, so a single huge
            # jump satisfies it exactly as well as an even staircase -- it
            # enforces a minimum slope, not even risers. Penalising
            # |actual - target| instead pins each riser to m*dd from both
            # sides, with `tol` leaving room for genuine variation.
            l_order = f.sum() * 0
            if rung_means:
                want = args.margin * rung_d[0]
                got = rung_means[0] - e_floor
                l_order = l_order + F.relu((got - want).abs() - args.tol)
                for i in range(len(rung_means) - 1):
                    want = args.margin * (rung_d[i + 1] - rung_d[i])
                    got = rung_means[i + 1] - rung_means[i]
                    l_order = l_order + F.relu((got - want).abs() - args.tol)
            l_de = (F.relu((rung_means[-1] - e_floor).abs() - args.gamma)
                    if rung_means else f.sum() * 0)

            loss = l_cls + args.lam_order * l_order + args.lam_de * l_de
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot["loss"] += float(loss.detach())
            tot["cls"] += float(l_cls.detach())
            tot["order"] += float(l_order.detach())
            tot["de"] += float(l_de.detach())
        sched.step()

        acc, means = evaluate()
        row = {"epoch": ep, **{k: v / steps for k, v in tot.items()},
               "val_acc": acc,
               "E": {f"{k[0]}_{k[1]}": v for k, v in means.items()},
               "minutes": (time.time() - t0) / 60}
        log.append(row)
        print(f"  {ep:>6}{tot['loss']/steps:>9.3f}{tot['cls']/steps:>9.3f}"
              f"{tot['order']/steps:>9.3f}{tot['de']/steps:>8.3f}"
              f"{acc:>8.1f}%{means[('floor',0.0)]:>9.2f}"
              f"{means[('ascend',rungs[-1])]:>8.2f}"
              f"{row['minutes']:>7.1f}", flush=True)
        torch.save({"state_dict": net.state_dict(), "classes": classes,
                    "class_names": [names[c] for c in classes],
                    "rungs": rungs, "config": {k: str(v) for k, v in
                                               vars(args).items()},
                    "val_sources": sorted(val_src), "log": log}, out)

    print("")
    print(f"  saved {out.name}")
    print("")
    print("  READ THE ENERGY COLUMNS, NOT ONLY ACCURACY")
    print("  E floor should sit well below E far, and the gap should be")
    print("  stable rather than growing without bound -- unbounded growth is")
    print("  the overfitting signature the dE term exists to prevent.")
    print("  Next: per-class energy distributions and the OOD threshold.")


if __name__ == "__main__":
    main()
