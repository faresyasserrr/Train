"""Stage 0: the real-only classifier. Arm A of the control, and the gradient
source for stage 1.

WHY THIS RUNS FIRST
The generated samples are meant to be walked along the CLASSIFIER gradient --
pushed into the regions where the classifier is actually weak. That requires a
classifier to exist before any sample is generated. Training one on real clouds
alone provides that gradient, and it is simultaneously the missing control arm:
without it, the accuracy and OOD numbers of a ladder-trained model cannot be
attributed to the ladder rather than to the real objects.

    arm A   this script        real clouds only
    arm B   stage 2            real clouds + classifier-guided samples

Same architecture, same seed, same optimiser steps in both. Only the rows differ.

NO LADDER, SO NO ORDERING LOSS
L_order and L_dE are defined over rungs. With no generated rows there are no
rungs, so this arm trains on cross-entropy alone. That is the point: arm A must
be the model you get from real data and nothing else.

SPLIT
Objects come from the ModelNet TRAIN split and are divided by object index into
fit and validation. The official test split is never opened here. The saved
val_sources list is what stage 1 and stage 2 must reuse, or the comparison
leaks.

REPORTED
    class accuracy on validation real objects
    OOD AUROC over ModelNet categories absent from the trained set, drawn from
      the TRAIN split so the test set stays spent
    the count of those categories scoring below 0.5 -- an inversion, where
      unknown objects score LOWER energy than known ones
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
PARENT = ROOT.parent
REPO = PARENT / "Pointnet_Pointnet2_pytorch"
for _p in (str(ROOT), str(PARENT), str(REPO), str(REPO / "models")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from train_energy_classifier import EnergyNet                    # noqa: E402


def auroc(pos, neg):
    """P(a random neg scores higher than a random pos). 0.5 = no separation."""
    a = np.concatenate([pos, neg])
    r = a.argsort().argsort().astype(np.float64)
    rp = r[:len(pos)].sum()
    n1, n2 = len(pos), len(neg)
    u = rp - n1 * (n1 - 1) / 2
    return 1.0 - u / (n1 * n2)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--classes", type=int, nargs="+", default=[8, 22, 30],
                    help="ModelNet40 indices; default chair monitor sofa. "
                         "chair/sofa are the confusable pair and monitor is "
                         "distinct, so the set spans both cases.")
    ap.add_argument("--steps", type=int, default=2000,
                    help="optimiser steps; arm B MUST use the same number")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--per-class", type=int, default=0,
                    help="cap the FIT objects per class (0 = use all). The "
                         "full-data run reached 100%% accuracy and OOD 0.964, "
                         "leaving nothing for generated negatives to add; a "
                         "cap puts the comparison in the low-data regime where "
                         "extra negatives are supposed to matter. Validation "
                         "is never capped, so runs stay comparable.")
    ap.add_argument("--spectral", action="store_true",
                    help="wrap every Linear and Conv in spectral "
                         "normalisation, bounding the network's Lipschitz "
                         "constant. Du & Mordatch (NeurIPS 2019) argue a "
                         "Langevin chain cannot navigate a rough energy "
                         "surface and constrain the network so that it can; "
                         "it is standard in the EBM sampling literature and "
                         "is the one ingredient of that recipe this project "
                         "has never used. Testing it here is narrow on "
                         "purpose: the question is whether the chain's hole "
                         "rate rises from 34% toward PGD's 98% on a smoother "
                         "surface. There is a measured reason to expect a "
                         "cost -- nine runs established that smoothing this "
                         "energy reduces detection, because the steep rise "
                         "away from the data IS the detector. So this tests "
                         "a tension rather than an unexplored trick, and the "
                         "sampler diagnostic decides it before any pipeline "
                         "is rebuilt on top.")
    ap.add_argument("--cls-loss", choices=("ce", "margin"), default="ce",
                    help="'ce': softmax cross-entropy. 'margin': no softmax. "
                         "Every wrong class's energy must sit --cls-margin "
                         "above the true class's (Weston & Watkins 1999), "
                         "the same classification loss as the margin head "
                         "in stage45_rank.py")
    ap.add_argument("--cls-margin", type=float, default=9.7)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--ood-cap", type=int, default=120,
                    help="clouds per unseen category, to keep eval quick")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    out = args.out or (ROOT / f"stage0_real_s{args.seed}.pth")
    if out.exists() and not args.overwrite:
        raise SystemExit(f"refusing to overwrite {out}; pass --overwrite")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    import pointnet2_features as P
    P.patch_deterministic_fps(True)
    from decoder import load_clouds
    with (P.DATA_ROOT / "modelnet40_shape_names.txt").open() as handle:
        names = [line.rstrip() for line in handle]

    classes = sorted(int(c) for c in args.classes)
    C = len(classes)
    remap = {c: i for i, c in enumerate(classes)}

    Xtr, ytr = load_clouds("train")
    Xtr = torch.as_tensor(np.asarray(Xtr)).float()
    ytr = np.asarray(ytr)

    keep = np.where(np.isin(ytr, classes))[0]
    rng.shuffle(keep)
    n_val = max(1, int(len(keep) * args.val_frac))
    val_src = np.array(sorted(keep[:n_val].tolist()))
    fit_src = np.array(sorted(keep[n_val:].tolist()))
    # LOW-DATA REGIME. The cap applies to the FIT set only; the validation
    # objects are chosen first, from the same seed and the same val_frac, so a
    # capped run is measured against the IDENTICAL validation set as the full
    # run. Shrinking the measurement set too would confound "less training
    # data" with "noisier evaluation".
    if args.per_class > 0:
        sel = []
        for c in classes:
            have = fit_src[ytr[fit_src] == c]
            sel.append(have[:args.per_class])
        fit_src = np.array(sorted(np.concatenate(sel).tolist()))

    y_fit = np.array([remap[int(v)] for v in ytr[fit_src]])
    y_val = np.array([remap[int(v)] for v in ytr[val_src]])
    X_val = Xtr[val_src]

    unseen = [c for c in sorted(set(int(v) for v in ytr))
              if c not in classes and (ytr == c).sum() >= 40]

    print("=" * 78)
    print("STAGE 0   real-only energy classifier  (arm A, and the gradient "
          "source)")
    print("=" * 78)
    print(f"  classes   {[names[c] for c in classes]}")
    for c in classes:
        n_f = int((ytr[fit_src] == c).sum())
        n_v = int((ytr[val_src] == c).sum())
        print(f"    {names[c]:<12}{n_f:>5} fit  {n_v:>4} val")
    print(f"  total     {len(fit_src)} fit / {len(val_src)} val real objects")
    print(f"  {args.steps} steps, batch {args.batch}, lr {args.lr}, "
          f"seed {args.seed}")
    print("  loss      cross-entropy only (no rungs exist in this arm)")
    print(f"  OOD       {len(unseen)} unseen categories from the TRAIN split")
    print("")

    # Class counts are uneven, so weight the CE or the larger class dominates
    # every energy field.
    counts = np.array([(y_fit == i).sum() for i in range(C)], dtype=np.float64)
    w = torch.tensor(counts.sum() / (C * np.maximum(counts, 1)),
                     dtype=torch.float32, device=dev)

    net = EnergyNet(C).to(dev)
    if args.spectral:
        from torch.nn.utils import spectral_norm
        n_wrapped = 0
        for mod in net.modules():
            for name, child in list(mod.named_children()):
                if isinstance(child, (torch.nn.Linear, torch.nn.Conv1d,
                                      torch.nn.Conv2d)):
                    setattr(mod, name, spectral_norm(child))
                    n_wrapped += 1
        print(f"  spectral  normalised {n_wrapped} Linear/Conv layers")
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)

    @torch.no_grad()
    def energies(X, b=48):
        e = []
        for i in range(0, len(X), b):
            e.append((-torch.logsumexp(net(X[i:i + b].to(dev)), 1)).cpu())
        return torch.cat(e).numpy()

    @torch.no_grad()
    def logits(X, b=48):
        f = []
        for i in range(0, len(X), b):
            f.append(net(X[i:i + b].to(dev)).cpu())
        return torch.cat(f)

    def evaluate():
        net.eval()
        acc = float((logits(X_val).argmax(1).numpy() == y_val).mean()) * 100
        e_in = energies(X_val)
        per, pool = {}, []
        for c in unseen:
            idx = np.where(ytr == c)[0][:args.ood_cap]
            eo = energies(Xtr[idx])
            per[names[c]] = auroc(e_in, eo)
            pool.append(eo)
        net.train()
        return acc, auroc(e_in, np.concatenate(pool)), per

    g = torch.Generator().manual_seed(args.seed)
    pool = torch.from_numpy(fit_src)
    yf = torch.from_numpy(y_fit)
    t0 = time.time()
    log = []
    print(f"  {'step':>7}{'loss':>9}{'val acc':>10}{'OOD':>8}{'inv':>6}"
          f"{'min':>7}")
    print("  " + "-" * 47)
    run = 0.0
    for st in range(1, args.steps + 1):
        sel = torch.randint(len(pool), (args.batch,), generator=g)
        xb = Xtr[pool[sel]].to(dev)
        yb = yf[sel].to(dev)
        if args.cls_loss == "margin":
            f = net(xb)
            f_true = f.gather(1, yb[:, None]).squeeze(1)
            is_true = F.one_hot(yb, f.shape[1]).bool()
            viol = F.relu(args.cls_margin - (f_true[:, None] - f))
            hinge = viol.masked_fill(is_true, 0.0).sum(1) / (f.shape[1] - 1)
            loss = (hinge * w[yb]).sum() / w[yb].sum()
        else:
            loss = F.cross_entropy(net(xb), yb, weight=w)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        run += float(loss.detach())
        if st % args.eval_every == 0 or st == args.steps:
            acc, ood, per = evaluate()
            inv = sum(1 for v in per.values() if v < 0.5)
            mins = (time.time() - t0) / 60
            log.append({"step": st, "loss": run / args.eval_every,
                        "val_acc": acc, "ood": ood, "inverted": inv,
                        "per_category": per, "minutes": mins})
            print(f"  {st:>7}{run/args.eval_every:>9.4f}{acc:>9.1f}%"
                  f"{ood:>8.3f}{inv:>6}{mins:>7.1f}", flush=True)
            run = 0.0
            torch.save({"state_dict": net.state_dict(), "classes": classes,
                        "class_names": [names[c] for c in classes],
                        "val_sources": val_src.tolist(),
                        "fit_sources": fit_src.tolist(),
                        "spectral": bool(args.spectral),
                "config": {k: str(v) for k, v in vars(args).items()},
                        "log": log}, out)

    print("")
    print(f"  saved {out.name}")
    print("")
    acc, ood, per = log[-1]["val_acc"], log[-1]["ood"], log[-1]["per_category"]
    print("  ARM A RESULT   the baseline any generated sample must beat")
    print(f"    val accuracy        {acc:.1f}%")
    print(f"    OOD AUROC           {ood:.3f}   over {len(per)} categories")
    print(f"    inverted (<0.5)     {sum(1 for v in per.values() if v < 0.5)}")
    print("")
    print("  worst 6 categories")
    for nm in sorted(per, key=lambda z: per[z])[:6]:
        print(f"    {nm:<14}{per[nm]:>8.3f}")
    print("")
    print("  These are arm A numbers. Stage 2 must run at the SAME number of")
    print("  steps and the SAME val_sources, or the comparison measures")
    print("  compute or split rather than the generated samples.")


if __name__ == "__main__":
    main()
