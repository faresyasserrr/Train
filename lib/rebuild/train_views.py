"""Controlled trial: does FPS view variation improve local distribution learning?

ONE HYPOTHESIS, ONE VARIABLE
----------------------------
    control     every object is represented by its canonical feature
    augment     every object is represented by ONE of its 5 views, drawn fresh
                each time it is sampled (canonical + 4 FPS starts)

Everything else is held identical: the same objects, the same stratified
split, the same seed, the same iteration count, the same architecture, the same
sampler, the same sd, and the same transform.npz. The PCA is deliberately NOT
refit -- doing that here would introduce a second variable, and it is a
separate adoption step if this screen wins.

THE SAMPLE COUNT DOES NOT CHANGE
--------------------------------
Each draw picks an OBJECT, then picks one of its views. The augmented arm sees
exactly as many objects per iteration as the control. Views are correlated, so
treating five siblings as five independent examples would be wrong; this
smooths which vector represents an object rather than inflating the dataset.

THE RNG TRAP
------------
The augmented arm needs one extra random number per sample -- which view. Drawn
from the main generator, it would desynchronise every subsequent Langevin draw,
and the two arms would then differ by the augmentation AND by their entire
random trajectory. This is the same failure that made the first conditioning
screen uncontrolled.

So view selection uses its own CPU generator. The main stream stays
bit-identical between arms, which `--verify-rng` checks directly: with views
forced to canonical, the augmented arm must reproduce the control's first
iteration exactly.

SPLIT DISCIPLINE
----------------
The split is over OBJECTS, stratified by class. All five views of an object
land on the same side. A sibling crossing the boundary would be the most
direct leakage available, and it would make every generative metric here
meaningless.

sd IS SHARED
------------
Both arms use the sd of the CONTROL's canonical fit features. Letting the
augmented arm compute its own sd would change the sampler's noise scale and
preconditioning, which is a second variable.

EXPECTED TRADE-OFF, LOGGED IN ADVANCE
-------------------------------------
A smoothed training distribution should push mild-corruption AUC further toward
0.5, because perturbations at that scale become genuinely in-distribution. That
is predicted here, before any number exists, so it is read as the cost of the
trade rather than explained away afterwards.

Run:  python rebuild/train_views.py --arm control
      python rebuild/train_views.py --arm augment
      python rebuild/train_views.py --verify-rng
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from transform import Transform  # noqa: E402
from train_conditional import (  # noqa: E402
    BANK, DATA_EPSILON, EMA, LR, PER_CLASS, TRAIN_STEPS, VAL_FRAC, dev,
    langevin,
)
from train_combined import CombinedEnergy  # noqa: E402

VIEWS = ROOT / "views.npz"
LAMBDA = 1.0
BETA = 1.0


def load(seed):
    """(n_obj, n_views, k) encoded features, labels, and the object split.

    View 0 is canonical (bank.npz); views 1..4 are the FPS starts in views.npz.
    """
    t = Transform.load()
    bank = np.load(BANK, allow_pickle=True)
    canon = bank["train_features"].astype(np.float64)
    y = bank["train_labels"].astype(np.int64)

    vd = np.load(VIEWS, allow_pickle=True)
    if int(vd["n_train"]) != len(canon):
        raise ValueError("views.npz object count does not match bank.npz")
    raw = np.concatenate([canon[:, None, :], vd["view_features"]], axis=1)
    n_obj, n_views, _ = raw.shape
    Z = t.encode(raw.reshape(-1, raw.shape[-1])).astype(np.float32)
    Z = Z.reshape(n_obj, n_views, -1)

    # Split over OBJECTS, stratified. Views follow their object.
    obj = np.arange(n_obj)
    fit_obj, val_obj = train_test_split(
        obj, test_size=VAL_FRAC, random_state=seed, stratify=y)
    return t, Z, y, np.sort(fit_obj), np.sort(val_obj), n_views


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", choices=("control", "augment"), default="control")
    ap.add_argument("--iters", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=500)
    ap.add_argument("--verify-rng", action="store_true",
                    help="run one iteration of each arm with views forced to "
                         "canonical; the two must match bit-exactly")
    args = ap.parse_args()

    t, Z, y, fit_obj, val_obj, n_views = load(args.seed)
    dim = t.k
    Zt = torch.from_numpy(Z).to(dev)
    yfit = torch.from_numpy(y[fit_obj]).long().to(dev)
    fit_t = torch.from_numpy(fit_obj).long().to(dev)

    # sd from the CONTROL's canonical fit features, shared by both arms.
    sd = Zt[fit_t, 0].std(0)
    data_scale = (DATA_EPSILON * sd)[None, :]
    by_class = [torch.where(yfit == c)[0].to(dev) for c in range(40)]
    labels = torch.arange(40, device=dev).repeat_interleave(PER_CLASS)
    batch = len(labels)

    def build(arm, seed):
        torch.manual_seed(seed)
        net = CombinedEnergy(dim, lam=LAMBDA).to(dev)
        ema = EMA(net)
        opt = torch.optim.Adam(net.parameters(), lr=LR)
        torch.manual_seed(seed + 1000)          # after construction
        vg = torch.Generator().manual_seed(seed + 2000)   # views only
        return net, ema, opt, vg

    def pick(obj_local, vg, arm, force_canonical=False):
        """obj_local: indices into fit_obj. Returns encoded features."""
        rows = fit_t[obj_local]
        if arm == "control" or force_canonical:
            v = torch.zeros(len(rows), dtype=torch.long, device=dev)
        else:
            v = torch.randint(0, n_views, (len(rows),),
                              generator=vg).to(dev)
        return Zt[rows, v]

    def step(net, opt, ema, vg, arm, force_canonical=False):
        idx_c = torch.cat([g[torch.randint(len(g), (PER_CLASS,), device=dev)]
                           for g in by_class])
        real_c = pick(idx_c, vg, arm, force_canonical) \
            + data_scale * torch.randn(batch, dim, device=dev)
        idx_g = torch.randint(len(fit_t), (batch,), device=dev)
        real_g = pick(idx_g, vg, arm, force_canonical) \
            + data_scale * torch.randn(batch, dim, device=dev)

        x0g = torch.randn(batch, dim, device=dev) * sd
        fake_g = langevin(lambda x: net.global_energy(x), x0g, TRAIN_STEPS, sd,
                          decay=False)
        x0c = torch.randn(batch, dim, device=dev) * sd
        fake_c = langevin(lambda x: net(x, labels), x0c, TRAIN_STEPS, sd,
                          decay=False)

        gr = net.global_energy(real_g)
        gc, cr = net.both(real_c, labels)
        gfg = net.global_energy(fake_g)
        gfc, cfc = net.both(fake_c, labels)
        loss_global = gr.mean() - gfg.mean()
        loss_class = (gc + net.lam * cr).mean() - (gfc + net.lam * cfc).mean()
        loss = BETA * loss_global + loss_class

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        ema.update(net)
        return loss_global, loss_class, gr, cr, fake_g, fake_c

    if args.verify_rng:
        out = {}
        for arm in ("control", "augment"):
            net, ema, opt, vg = build(arm, args.seed)
            _, _, _, _, fg, fc = step(net, opt, ema, vg, arm,
                                      force_canonical=True)
            out[arm] = (fg.norm(dim=1).mean().item(),
                        fc.norm(dim=1).mean().item())
        print("RNG match check -- views forced to canonical, one iteration:")
        for arm, (a, b) in out.items():
            print(f"  {arm:<9} |fake_g| {a:.10f}   |fake_c| {b:.10f}")
        same = out["control"] == out["augment"]
        print(f"\n  {'MATCHED' if same else 'DIVERGED'} -- the arms "
              f"{'share' if same else 'DO NOT share'} a random trajectory")
        if not same:
            raise SystemExit("view selection is leaking into the main RNG")
        return

    net, ema, opt, vg = build(args.arm, args.seed)
    print(f"device {dev}  k={dim}  arm={args.arm}  seed={args.seed}")
    print(f"objects: {len(fit_obj)} fit / {len(val_obj)} val, split by object, "
          f"stratified; {n_views} views available")
    print(f"representation: "
          + ("canonical only" if args.arm == "control"
             else f"one of {n_views} views drawn per sample (separate RNG)"))
    print(f"combined v2  lambda={LAMBDA}  beta={BETA}  batch {batch}  "
          f"{TRAIN_STEPS} steps, two chains")
    print(f"sd from control canonical fit features (shared by both arms)")
    print(f"transform.npz unchanged -- PCA refit is a separate adoption step")
    print(f"data norm {Zt[fit_t, 0].norm(dim=1).mean():.2f}\n")

    t0 = time.time()
    for it in range(1, args.iters + 1):
        lg, lc, gr, cr, fg, fc = step(net, opt, ema, vg, args.arm)
        if it % args.log_every == 0 or it == 1:
            el = time.time() - t0
            print(f"  {it:>6}/{args.iters}  Lg {lg.item():+8.4f}  "
                  f"Lc {lc.item():+8.4f} | |Eg| {gr.abs().mean().item():6.3f}  "
                  f"|Ec| {cr.abs().mean().item():6.3f} | "
                  f"|fg| {fg.norm(dim=1).mean():6.2f}  "
                  f"|fc| {fc.norm(dim=1).mean():6.2f}   "
                  f"[{el/60:5.1f}m, ETA {el/it*(args.iters-it)/60:5.1f}m]")

    meta = {"dim": dim, "n_out": 40, "lam": LAMBDA, "beta": BETA, "alpha": 1.0,
            "iters": args.iters, "seed": args.seed, "arm": args.arm,
            "n_views": n_views if args.arm == "augment" else 1,
            "val_obj": val_obj, "trained_on": f"{len(fit_obj)} objects"}
    for name, m in ((f"views_{args.arm}", net),
                    (f"views_{args.arm}_ema", ema.shadow)):
        torch.save({"state_dict": m.state_dict(), **meta},
                   ROOT / f"{name}_s{args.seed}_{args.iters}.pth")
    print(f"\ndone in {(time.time()-t0)/60:.1f} min -- saved "
          f"views_{args.arm}[_ema]_s{args.seed}_{args.iters}.pth")


if __name__ == "__main__":
    main()
