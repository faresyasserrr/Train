"""Step 5c: combined EBM -- global realism plus class guidance.

WHAT THE THREE-SEED SCREEN ESTABLISHED
--------------------------------------
Conditioning helps generation and hurts corruption detection, both robustly:

    1-NN (cond - uncond, EMA)   -5.61 / -4.74 / -4.98   = -5.11 +/- 0.45
    per-class macro AUC          0.671 / 0.671 / 0.644, 0 of 21 inverted, all seeds
    corruption AUC at x0.5       0.665 vs 0.764 | 0.645 vs 0.751   (~-0.10)
    monotone% (cond - uncond)    -3.6 / -3.5

The mechanism: each class head sees 2 real and 2 fake per iteration, both from
its own class and both on or near that class's manifold. Nothing shows it a
mildly corrupted feature or another class's feature, so the near-manifold band
has no training signal. The single unconditional head sees all 40 classes and
therefore does get signal there.

THE MODEL
---------
    E_total(x, y) = E_global(x) + lambda * E_class(x, y)      lambda = 1.0

    shared trunk:  195 -> 1024 -> 1024 -> 512      SiLU
    head_g:        512 -> 1        general realism
    head_c:        512 -> 40       class-specific guidance

E_global sees all 80 samples in every batch across all 40 classes, so it gets
the near-manifold coverage the per-class heads lack. E_class supplies the class
steering that produced the 1-NN gain.

WHY TWO LOSSES AND NOT ONE
--------------------------
    loss = [E_total(real,y)  - E_total(fake,y)]      loss_total
         + [E_global(real)   - E_global(fake)]       loss_global

A single loss on E_total leaves the decomposition ambiguous: the class head
could learn everything while the global head learns almost nothing, and the
architecture alone would not prevent it. The explicit auxiliary term forces
E_global to be a realism model in its own right.

The same balanced fake batch serves both losses, so sampling cost does not
double -- and the negatives come from E_total, since that is the generative
model whose samples we care about.

MONITORING
----------
E_global and E_class are logged separately, with their own dE and gradient
norms, so the decomposition can be checked rather than assumed. If E_class
carries everything and E_global stays flat, the auxiliary loss is not doing its
job and that will be visible here.

WHAT WOULD JUSTIFY SEPARATE TRUNKS
-----------------------------------
Only evidence of interference -- global corruption sensitivity still weak
despite its explicit auxiliary loss. Starting with separate trunks would double
cost before proving separation is necessary.

CONTINUE TO 20,000 ONLY IF THIS RETAINS ALL THREE
--------------------------------------------------
  1. the conditional model's 1-NN and coverage
  2. the unconditional model's corruption AUC
  3. correct per-class energy orientation

The conditional and unconditional checkpoints for the same seed are re-measured
in this run through the identical code path, so no comparison rests on a
recorded number.

Run:  python rebuild/train_combined.py
      python rebuild/train_combined.py --seed 1
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from transform import Transform  # noqa: E402
from train_conditional import (  # noqa: E402
    BANK, DATA_EPSILON, EMA, EMA_DECAY, EVAL_STEPS, LR, MIN_VAL, PER_CLASS,
    TRAIN_STEPS, VAL_FRAC, Energy, dev, evaluate, langevin,
)
from check_energy import (  # noqa: E402
    LEVELS, N, classwise_auc, generate, global_check, monotonicity, tag,
)

LAMBDA = 1.0


class CombinedEnergy(nn.Module):
    """E_total(x, y) = E_global(x) + lambda * E_class(x, y), shared trunk.

    `forward` returns E_total and `n_out` is 40, so this drops into the same
    evaluation code as the plain conditional model with no special casing.
    """

    def __init__(self, dim, n_classes=40, hidden=(1024, 1024, 512),
                 lam=LAMBDA):
        super().__init__()
        layers, d = [], dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.SiLU()]
            d = h
        self.trunk = nn.Sequential(*layers)
        self.head_g = nn.Linear(d, 1)
        self.head_c = nn.Linear(d, n_classes)
        self.lam = lam
        self.n_out = n_classes

    def both(self, x, y):
        h = self.trunk(x)
        e_g = self.head_g(h).squeeze(-1)
        e_c = self.head_c(h).gather(1, y[:, None]).squeeze(1)
        return e_g, e_c

    def forward(self, x, y):
        e_g, e_c = self.both(x, y)
        return e_g + self.lam * e_c

    def global_energy(self, x):
        return self.head_g(self.trunk(x)).squeeze(-1)


def train(args, Zfit, yfit, sd, dim):
    torch.manual_seed(args.seed)
    net = CombinedEnergy(dim, lam=args.lam).to(dev)
    ema = EMA(net)
    opt = torch.optim.Adam(net.parameters(), lr=LR)

    by_class = [torch.where(yfit == c)[0].to(dev) for c in range(40)]
    labels = torch.arange(40, device=dev).repeat_interleave(PER_CLASS)
    batch = len(labels)
    data_scale = (DATA_EPSILON * sd)[None, :]

    print(f"\ncombined: {sum(p.numel() for p in net.parameters()):,} parameters"
          f"   lambda={args.lam}")
    print(f"  loss = [E_total(real) - E_total(fake)] + "
          f"[E_global(real) - E_global(fake)]")

    # Reseed after construction: the heads consume random draws, so seeding
    # before the model is built leaves the RNG in a model-dependent state.
    torch.manual_seed(args.seed + 1000)

    t0 = time.time()
    for it in range(1, args.iters + 1):
        idx = torch.cat([g[torch.randint(len(g), (PER_CLASS,), device=dev)]
                         for g in by_class])
        real = Zfit[idx] + data_scale * torch.randn(batch, dim, device=dev)

        # negatives follow E_total -- that is the generative model
        x0 = torch.randn(batch, dim, device=dev) * sd
        fake = langevin(lambda x: net(x, labels), x0, TRAIN_STEPS, sd,
                        decay=False)

        gr, cr = net.both(real, labels)
        gf, cf = net.both(fake, labels)
        loss_total = (gr + net.lam * cr).mean() - (gf + net.lam * cf).mean()
        loss_global = gr.mean() - gf.mean()
        loss = loss_total + loss_global

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        ema.update(net)

        if it % args.log_every == 0 or it == 1:
            print(f"  {it:>6}/{args.iters}  L {loss.item():+8.4f}  "
                  f"Lt {loss_total.item():+8.4f}  Lg {loss_global.item():+8.4f} | "
                  f"dE_g {(gf - gr).mean().item():+8.4f}  "
                  f"dE_c {(cf - cr).mean().item():+8.4f} | "
                  f"|Eg| {gr.abs().mean().item():6.3f}  "
                  f"|Ec| {cr.abs().mean().item():6.3f}  "
                  f"|fake| {fake.norm(dim=1).mean():6.2f}   "
                  f"[{time.time() - t0:.0f}s]")
    print(f"  trained in {(time.time() - t0) / 60:.1f} min")
    return net, ema.shadow


def decomposition_report(net, real, y_real, gen, y_gen, sd, seed, label):
    """Report each head and directly test the global head under corruption."""
    with torch.no_grad():
        gr, cr = net.both(real, y_real)
        gf, cf = net.both(gen, y_gen)

    class GlobalView(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
            self.n_out = 1

        def forward(self, x):
            return self.model.global_energy(x)

    # dE_g can correctly be near zero when generated and real global marginals
    # match. Corruption AUC tests whether the head learned useful geometry.
    _, aucs, _ = monotonicity(GlobalView(net), real, y_real, sd, seed)
    print(f"  {label:<24}{gr.mean().item():>10.4f}{(gf - gr).mean().item():>+10.4f}"
          f"{cr.mean().item():>10.4f}{(cf - cr).mean().item():>+10.4f}"
          f"{gr.std().item():>10.4f}{cr.std().item():>10.4f}"
          f"{aucs[2]:>10.3f}{aucs[3]:>10.3f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--iters", type=int, default=5000)
    ap.add_argument("--log-every", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lam", type=float, default=LAMBDA)
    args = ap.parse_args()

    t = Transform.load()
    d = np.load(BANK, allow_pickle=True)
    Z = t.encode(d["train_features"].astype(np.float64)).astype(np.float32)
    y = d["train_labels"]
    Zf, Zv, yf, yv = train_test_split(
        Z, y, test_size=VAL_FRAC, random_state=args.seed, stratify=y)
    Zfit = torch.from_numpy(Zf).to(dev)
    yfit = torch.from_numpy(yf).long().to(dev)
    Zval, yval = torch.from_numpy(Zv), torch.from_numpy(yv).long()
    sd = Zfit.std(0)
    dim = t.k

    counts = np.bincount(yv, minlength=40)
    eligible = [c for c in range(40) if counts[c] >= MIN_VAL]
    cnt = torch.bincount(yval, minlength=40).float()
    torch.manual_seed(args.seed + 5000)
    y_gen = torch.multinomial(cnt / cnt.sum(), N, replacement=True).to(dev)
    idx = torch.randperm(len(Zval),
                         generator=torch.Generator().manual_seed(args.seed))[:N]
    real, y_real = Zval[idx].to(dev), yval[idx].to(dev)

    print(f"device {dev}  k={dim}  fit {tuple(Zfit.shape)}  val {tuple(Zval.shape)}"
          f"  seed {args.seed}  (test not read)")

    net, net_ema = train(args, Zfit, yfit, sd, dim)
    for name, m in [("combined", net), ("combined_ema", net_ema)]:
        torch.save({"state_dict": m.state_dict(), "n_out": m.n_out, "dim": dim,
                    "lam": args.lam, "iters": args.iters, "seed": args.seed},
                   ROOT / f"{name}_s{args.seed}_{args.iters}.pth")

    # every model re-measured here through the identical code path
    models = [("combined", net), ("combined_ema", net_ema)]
    for name in ("conditional", "conditional_ema",
                 "unconditional", "unconditional_ema"):
        path = ROOT / f"{name}_s{args.seed}_{args.iters}.pth"
        if not path.is_file():
            print(f"\n  note: {path.name} missing, skipping that comparison row")
            continue
        ck = torch.load(path, map_location=dev, weights_only=False)
        m = Energy(ck["dim"], ck["n_out"]).to(dev)
        m.load_state_dict(ck["state_dict"])
        m.eval()
        models.append((name, m))
    for _, m in models:
        m.eval()

    print("\n" + "=" * 94)
    print("A. GENERATION   (val, 125 steps, decay over last 5)")
    print(f"{'':<26}{'GLOBAL (N=2048, empirical labels)':^32} | "
          f"{'MACRO PER CLASS':^32}")
    print(f"  {'model':<26}{'1-NN':>8}{'COV':>8}{'Prec':>8}{'Rec':>8} | "
          f"{'1-NN':>8}{'COV':>8}{'Prec':>8}{'Rec':>8}")
    print("-" * 94)
    for name, m in models:
        evaluate(m, Zval, yval, sd, dim, args.seed, name, eligible)

    print("\nB. ENERGY ORIENTATION")
    print(f"  {'model':<24}{'E(real)':>9}{'E(gen)':>9}{'dE':>9}{'AUC':>8}"
          f"{'':>11}{'|g|real':>9}{'|g|gen':>9}")
    print("-" * 88)
    gens = {}
    for name, m in models:
        gens[name] = generate(m, y_gen, sd, dim, args.seed)
        er, eg, dE, a, gr_, gg = global_check(m, real, y_real, gens[name], y_gen)
        print(f"  {name:<24}{er:>9.4f}{eg:>9.4f}{dE:>+9.4f}{a:>8.3f}"
              f"{tag(a):>11}{gr_:>9.4f}{gg:>9.4f}")

    print("\nC. PER-CLASS ORIENTATION")
    print(f"  {'model':<24}{'macro AUC':>11}{'sd':>7}{'worst class':>13}"
          f"{'worst AUC':>11}{'# below 0.5':>13}")
    print("-" * 88)
    for name, m in models:
        if m.n_out == 1:
            print(f"  {name:<24}{'n/a -- unconditional':>40}")
            continue
        mm, ss, wc, wa, nb = classwise_auc(m, Zval, yval, sd, dim,
                                           args.seed, eligible)
        print(f"  {name:<24}{mm:>11.3f}{ss:>7.3f}{d['classes'][wc]:>13}"
              f"{wa:>11.3f}{nb:>7}/{len(eligible)}")

    print("\nD. CORRUPTION SENSITIVITY   (fixed direction, increasing magnitude)")
    head = "".join(f"{f'x{lv}':>9}" for lv in LEVELS)
    print(f"  {'model':<24}{'clean':>9}{head}{'monotone%':>11}")
    print("-" * 88)
    for name, m in models:
        curve, aucs, mono = monotonicity(m, real, y_real, sd, args.seed)
        print(f"  {name:<24}" + "".join(f"{v:>9.4f}" for v in curve)
              + f"{mono:>11.1f}")
        print(f"  {'':<24}{'AUC vs clean':>9}"
              + "".join(f"{a:>9.3f}" for a in aucs))

    print("\nE. DECOMPOSITION   (is E_global pulling its weight?)")
    print(f"  {'model':<24}{'E_g(real)':>10}{'dE_g':>10}{'E_c(real)':>10}"
          f"{'dE_c':>10}{'sd(E_g)':>10}{'sd(E_c)':>10}"
          f"{'gAUC.5':>10}{'gAUC1':>10}")
    print("-" * 108)
    for name, m in models:
        if isinstance(m, CombinedEnergy):
            decomposition_report(m, real, y_real, gens[name], y_gen, sd,
                                 args.seed, name)

    print("=" * 94)
    print("CONTINUE TO 20,000 ONLY IF THE COMBINED MODEL RETAINS ALL THREE:")
    print("  1. conditional's 1-NN and coverage        (block A, global)")
    print("  2. unconditional's corruption AUC          (block D, x0.5 and x1.0)")
    print("  3. correct per-class orientation           (block C, 0 below 0.5)")
    print()
    print("Block E checks the decomposition rather than assuming it. dE_g near")
    print("zero is not itself a failure: it can mean the generated and real global")
    print("marginals match. gAUC.5 and gAUC1 test whether E_global itself learned")
    print("useful corruption geometry. Separate trunks are justified only if those")
    print("remain weak while the standalone unconditional head is strong.")


if __name__ == "__main__":
    main()
