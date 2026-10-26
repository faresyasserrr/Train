"""Is the energy usable as a realism constraint?

The adversarial stage uses E(x, true_label) as a realism constraint while a
separate classifier supplies the misclassification. That assumes low energy
means realistic. Three separate things have to hold, and this script measures
all three.

1. GLOBAL ORIENTATION -- dE and AUC
   dE  = mean E(generated) - mean E(real). Negative is the pathology: the
         model rating its own output as more plausible than the data.
   AUC = P(a random generated sample scores higher energy than a random real
         one). 0.5 means the two populations are interchangeable under the
         model's own energy, which is what a well-fit model should give.
         Clearly below 0.5 means inverted; clearly above means the model can
         still tell its samples from real data. Over-separation is the milder
         failure -- real data is still cheapest, so the constraint still points
         the right way.

         Do NOT assume this drifts to 0.5 with longer training. It could
         improve, plateau, or overshoot. It is reported so it can be watched.

2. PER-CLASS ORIENTATION -- macro AUC
   A global AUC can sit at 0.55 while individual classes are inverted, with the
   errors cancelling. Since generation is per-class and the adversarial stage
   works one class at a time, orientation has to hold CLASSWISE. Reported as a
   macro average over classes with enough validation examples, plus the worst
   class, which is what actually limits the method.

3. LOCAL MONOTONICITY -- the corruption ladder
   The strictest and most relevant test. Take a clean real feature, corrupt it
   progressively, and check the energy RISES. Global AUC compares populations
   and says nothing about the neighbourhood of any particular real point. An
   adversarial search moves away from a real feature, so what it relies on is
   exactly this local behaviour.

   Reported as mean energy at each corruption level, the AUC of corrupted
   against clean at each level, and the fraction of individual samples whose
   energy increases monotonically across the whole ladder.

   CAVEAT, the same one detection_protocol.py records: the corruption here is
   isotropic Gaussian, scaled per dimension by sd. Adversarial perturbations
   are chosen adversarially and are not Gaussian, so these numbers are an
   upper bound on what to expect against a real attack, not a prediction.

WHY THE TRAINING LOSS WAS MISLEADING
------------------------------------
The training log shows loss = E(real) - E(fake) persistently positive, which
looks like the model rating its samples above real data. It is not the same
quantity. The training loop measures E(real + 0.03*sd noise) against fake from
a 150-step no-decay chain on the fit split; this script measures E(clean real)
against the 125-step decayed eval sampler on held-out val. The data noise alone
raises real energy enough to explain the positive training gap.

PROTOCOL
--------
Uses the SAME fit/val split as the training run at the given seed, so the real
features are genuinely held out from the models being scored. Test is not read.
Generated labels follow empirical val frequencies, and every generated sample
is scored under the label it was generated from.

Run:  python rebuild/check_energy.py
      python rebuild/check_energy.py --seed 1
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import rankdata
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from transform import Transform  # noqa: E402
from train_conditional import (  # noqa: E402
    BANK, EVAL_STEPS, MIN_VAL, PER_CLASS_EVAL, VAL_FRAC, Energy, dev, langevin,
)

N = 2048
LEVELS = [0.1, 0.25, 0.5, 1.0, 2.0]      # matches detection_protocol.py
NEUTRAL = 0.03                            # AUC within this of 0.5 reads neutral


def auc(pos, neg):
    """P(a random `pos` scores higher than a random `neg`)."""
    a, b = pos.cpu().numpy(), neg.cpu().numpy()
    r = rankdata(np.concatenate([a, b]))
    n1, n2 = len(a), len(b)
    return float((r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n2))


def energies(net, X, y, chunk=512):
    out = []
    with torch.no_grad():
        for s in range(0, len(X), chunk):
            xb, yb = X[s:s + chunk], y[s:s + chunk]
            out.append(net(xb, yb) if net.n_out > 1 else net(xb))
    return torch.cat(out)


def grad_norms(net, X, y, chunk=256):
    out = []
    for s in range(0, len(X), chunk):
        xb = X[s:s + chunk].clone().detach().requires_grad_(True)
        e = net(xb, y[s:s + chunk]) if net.n_out > 1 else net(xb)
        out.append(torch.autograd.grad(e.sum(), [xb])[0].norm(dim=1).detach())
    return torch.cat(out)


def generate(net, y_gen, sd, dim, seed):
    torch.manual_seed(seed + 5000)
    x0 = torch.randn(len(y_gen), dim, device=dev) * sd
    fn = (lambda x: net(x, y_gen)) if net.n_out > 1 else (lambda x: net(x))
    return langevin(fn, x0, EVAL_STEPS, sd, decay=True)


def tag(a):
    if a < 0.5 - NEUTRAL:
        return "inverted"
    if a > 0.5 + NEUTRAL:
        return "separated"
    return "neutral"


def global_check(net, real, y_real, gen, y_gen):
    e_real, e_gen = energies(net, real, y_real), energies(net, gen, y_gen)
    g_real = grad_norms(net, real, y_real).mean().item()
    g_gen = grad_norms(net, gen, y_gen).mean().item()
    a = auc(e_gen, e_real)
    return (e_real.mean().item(), e_gen.mean().item(),
            e_gen.mean().item() - e_real.mean().item(), a, g_real, g_gen)


def classwise_auc(net, Zval, yval, sd, dim, seed, eligible):
    """Orientation must hold per class -- a good global AUC can average over
    inverted classes. The worst class is what limits the method."""
    per = {}
    for c in eligible:
        rc = Zval[yval == c]
        n = min(PER_CLASS_EVAL, len(rc))
        rc = rc[torch.randperm(len(rc), generator=torch.Generator()
                               .manual_seed(seed + c))[:n]].to(dev)
        yc = torch.full((n,), c, dtype=torch.long, device=dev)
        gc = generate(net, yc, sd, dim, seed + c)
        per[c] = auc(energies(net, gc, yc), energies(net, rc, yc))
    vals = np.array(list(per.values()))
    worst = min(per, key=per.get)
    return vals.mean(), vals.std(ddof=1), worst, per[worst], (vals < 0.5).sum()


def monotonicity(net, real, y_real, sd, seed):
    """Does energy RISE as a clean real feature is progressively corrupted?

    This is the property the adversarial stage depends on, and it is local --
    population-level AUC does not imply it.
    """
    torch.manual_seed(seed + 777)
    clean = energies(net, real, y_real)
    # Follow one ray away from each real feature. Drawing fresh noise at every
    # level would compare unrelated points and make per-sample monotonicity
    # meaningless.
    direction = torch.randn_like(real)
    curve, aucs = [clean.mean().item()], []
    stack = [clean]
    for lv in LEVELS:
        pert = real + lv * sd[None, :] * direction
        e = energies(net, pert, y_real)
        stack.append(e)
        curve.append(e.mean().item())
        aucs.append(auc(e, clean))
    S = torch.stack(stack, dim=1)                    # (N, 1 + len(LEVELS))
    mono = float((S[:, 1:] > S[:, :-1]).all(dim=1).float().mean() * 100)
    return curve, aucs, mono


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--iters", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    t = Transform.load()
    d = np.load(BANK, allow_pickle=True)
    Z = t.encode(d["train_features"].astype(np.float64)).astype(np.float32)
    y = d["train_labels"]
    Zf, Zv, yf, yv = train_test_split(
        Z, y, test_size=VAL_FRAC, random_state=args.seed, stratify=y)
    Zval, yval = torch.from_numpy(Zv), torch.from_numpy(yv).long()
    sd = torch.from_numpy(Zf).to(dev).std(0)
    dim = t.k

    counts = np.bincount(yv, minlength=40)
    eligible = [c for c in range(40) if counts[c] >= MIN_VAL]
    cnt = torch.bincount(yval, minlength=40).float()
    torch.manual_seed(args.seed + 5000)
    y_gen = torch.multinomial(cnt / cnt.sum(), N, replacement=True).to(dev)
    idx = torch.randperm(len(Zval),
                         generator=torch.Generator().manual_seed(args.seed))[:N]
    real, y_real = Zval[idx].to(dev), yval[idx].to(dev)

    print(f"device {dev}  k={dim}  val {tuple(Zval.shape)}  seed {args.seed}  "
          f"(test not read)")
    print(f"{N} real vs {N} generated, {EVAL_STEPS}-step sampler, "
          f"{len(eligible)} classes for the per-class read\n")

    models = []
    for name in ("conditional", "conditional_ema",
                 "unconditional", "unconditional_ema"):
        path = ROOT / f"{name}_s{args.seed}_{args.iters}.pth"
        if not path.is_file():
            print(f"  {name}: missing ({path.name})")
            continue
        ck = torch.load(path, map_location=dev, weights_only=False)
        net = Energy(ck["dim"], ck["n_out"]).to(dev)
        net.load_state_dict(ck["state_dict"])
        net.eval()
        models.append((name, net))

    print("1. GLOBAL ORIENTATION")
    print(f"  {'model':<24}{'E(real)':>9}{'E(gen)':>9}{'dE':>9}{'AUC':>8}"
          f"{'':>11}{'|g|real':>9}{'|g|gen':>9}")
    print("-" * 88)
    gens = {}
    for name, net in models:
        gens[name] = generate(net, y_gen, sd, dim, args.seed)
        er, eg, dE, a, gr, gg = global_check(net, real, y_real, gens[name], y_gen)
        print(f"  {name:<24}{er:>9.4f}{eg:>9.4f}{dE:>+9.4f}{a:>8.3f}"
              f"{tag(a):>11}{gr:>9.4f}{gg:>9.4f}")

    print("\n2. PER-CLASS ORIENTATION  (macro AUC over eligible classes)")
    print(f"  {'model':<24}{'macro AUC':>11}{'sd':>7}{'worst class':>13}"
          f"{'worst AUC':>11}{'# below 0.5':>13}")
    print("-" * 88)
    for name, net in models:
        if net.n_out == 1:
            print(f"  {name:<24}{'n/a -- unconditional, cannot target a class':>62}")
            continue
        m, s, wc, wa, nbelow = classwise_auc(net, Zval, yval, sd, dim,
                                             args.seed, eligible)
        print(f"  {name:<24}{m:>11.3f}{s:>7.3f}"
              f"{d['classes'][wc]:>13}{wa:>11.3f}{nbelow:>7}/{len(eligible)}")

    print("\n3. LOCAL MONOTONICITY  (mean energy as clean real data is corrupted)")
    head = "".join(f"{f'x{lv}':>9}" for lv in LEVELS)
    print(f"  {'model':<24}{'clean':>9}{head}{'monotone%':>11}")
    print("-" * 88)
    for name, net in models:
        curve, aucs, mono = monotonicity(net, real, y_real, sd, args.seed)
        print(f"  {name:<24}" + "".join(f"{v:>9.4f}" for v in curve)
              + f"{mono:>11.1f}")
        print(f"  {'':<24}{'AUC vs clean':>9}"
              + "".join(f"{a:>9.3f}" for a in aucs))

    print("-" * 88)
    print("1. dE > 0 and AUC >= 0.5: real data is cheapest, constraint points")
    print("   the right way. AUC within 0.03 of 0.5 reads as neutral, not")
    print("   inverted -- the difference is within sampling variation at this N.")
    print("2. Macro AUC must hold CLASSWISE. Watch the worst class and the")
    print("   count below 0.5: the adversarial stage works one class at a time.")
    print("3. Energy must RISE with corruption, and 'monotone%' is the fraction")
    print("   of individual samples where it rises at EVERY step. This is the")
    print("   local property an adversarial search actually relies on; global")
    print("   AUC does not imply it. Gaussian corruption is an upper bound --")
    print("   real adversarial perturbations are not Gaussian.")


if __name__ == "__main__":
    main()
