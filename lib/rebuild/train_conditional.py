"""Step 5: does class conditioning improve feature generation?

A matched screen. Two models, identical in every respect except conditioning,
trained on identical data with identical batch sequences and the same seed:

    conditional     trunk -> 40 energies, E(x, y) = head(trunk(x))[y]
    unconditional   trunk -> 1 energy,    E(x)

The diagonal preconditioner (gradient x sd^2) is NOT used. It is a separate
change and gets its own experiment once this one resolves. The gradient step
here is the inherited -grad E at coefficient 1.0.

PROTOCOL -- test is never read
------------------------------
    fit  (70% of train)   trains both EBMs
    val  (30% of train)   the screen decides from these numbers
    test                  reserved for the final chosen model only

Deciding whether to continue training from test numbers would be test-set model
selection, which is what every earlier step in this rebuild was careful to
avoid.

One caveat stated rather than hidden: transform.npz was fitted on all 9,843
training features, so the PCA basis has seen the validation slice. The
transform is unsupervised and identical for both models, so the COMPARISON is
unaffected; only the absolute val numbers are slightly optimistic. The test
evaluation later is clean.

WHY ema0.99_ebm.pth IS NOT IN THIS SCREEN
-----------------------------------------
It was trained on all of train, so our validation set is its training data --
scoring it here would flatter it. It is also a 26,500-iteration model with EMA,
which is not an equal-budget control for a 5,000-iteration run. It returns as a
reference at the final test evaluation.

EMA
---
Both models keep an EMA at 0.99 and are reported raw and EMA, so neither the
comparison against each other nor the eventual comparison against the EMA
baseline is confounded by averaging.

LOSS
----
    loss = E(real, label) - E(fake, label)          (conditional)
    loss = E(real)        - E(fake)                 (unconditional)

Bare contrastive, no regularizer: alpha*E^2 is minimized perfectly by E=0 and
collapsed every early run, and the gradient penalty was falsified separately
(lambda 0.01 to 100 drove its target quantity past inversion with no effect).

Only E(x, true_label) is ever evaluated. Mismatched-label energies stay
untrained on purpose: the downstream adversarial stage needs the EBM only as a
realism constraint, and a separate classifier supplies the misclassification.

LABELS
------
    training    uniform -- batch of 80 is exactly 2 per class, giving every
                class equal capacity despite the 64-to-889 imbalance
    evaluation  empirical val frequencies, so the global comparison is fair

Per-class metrics use EQUAL set sizes. 1-NN with |A| != |B| is biased toward
the larger set, and coverage would cap at n_gen / n_real.

WHICH BLOCK DECIDES
-------------------
GLOBAL decides. Both models are asked for the same label mix and scored against
the same reference, so it is a fair head-to-head.

MACRO PER CLASS is not, and must not be read as one. Asked for class c the
unconditional model can only sample from p(x) globally, so it is scored against
a single class's real data while generating from all 40 -- it loses by
construction, not by quality. That block verifies the conditional model's class
control works; it cannot compare the two models.

5,000 iterations is a SCREEN. It ranks conditioning against no conditioning; it
is not a converged model.

Run:  python rebuild/train_conditional.py
      python rebuild/train_conditional.py --iters 20000
"""

import argparse
import copy
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from transform import Transform  # noqa: E402

BANK = ROOT / "bank.npz"

VAL_FRAC = 0.30         # 21 of 40 classes then have >= 50 val examples
PER_CLASS = 2           # examples per class per batch -> batch 80
TRAIN_STEPS = 150
EVAL_STEPS = 125
EPSILON = 0.01          # noise scale, per dimension via sd
DATA_EPSILON = 0.03     # noise on real features, per dimension via sd
LR = 1e-4
EMA_DECAY = 0.99
N_EVAL = 2048
PER_CLASS_EVAL = 50
MIN_VAL = 50            # a class needs this many val examples to be reported
PR_K = 5

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------------------------------------------- models
class Energy(nn.Module):
    """Shared trunk; n_out=40 for the conditional model, 1 for the control.

    The two differ only in the width of the final layer, so the comparison
    isolates conditioning rather than capacity: 1,795,624 parameters against
    1,775,617, a difference of about 1%.
    """

    def __init__(self, dim, n_out, hidden=(1024, 1024, 512)):
        super().__init__()
        layers, d = [], dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.SiLU()]
            d = h
        self.trunk = nn.Sequential(*layers)
        self.head = nn.Linear(d, n_out)
        self.n_out = n_out

    def forward(self, x, y=None):
        e = self.head(self.trunk(x))
        if self.n_out == 1:
            return e.squeeze(-1)
        return e.gather(1, y[:, None]).squeeze(1)


class EMA:
    """Exponential moving average of the weights, kept alongside training."""

    def __init__(self, model, decay=EMA_DECAY):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for s, p in zip(self.shadow.parameters(), model.parameters()):
            s.mul_(self.decay).add_(p, alpha=1 - self.decay)
        for s, b in zip(self.shadow.buffers(), model.buffers()):
            s.copy_(b)


# ----------------------------------------------------------------- sampler
def langevin(energy_fn, x, steps, sd, decay):
    """Inherited update: x += -grad E + noise, gradient coefficient 1.0.

    No preconditioner by design -- `a * sd**2` is the next experiment. Noise is
    per-dimension because the components are unwhitened and span 17x; that part
    is already the existing recipe.
    """
    noise_scale = (EPSILON * sd)[None, :]
    x = x.clone().detach().requires_grad_(True)
    for i in range(steps):
        g = torch.autograd.grad(energy_fn(x).sum(), [x])[0]
        nd = max(0.0, (steps - i - 5) / (steps - 5)) if decay else 1.0
        x.data += -g + nd * noise_scale * torch.randn_like(x)
    return x.detach()


# ----------------------------------------------------------------- metrics
def nn_two(A, B):
    X = torch.cat([A, B])
    y = torch.cat([torch.zeros(len(A)), torch.ones(len(B))]).to(dev)
    d = torch.cdist(X, X)
    d.fill_diagonal_(float("inf"))
    return float((y[d.argmin(1)] == y).float().mean() * 100)


def coverage(gen, real):
    return len(torch.unique(torch.cdist(gen, real).argmin(1))) / len(real) * 100


def prec_rec(gen, real, k=PR_K):
    def radii(X):
        d = torch.cdist(X, X)
        d.fill_diagonal_(float("inf"))
        return d.topk(min(k, len(X) - 1), largest=False).values[:, -1]
    rr, rg = radii(real), radii(gen)
    d = torch.cdist(gen, real)
    return (float((d <= rr[None, :]).any(1).float().mean() * 100),
            float((d.T <= rg[None, :]).any(1).float().mean() * 100))


def score(gen, real):
    p, r = prec_rec(gen, real)
    return [nn_two(gen, real), coverage(gen, real), p, r]


# ----------------------------------------------------------------- evaluation
def evaluate(net, Zval, yval, sd, dim, seed, label, eligible):
    """Identical protocol for every model. Conditional nets use the labels the
    chain is handed; the unconditional control ignores them."""
    net.eval()
    conditional = net.n_out > 1
    torch.manual_seed(seed + 5000)
    counts = torch.bincount(yval, minlength=40).float()

    # headline: global, labels at empirical val frequencies
    y_gen = torch.multinomial(counts / counts.sum(), N_EVAL,
                              replacement=True).to(dev)
    x0 = torch.randn(N_EVAL, dim, device=dev) * sd
    fn = (lambda x: net(x, y_gen)) if conditional else (lambda x: net(x))
    gen = langevin(fn, x0, EVAL_STEPS, sd, decay=True)
    ref = Zval[torch.randperm(len(Zval),
                              generator=torch.Generator().manual_seed(seed))
               [:N_EVAL]].to(dev)
    glob = score(gen, ref)

    # secondary: macro per class, equal set sizes on both sides
    rows = []
    for c in eligible:
        rc = Zval[yval == c]
        n = min(PER_CLASS_EVAL, len(rc))
        rc = rc[torch.randperm(len(rc),
                               generator=torch.Generator().manual_seed(seed + c))
                [:n]].to(dev)
        yc = torch.full((n,), c, dtype=torch.long, device=dev)
        xc = torch.randn(n, dim, device=dev) * sd
        fnc = (lambda x: net(x, yc)) if conditional else (lambda x: net(x))
        rows.append(score(langevin(fnc, xc, EVAL_STEPS, sd, decay=True), rc))
    macro = np.mean(rows, axis=0)

    print(f"  {label:<26}{glob[0]:>8.2f}{glob[1]:>8.2f}{glob[2]:>8.2f}"
          f"{glob[3]:>8.2f} | {macro[0]:>8.2f}{macro[1]:>8.2f}"
          f"{macro[2]:>8.2f}{macro[3]:>8.2f}")
    return glob, macro


# ----------------------------------------------------------------- training
def train(args, n_out, Zfit, yfit, sd, dim, tag):
    """Both models are trained by this one function, from the same seed, so
    their batch index sequences are identical -- a matched control."""
    torch.manual_seed(args.seed)
    net = Energy(dim, n_out).to(dev)
    ema = EMA(net)
    opt = torch.optim.Adam(net.parameters(), lr=LR)

    by_class = [torch.where(yfit == c)[0].to(dev) for c in range(40)]
    labels = torch.arange(40, device=dev).repeat_interleave(PER_CLASS)
    batch = len(labels)
    data_scale = (DATA_EPSILON * sd)[None, :]
    conditional = n_out > 1

    print(f"\n{tag}: {sum(p.numel() for p in net.parameters()):,} parameters")

    # Reseed HERE, not before construction. The two heads are Linear(512, 40)
    # and Linear(512, 1), which consume different numbers of random draws, so
    # seeding before the model is built leaves the RNG in a different state for
    # each run and shifts every later batch index and noise sample. Reseeding
    # after construction is what actually makes the two runs see identical
    # batch sequences.
    torch.manual_seed(args.seed + 1000)

    t0 = time.time()
    for it in range(1, args.iters + 1):
        idx = torch.cat([g[torch.randint(len(g), (PER_CLASS,), device=dev)]
                         for g in by_class])
        real = Zfit[idx] + data_scale * torch.randn(batch, dim, device=dev)

        x0 = torch.randn(batch, dim, device=dev) * sd
        fn = (lambda x: net(x, labels)) if conditional else (lambda x: net(x))
        fake = langevin(fn, x0, TRAIN_STEPS, sd, decay=False)

        e_real = net(real, labels) if conditional else net(real)
        e_fake = net(fake, labels) if conditional else net(fake)
        loss = e_real.mean() - e_fake.mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        ema.update(net)

        if it % args.log_every == 0 or it == 1:
            print(f"  {it:>6}/{args.iters}  loss {loss.item():+9.4f}  "
                  f"E_real {e_real.mean().item():+9.4f}  "
                  f"E_fake {e_fake.mean().item():+9.4f}  "
                  f"|fake| {fake.norm(dim=1).mean():7.2f}   "
                  f"[{time.time() - t0:.0f}s]")
    print(f"  {tag} trained in {(time.time() - t0) / 60:.1f} min")
    return net, ema.shadow


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--iters", type=int, default=5000)
    ap.add_argument("--log-every", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    t = Transform.load()
    d = np.load(BANK, allow_pickle=True)
    Z = t.encode(d["train_features"].astype(np.float64)).astype(np.float32)
    y = d["train_labels"]

    # stratified fit/val split of TRAIN. test is not loaded at all.
    Zf, Zv, yf, yv = train_test_split(
        Z, y, test_size=VAL_FRAC, random_state=args.seed, stratify=y)
    Zfit = torch.from_numpy(Zf).to(dev)
    yfit = torch.from_numpy(yf).long().to(dev)
    Zval = torch.from_numpy(Zv)
    yval = torch.from_numpy(yv).long()
    dim = t.k
    sd = Zfit.std(0)

    counts = np.bincount(yv, minlength=40)
    eligible = [c for c in range(40) if counts[c] >= MIN_VAL]

    print(f"device {dev}   k={dim}   fit {tuple(Zfit.shape)}  "
          f"val {tuple(Zval.shape)}   (test not read)")
    print(f"component sd {sd.min():.3f} to {sd.max():.3f}")
    print(f"batch {40 * PER_CLASS} = {PER_CLASS} per class, uniform labels")
    print(f"train {TRAIN_STEPS} steps no decay | eval {EVAL_STEPS} steps decay")
    print(f"gradient -grad E coefficient 1.0 -- NO preconditioner")
    print(f"per-class macro over {len(eligible)} classes with >= {MIN_VAL} val "
          f"examples, equal set sizes")

    cond, cond_ema = train(args, 40, Zfit, yfit, sd, dim, "conditional")
    unc, unc_ema = train(args, 1, Zfit, yfit, sd, dim, "unconditional")

    # Save EMA weights too -- an EMA row that wins is useless if it cannot be
    # reloaded.
    # The seed is part of the filename: without it, running seed 1 would
    # overwrite seed 0's checkpoints and make the replication unrepeatable.
    for name, net in [("conditional", cond), ("conditional_ema", cond_ema),
                      ("unconditional", unc), ("unconditional_ema", unc_ema)]:
        torch.save({"state_dict": net.state_dict(), "n_out": net.n_out,
                    "dim": dim, "iters": args.iters, "seed": args.seed,
                    "ema_decay": EMA_DECAY if name.endswith("ema") else None},
                   ROOT / f"{name}_s{args.seed}_{args.iters}.pth")

    print("\n" + "=" * 94)
    print(f"all rows: {EVAL_STEPS} steps, decay over last 5, evaluated on VAL")
    print(f"{'':<26}{'GLOBAL (N=2048, empirical labels)':^32} | "
          f"{'MACRO PER CLASS':^32}")
    print(f"  {'model':<26}{'1-NN':>8}{'COV':>8}{'Prec':>8}{'Rec':>8} | "
          f"{'1-NN':>8}{'COV':>8}{'Prec':>8}{'Rec':>8}")
    print("-" * 94)
    evaluate(cond, Zval, yval, sd, dim, args.seed, "conditional (raw)", eligible)
    evaluate(cond_ema, Zval, yval, sd, dim, args.seed, "conditional (EMA 0.99)", eligible)
    evaluate(unc, Zval, yval, sd, dim, args.seed, "unconditional (raw)", eligible)
    evaluate(unc_ema, Zval, yval, sd, dim, args.seed, "unconditional (EMA 0.99)", eligible)
    print("=" * 94)
    print("1-NN: lower is better, 50 = indistinguishable. COV/Prec/Rec: higher.")
    print()
    print("HOW TO READ THESE TWO BLOCKS -- THEY ANSWER DIFFERENT QUESTIONS.")
    print()
    print("  GLOBAL is the fair generative-quality comparison. Both models are")
    print("  asked for the same label mix and scored against the same reference,")
    print("  so conditional vs unconditional is decided HERE.")
    print()
    print("  MACRO PER CLASS is NOT a fair head-to-head. Asked for class c, the")
    print("  unconditional model can only sample from p(x) globally, so it is")
    print("  scored against one class's real data while generating from all 40.")
    print("  It is structurally guaranteed to lose. Read this block only as a")
    print("  check that the conditional model's class control actually works.")
    print()
    print("ONE EVALUATION SEED. Treat any difference below about 1 point as")
    print("uncertain. If GLOBAL favours conditioning, repeat with two more")
    print("training seeds before spending the 20,000-iteration run.")
    print()
    print("ema0.99_ebm.pth is deliberately absent: it trained on all of train,")
    print("so this val set is its training data, and it had 26,500 iterations.")
    print("It returns at the final test evaluation.")


if __name__ == "__main__":
    main()
