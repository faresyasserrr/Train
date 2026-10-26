"""Does the decoder still work on latents it was never trained on?

THE QUESTION THIS ANSWERS
-------------------------
Every decoder number so far was measured on REAL encoded latents, z = E(X)
for an actual cloud. An EBM sample is not that: it is a point the sampler
produced, which may sit off the encoder's latent manifold in a region the
decoder never saw. 85.83% recognisable is therefore a CEILING for generated
clouds, not a prediction.

Nothing measured so far says how far that falls off. This does, without
training an EBM first.

THE ARMS
--------
    real        z = E(X)                        the ceiling; should reproduce
                                                the final gate's number
    jitter d    z = E(X) + d * sd * noise       how far can you move from a
                                                real latent before the cloud
                                                stops being recognisable
    gauss-cls   z ~ N(mu_c, sd_c) per class     a crude class-conditional
                                                generator: the FLOOR any real
                                                generative model should beat
    gauss-all   z ~ N(mu, sd) global            the same, ignoring class
                                                structure

gauss-cls is the honest proxy for a class-conditional EBM. A Gaussian fitted
per class is the weakest generator that still respects the conditioning, so
an EBM -- which models the distribution far better -- should do at least this
well. If gauss-cls decodes to recognisable objects, the latent is well
behaved away from the data points and the pipeline is viable. If it collapses,
the EBM's job includes staying on-manifold, and that is worth knowing before
spending hours training one.

WHAT IS REPORTED, AND WHY EACH IS NEEDED
----------------------------------------
    class accuracy      does the frozen classifier still see the intended
                        class -- the number that decides everything
    geometry            mean max radius and the fraction beyond 1.05.
                        pc_normalize puts real clouds at exactly 1
    novelty             CD to the nearest training cloud (via a 64-nearest
                        latent shortlist). Near zero means the decoder is
                        reproducing training objects rather than generating
    diversity           mean pairwise CD between samples of the same class.
                        Near zero means every sample of a class is the same
                        object -- high class accuracy would then be worthless

Novelty and diversity are here because class accuracy alone can be passed by
a decoder that emits one memorised prototype per class. Both failure modes
have to be excluded for the number to mean anything.

Held-out val objects only. The test split is untouched.
"""

import argparse
import json
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

from autoencoder import PointAE, canonical_order          # noqa: E402
from decoder import load_clouds, chamfer_pairs_unsquared  # noqa: E402
from train_views import load                              # noqa: E402


def as_tensor(a):
    return a if torch.is_tensor(a) else torch.from_numpy(a)


@torch.no_grad()
def classify(clouds, dev, chunk=64):
    import pointnet2_features as P
    from eval_four import backbone
    bb = backbone(dev)
    return torch.cat([bb.classify(P.to_model_input(clouds[i:i + chunk])).argmax(1)
                      for i in range(0, len(clouds), chunk)])


@torch.no_grad()
def decode(model, z, y, chunk=64):
    return torch.cat([model.decode(z[i:i + chunk], y[i:i + chunk])
                      for i in range(0, len(z), chunk)])


@torch.no_grad()
def novelty(clouds, z, Zf, Xfc, k=64):
    """CD from each generated cloud to the closest of its k nearest training
    clouds by latent. Small = the decoder is reproducing a training object."""
    nn = torch.cdist(z, Zf).topk(k, largest=False).indices
    out = torch.empty(len(z), device=z.device)
    gc = canonical_order(clouds)
    for i in range(len(z)):
        cand = Xfc[nn[i]]
        out[i] = chamfer_pairs_unsquared(cand, gc[i].expand_as(cand)).min()
    return float(out.mean())


@torch.no_grad()
def diversity(clouds, y, max_per_class=8):
    """Mean pairwise CD between same-class samples. Small = every sample of a
    class decodes to the same object."""
    gc = canonical_order(clouds)
    vals = []
    for c in torch.unique(y):
        idx = (y == c).nonzero(as_tuple=True)[0][:max_per_class]
        if len(idx) < 2:
            continue
        a = gc[idx]
        # all ordered pairs i != j
        i, j = torch.triu_indices(len(idx), len(idx), offset=1)
        vals.append(chamfer_pairs_unsquared(a[i], a[j]).mean())
    return float(torch.stack(vals).mean()) if vals else float("nan")


def arm(name, model, z, y, dev, Zf, Xfc, want):
    """Decode, classify against `want`, and report the four numbers."""
    cl = decode(model, z, y)
    pred = classify(cl, dev)
    acc = float((pred == want).float().mean()) * 100
    rad = cl.norm(dim=2).max(1).values
    dup = float((torch.cdist(cl, cl).median(2).values > 1e-6).float().mean()) * 100
    res = {"accuracy": acc, "mean_max_radius": float(rad.mean()),
           "over_105_pct": float((rad > 1.05).float().mean()) * 100,
           "distinct_pct": dup,
           "novelty_cd": novelty(cl, z, Zf, Xfc),
           "diversity_cd": diversity(cl, want)}
    print(f"  {name:<14}{res['accuracy']:>9.2f}%{res['mean_max_radius']:>10.3f}"
          f"{res['over_105_pct']:>9.2f}%{res['novelty_cd']:>11.5f}"
          f"{res['diversity_cd']:>11.5f}")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, default=ROOT /
                    "pointae_sphere_unsquared_L195_h1024x512x256x128_s0.pth")
    ap.add_argument("--n", type=int, default=480)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--jitter", type=float, nargs="+",
                    default=[0.25, 0.5, 1.0, 2.0])
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    import pointnet2_features as P
    P.patch_deterministic_fps(True)

    ck = torch.load(args.model, map_location="cpu", weights_only=False)
    hidden = ck.get("hidden") or (256, 256)
    model = PointAE(ck["latent"], n_points=ck["n_points"],
                    decoder=ck["decoder"], hidden=tuple(hidden)).to(dev)
    model.load_state_dict(ck["state_dict"])
    model.eval()

    print("=" * 78)
    print(f"OFF-MANIFOLD CHECK  --  {args.model.name}")
    print("=" * 78)
    print(f"  decoder {ck['decoder']}  latent {ck['latent']}  "
          f"hidden {'x'.join(str(h) for h in hidden)}")

    _, Z, y, fit_obj, val_obj, _ = load(args.seed)
    clouds, _ = load_clouds("train")

    rng = np.random.default_rng(args.seed)
    yv = y[val_obj]
    classes = np.unique(yv)
    per = max(1, args.n // len(classes))
    sel = np.concatenate([rng.permutation(np.where(yv == c)[0])[:per]
                          for c in classes])
    obj = val_obj[np.sort(sel[:args.n])]
    X = as_tensor(clouds[obj]).float().to(dev)
    yt = torch.from_numpy(y[obj]).long().to(dev)

    Xf = as_tensor(clouds[fit_obj]).float().to(dev)
    yf = torch.from_numpy(y[fit_obj]).long().to(dev)
    with torch.no_grad():
        Zf = torch.cat([model.encode(Xf[i:i + 64])
                        for i in range(0, len(Xf), 64)])
        z = torch.cat([model.encode(X[i:i + 64]) for i in range(0, len(X), 64)])
    Xfc = canonical_order(Xf)

    # Per-class Gaussian fitted on the FIT split only.
    mu_c = torch.stack([Zf[yf == c].mean(0) for c in range(40)])
    sd_c = torch.stack([Zf[yf == c].std(0) for c in range(40)])
    mu, sd = Zf.mean(0), Zf.std(0)
    print(f"  fit latents {len(Zf)}   per-class sd mean {float(sd_c.mean()):.4f}"
          f"   global sd mean {float(sd.mean()):.4f}")
    print(f"\n  evaluating {len(obj)} held-out objects\n")

    print(f"  {'arm':<14}{'class acc':>10}{'radius':>10}{'>1.05':>10}"
          f"{'novelty':>11}{'diversity':>11}")
    print("  " + "-" * 66)
    out = {}
    g = torch.Generator(device=dev).manual_seed(args.seed)

    out["real"] = arm("real", model, z, yt, dev, Zf, Xfc, yt)
    for d in args.jitter:
        zz = z + d * sd[None, :] * torch.randn(z.shape, generator=g, device=dev)
        out[f"jitter{d}"] = arm(f"jitter {d}", model, zz, yt, dev, Zf, Xfc, yt)
    zc = mu_c[yt] + sd_c[yt] * torch.randn(z.shape, generator=g, device=dev)
    out["gauss_class"] = arm("gauss-cls", model, zc, yt, dev, Zf, Xfc, yt)
    zg = mu[None, :] + sd[None, :] * torch.randn(z.shape, generator=g, device=dev)
    out["gauss_global"] = arm("gauss-all", model, zg, yt, dev, Zf, Xfc, yt)

    print("\n  novelty   CD to nearest training cloud; near 0 = reproducing"
          " training objects")
    print("  diversity mean pairwise CD within a class; near 0 = one"
          " prototype per class")
    print("\n  Read: gauss-cls is the FLOOR a class-conditional EBM should"
          " beat. If it")
    print("  holds up, the latent is well behaved off the data and the"
          " pipeline is viable.")

    dest = ROOT / f"offmanifold_{args.model.stem}_n{len(obj)}.json"
    json.dump({"model": args.model.name, "n": len(obj), "seed": args.seed,
               "arms": out}, open(dest, "w"), indent=1)
    print(f"\n  saved {dest.name}")


if __name__ == "__main__":
    main()
