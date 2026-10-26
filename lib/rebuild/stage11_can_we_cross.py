"""Can the latent path travel from one real class into another real class?

THE QUESTION
Every negative we generate must decode to a point cloud, and the decoder only
ever learned to map latents-of-real-objects to real objects. That bounded the
cloud-space search: at r = 1.5 the samples were still objects (crowd ratio
1.1-1.8), at r = 3.0 they were not (2.0-4.4). The conclusion drawn was that the
generator can produce NEAR-OOD (warped furniture) but not FAR-OOD (a different
real object such as a laptop).

That conclusion was inferred, not measured. It is testable directly: both
endpoints are known to decode validly, so walk between them.

    desk latent  o --------------------- o  laptop latent
                    what is in between?

If the intermediate latents decode into coherent objects that transition
gradually, the region between classes IS reachable and the sampler could in
principle cross it. If they collapse into shapes near neither endpoint, the
path runs through garbage and the two class regions are separated by a gap the
decoder cannot represent.

WHAT IS MEASURED ALONG THE PATH
For each interpolation step t, the decoded cloud is scored against REAL objects
of both endpoint classes:

    d_start   Chamfer to the nearest real object of the start class
    d_end     Chamfer to the nearest real object of the end class

both in units of the start class's ruler (the mean Chamfer between two
different real objects of that class). A coherent path shows d_start rising and
d_end falling, with the smaller of the two staying near 1 -- always close to
SOMETHING real. A broken path shows both large in the middle: close to nothing.

Also recorded: the classifier's predicted class and energy at each step, which
says whether the crossing is visible to the model as a change of identity
rather than merely a change of shape.

THE CLASS LABEL
The decoder is conditional, taking (z, y). A latent alone does not determine the
output. Three conditionings are therefore walked separately:

    start label held fixed      the setting our hole search actually uses
    end label held fixed
    label embedding interpolated alongside the latent

If only the third crosses successfully, then the latent walk was never the
limitation -- the fixed class conditioning was, and that is a different and
much cheaper thing to change.

DISCIPLINE
This uses a real laptop latent as an endpoint. It is a capability test of the
frozen decoder, which was trained on all 40 ModelNet classes regardless; no
classifier is trained and no hyper-parameter is selected from the result. The
official test split is not opened.
"""

import argparse
import sys
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

AE_DEFAULT = "pointae_sphere_unsquared_L195_h1024x512x256x128_s0.pth"


def nearest_cd(Q, R, dev, points, seed=0, chunk=16):
    """Chamfer from every cloud of Q to its nearest cloud in R."""
    rng = np.random.default_rng(seed)
    sub = rng.choice(Q.shape[1], points, replace=False)
    Q = Q[:, sub, :]
    R = R[:, sub, :].to(dev)
    out = []
    for i in range(0, len(Q), chunk):
        q = Q[i:i + chunk].to(dev)
        best = torch.full((len(q),), float("inf"), device=dev)
        for j in range(len(R)):
            d2 = torch.cdist(q, R[j:j + 1].expand(len(q), -1, -1)).pow(2)
            cd = d2.min(2).values.mean(1) + d2.min(1).values.mean(1)
            best = torch.minimum(best, cd)
        out.append(best.cpu())
    return torch.cat(out).numpy()


def ruler_of(R, dev, points, seed=0, chunk=16):
    rng = np.random.default_rng(seed)
    sub = rng.choice(R.shape[1], points, replace=False)
    R = R[:, sub, :].to(dev)
    tot = n = 0
    for i in range(0, len(R), chunk):
        a = R[i:i + chunk]
        for j in range(len(R)):
            d2 = torch.cdist(a, R[j:j + 1].expand(len(a), -1, -1)).pow(2)
            cd = d2.min(2).values.mean(1) + d2.min(1).values.mean(1)
            idx = torch.arange(i, i + len(a), device=dev)
            cd = cd[idx != j]
            tot += float(cd.sum())
            n += len(cd)
    return tot / max(n, 1)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", type=Path, default=ROOT / "stage6_armd_s0.pth")
    ap.add_argument("--ae", type=Path, default=ROOT / AE_DEFAULT)
    ap.add_argument("--start", default="desk")
    ap.add_argument("--end", nargs="+", default=["laptop", "bed", "glass_box"])
    ap.add_argument("--pairs", type=int, default=24)
    ap.add_argument("--steps", type=int, default=11)
    ap.add_argument("--ref", type=int, default=40,
                    help="real objects per class used as the reference set")
    ap.add_argument("--points", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    import pointnet2_features as P
    P.patch_deterministic_fps(True)
    from decoder import load_clouds
    from autoencoder import PointAE, canonical_order
    from train_latent_ebm import BANK
    from stage6_cosine_head import CosineEnergyNet
    with (P.DATA_ROOT / "modelnet40_shape_names.txt").open() as handle:
        names = [line.rstrip() for line in handle]

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    classes, cname = ck["classes"], ck["class_names"]
    net = CosineEnergyNet(len(classes),
                          scale=float(ck.get("scale", 16.0))).to(dev)
    net.load_state_dict(ck["state_dict"])
    net.eval()

    bank = np.load(BANK, allow_pickle=True)
    Z = torch.from_numpy(bank["latents"]).float().to(dev)
    zlab = bank["labels"]
    ack = torch.load(args.ae, map_location="cpu", weights_only=False)
    ae = PointAE(ack["latent"], n_points=ack["n_points"],
                 decoder=ack["decoder"],
                 hidden=tuple(ack["hidden"])).to(dev).eval()
    ae.load_state_dict(ack["state_dict"])
    ae.out_points = ack.get("out_points")
    ae.requires_grad_(False)
    emb = ae.decoder.embed

    Xtr, ytr = load_clouds("train")
    Xtr = torch.as_tensor(np.asarray(Xtr)).float()
    ytr = np.asarray(ytr)

    ci = names.index(args.start)
    Rs = Xtr[np.where(ytr == ci)[0][:args.ref]]
    ruler = ruler_of(Rs, dev, args.points, args.seed)

    print("=" * 78)
    print(f"CAN THE LATENT PATH CROSS FROM {args.start.upper()} INTO ANOTHER "
          f"CLASS?")
    print("=" * 78)
    print(f"  decoder {args.ae.name[:34]}...   classifier {args.ckpt.name}")
    print(f"  {args.start} ruler = {ruler:.4f}   "
          f"distances below are in rulers")
    print("  d_start / d_end = Chamfer to the nearest REAL object of each end")
    print("  the SMALLER of the two is what matters: near 1 means the shape is")
    print("  close to something real; both large means it is close to nothing")
    print("")

    @torch.no_grad()
    def classify(X):
        f = net(X.to(dev))
        return (f.argmax(1).cpu().numpy(),
                (-torch.logsumexp(f, 1)).cpu().numpy())

    for endname in args.end:
        cj = names.index(endname)
        Re = Xtr[np.where(ytr == cj)[0][:args.ref]]
        zi = np.where(zlab == ci)[0]
        zj = np.where(zlab == cj)[0]
        n = min(args.pairs, len(zi), len(zj))
        a = Z[torch.from_numpy(rng.choice(zi, n, replace=False)).long().to(dev)]
        b = Z[torch.from_numpy(rng.choice(zj, n, replace=False)).long().to(dev)]
        ya = torch.full((n,), ci, device=dev, dtype=torch.long)
        yb = torch.full((n,), cj, device=dev, dtype=torch.long)

        print(f"  {args.start.upper()}  ->  {endname.upper()}"
              f"   ({n} pairs, {args.steps} steps)")
        for mode in ("start label", "end label", "label interpolated"):
            print(f"    conditioning: {mode}")
            print(f"      {'t':>5}{'d_start':>9}{'d_end':>8}{'min':>8}"
                  f"{'E':>8}   predicted")
            for t in np.linspace(0, 1, args.steps):
                z = (1 - t) * a + t * b
                with torch.no_grad():
                    if mode == "start label":
                        x = canonical_order(ae.decode(z, ya))
                    elif mode == "end label":
                        x = canonical_order(ae.decode(z, yb))
                    else:
                        e = (1 - t) * emb(ya) + t * emb(yb)
                        cond = torch.cat([z, e], dim=1).unsqueeze(2).expand(
                            -1, -1, ae.decoder.n_points)
                        grid = ae.decoder.anchors.t().unsqueeze(0).expand(
                            n, -1, -1)
                        d = ae.decoder.refine(ae.decoder.mlp(
                            torch.cat([cond, grid], dim=1))).transpose(2, 1)
                        x = canonical_order(d)
                ds = np.median(nearest_cd(x.cpu(), Rs, dev, args.points)) / ruler
                de = np.median(nearest_cd(x.cpu(), Re, dev, args.points)) / ruler
                pred, en = classify(x.cpu())
                top = np.bincount(pred, minlength=len(classes)).argmax()
                frac = 100 * (pred == top).mean()
                print(f"      {t:>5.2f}{ds:>9.2f}{de:>8.2f}"
                      f"{min(ds, de):>8.2f}{np.median(en):>8.2f}   "
                      f"{cname[top]} {frac:.0f}%")
            print("")

    print("  HOW TO READ IT")
    print("    min column stays near 1        the path stays on real shapes;")
    print("                                   the classes ARE connected and a")
    print("                                   sampler could cross")
    print("    min column rises in the middle the path leaves the data; the")
    print("                                   two regions are separated by a")
    print("                                   gap the decoder cannot represent")
    print("    only 'label interpolated'      the latent walk was never the")
    print("    crosses successfully           limit -- the fixed class")
    print("                                   conditioning was")


if __name__ == "__main__":
    main()
