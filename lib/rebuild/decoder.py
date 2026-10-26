"""Decoder: 195-D EBM feature -> 1024 x 3 point cloud.

    noise + requested class -> existing EBM -> 195-D feature -> DECODER -> cloud

The decoder is deterministic. All randomness and all class conditioning stay
in the EBM, which is already trained and frozen. Nothing here touches the EBM.

ARCHITECTURE, after Achlioptas et al. (fully connected AE decoder), adapted
from 2048x3 to 1024x3 to match the feature bank and the classifier:

    z / sd_fit -> 195 -> 256 -> ReLU -> 256 -> ReLU -> 3072 -> (1024, 3)

    905,472 parameters, small beside PointNet++.

WHY z / sd_fit
--------------
The 195 PCA components span a ~17x range of scale. Feeding raw z would make
the first layer's effective learning rate wildly uneven across components.
Dividing by the FIT-split standard deviation -- the same sd the EBM samples
with, never recomputed from validation or test -- puts every component on a
comparable footing without altering the EBM or the feature bank.

NO label input, no BatchNorm, no dropout, linear output. The feature already
carries the class; determinism is the point; and Chamfer decides the
coordinate range on its own.
"""

import numpy as np
import torch
import torch.nn as nn

N_POINTS = 1024
HIDDEN = 256


class CloudDecoder(nn.Module):
    def __init__(self, dim, n_points=N_POINTS, hidden=HIDDEN):
        super().__init__()
        self.n_points = n_points
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_points * 3))

    def forward(self, z_scaled):
        return self.net(z_scaled).view(-1, self.n_points, 3)


# Cloud pairs held in flight at once. Chamfer needs an (N x M) distance
# matrix per pair, so 1024x1024 floats = 4 MB each: 48 pairs is ~200 MB.
# Without a cap here a full retrieval sweep asks for tens of gigabytes.
PAIR_CHUNK = 48


def chamfer_pairs(A, B):
    """Symmetric squared Chamfer for ALIGNED pairs: A[i] against B[i].

    (P, N, 3), (P, M, 3) -> (P,). Chunked, so P can be large.
    """
    out = torch.empty(len(A), device=A.device)
    for s in range(0, len(A), PAIR_CHUNK):
        d = torch.cdist(A[s:s + PAIR_CHUNK], B[s:s + PAIR_CHUNK]) ** 2
        out[s:s + PAIR_CHUNK] = (d.min(2).values.mean(1)
                                 + d.min(1).values.mean(1))
    return out


def chamfer_pairs_unsquared(A, B):
    """Symmetric unsquared Chamfer for aligned cloud pairs.

    This keeps the existing sum of the two directional terms, but uses the
    Euclidean norm rather than its square, as in the reference decoder papers.
    """
    out = torch.empty(len(A), device=A.device)
    for s in range(0, len(A), PAIR_CHUNK):
        d = torch.cdist(A[s:s + PAIR_CHUNK], B[s:s + PAIR_CHUNK])
        out[s:s + PAIR_CHUNK] = (d.min(2).values.mean(1)
                                 + d.min(1).values.mean(1))
    return out


def chamfer_pairs_unsquared_max(A, B):
    """FoldingNet's own loss: the MAX of the two directional terms.

        d_CH = max( mean_x min_xhat ||x - xhat||_2 ,
                    mean_xhat min_x ||xhat - x||_2 )

    The paper's stated reason: max "enforces that the distance from S to Shat
    and the distance vice versa have to be small SIMULTANEOUSLY." With a sum,
    a decoder can trade one direction against the other -- clump the points
    and pay for it in one term while winning the other. Max forbids that
    trade, which is precisely the density blindness we diagnosed.

    Lee et al. use the sum instead, so the two references disagree here. This
    is the FoldingNet form, used to test FoldingNet on its own terms.
    """
    out = torch.empty(len(A), device=A.device)
    for s in range(0, len(A), PAIR_CHUNK):
        d = torch.cdist(A[s:s + PAIR_CHUNK], B[s:s + PAIR_CHUNK])
        out[s:s + PAIR_CHUNK] = torch.maximum(d.min(2).values.mean(1),
                                              d.min(1).values.mean(1))
    return out


def chamfer(a, b):
    """Symmetric squared Chamfer distance, mean over each cloud's points.

    Both directions are needed: the a->b term alone is minimised by collapsing
    every predicted point onto one target point, and the b->a term alone by
    spraying points everywhere.
    """
    return chamfer_pairs(a, b)


def pairwise_cd(X):
    """Mean Chamfer over all unordered pairs within one set of clouds."""
    n = len(X)
    if n < 2:
        return float("nan")
    i, j = torch.triu_indices(n, n, offset=1)
    return float(chamfer_pairs(X[i], X[j]).mean())


def nearest_cloud_cd(query, pool_clouds, pool_feats, query_feats, k=64):
    """Chamfer to the nearest cloud in `pool`, shortlisted by feature distance.

    Exhaustive retrieval over 6,890 training clouds for every query is 3.5M
    cloud-pair Chamfer evaluations -- not feasible. Feature distance and
    Chamfer distance are strongly related, so the k feature-nearest candidates
    are scored exactly and the best taken. Reported as such: this is the
    nearest among the k feature-nearest, an upper bound on the true minimum.
    """
    k = min(k, len(pool_clouds))
    idx = torch.cdist(query_feats, pool_feats).topk(k, largest=False).indices
    out = torch.empty(len(query), device=query.device)
    for i in range(len(query)):
        cand = pool_clouds[idx[i]]
        out[i] = chamfer_pairs(query[i:i + 1].expand(k, -1, -1), cand).min()
    return out


def load_clouds(split="train"):
    """Clouds in EXACTLY the order and form build_bank.py encoded them.

    Bank row i corresponds to ds.datapath[i]: the first 1024 lines of the
    file, XYZ only, unit-sphere normalised. Reproducing that path here is what
    makes (cloud, feature) pairs correspond; the verify step in
    train_decoder.py re-encodes a sample and checks it.
    """
    import sys
    from itertools import islice
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import pointnet2_features as P

    import os
    alt = os.environ.get("CLOUDS_NPZ")       # e.g. the 2,048-point clouds (same objects, same order)
    if alt and split == "train":
        d = np.load(alt)
        return torch.from_numpy(d["pts"]), d["lab"]

    cache = Path(__file__).resolve().parent / f"clouds_{split}.npz"
    if cache.is_file():
        d = np.load(cache)
        return torch.from_numpy(d["pts"]), d["lab"]

    ds = P.ModelNet40(split=split)
    pts = np.empty((len(ds), P.NUM_POINT, 3), dtype=np.float32)
    lab = np.empty(len(ds), dtype=np.int64)
    for i, (cls, path) in enumerate(ds.datapath):
        with open(path) as fh:
            rows = [ln.split(",")[:3] for ln in islice(fh, P.NUM_POINT)]
        pts[i] = P.pc_normalize(np.asarray(rows, dtype=np.float32))
        lab[i] = ds.classes[cls]
    # Parsing 9,843 text files takes minutes and this is called by every
    # decoder run, so the result is cached beside the bank.
    np.savez(cache, pts=pts, lab=lab)
    return torch.from_numpy(pts), lab
