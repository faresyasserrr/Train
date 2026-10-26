"""Pretrained PointNet++ SSG (no normals) as a frozen feature extractor.

Everything here is pinned to the exact configuration the shipped checkpoint was
trained with, read off its own training log
(log/classification/pointnet2_ssg_wo_normals/logs/pointnet2_cls_ssg.txt):

    num_category=40  num_point=1024  use_normals=False  use_uniform_sample=False

Reported by that run: 92.53% instance / 89.17% class accuracy on ModelNet40 test.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parent / "Pointnet_Pointnet2_pytorch"
CKPT = REPO / "log/classification/pointnet2_ssg_wo_normals/checkpoints/best_model.pth"
DATA_ROOT = REPO / "data/modelnet40_normal_resampled"

# models/pointnet2_cls_ssg.py does a bare `from pointnet2_utils import ...`,
# so models/ itself has to be importable, not just the repo root.
for p in (str(REPO), str(REPO / "models"), str(REPO / "data_utils")):
    if p not in sys.path:
        sys.path.insert(0, p)

# Training-time config, frozen. Do not change these without retraining.
NUM_POINT = 1024
NUM_CATEGORY = 40
USE_NORMALS = False
USE_UNIFORM_SAMPLE = False
FEATURE_DIM = 1024


# --------------------------------------------------------------------------
# Determinism patch
# --------------------------------------------------------------------------
#
# The stock farthest-point sampler seeds itself randomly:
#
#     farthest = torch.randint(0, N, (B,), ...)   # pointnet2_utils.py:75
#
# so sa1/sa2 pick a different subset on every forward pass and the 1024-D
# feature is stochastic for a fixed input. That is fine for training (it acts
# as augmentation) but not for an EBM: E(x) must be a function of x, and
# input gradients must not carry sampling noise.
#
# Starting FPS from index 0 makes the whole backbone deterministic. The
# greedy farthest-point walk after the first pick is already deterministic,
# so this is the only source of randomness.

# Which point FPS starts from. "index0" is what bank.npz, transform.npz and
# every existing checkpoint were built with, so it MUST stay the default --
# changing it globally would silently invalidate all of them, including the
# frozen classifier used as an independent evaluator.
#
# "centroid" starts from the point farthest from the cloud's centroid, which
# is a property of the SHAPE rather than of the file's point ordering, so it
# is permutation-invariant. New models (the autoencoder) opt into it through
# the fps_start context manager; nothing else is affected.
_FPS_START = "index0"


class fps_start:
    """Temporarily select the FPS starting rule. Restores on exit."""

    def __init__(self, mode):
        assert mode in ("index0", "centroid")
        self.mode = mode

    def __enter__(self):
        global _FPS_START
        self.prev, _FPS_START = _FPS_START, self.mode

    def __exit__(self, *exc):
        global _FPS_START
        _FPS_START = self.prev


def _deterministic_fps(xyz, npoint):
    """farthest_point_sample, seeded at index 0 instead of at random.

    Same algorithm and same output as the repo's version -- verified
    bit-identical -- but two changes make it ~2.2x faster, which matters
    because FPS dominates PointNet++'s forward pass (measured: 559 ms of a
    1,003 ms encoder call at batch 64, npoint 512).

      1. `distance = torch.minimum(distance, dist)` replaces
         `mask = dist < distance; distance[mask] = dist[mask]`. The masked
         scatter builds index tensors on every one of the npoint iterations;
         the elementwise minimum does the same job with one kernel.

      2. The chosen centroids accumulate in a python list and are stacked
         once at the end, instead of an indexed write into a preallocated
         tensor inside the loop.

    The loop itself stays sequential -- each centroid depends on the previous
    one -- so this is launch-overhead reduction, not a change of algorithm.
    """
    B, N, _ = xyz.shape
    device = xyz.device
    distance = torch.full((B, N), 1e10, device=device)
    if _FPS_START == "centroid":
        # Farthest from the centroid: determined by the shape, not by which
        # point happens to sit at index 0, so shuffling the input cannot
        # change the sampled set.
        centre = xyz.mean(1, keepdim=True)
        farthest = ((xyz - centre) ** 2).sum(-1).argmax(-1)
    else:
        farthest = torch.zeros(B, dtype=torch.long, device=device)
    batch_indices = torch.arange(B, dtype=torch.long, device=device)
    centroids = []
    for _ in range(npoint):
        centroids.append(farthest)
        centroid = xyz[batch_indices, farthest].unsqueeze(1)
        distance = torch.minimum(distance, ((xyz - centroid) ** 2).sum(-1))
        farthest = distance.argmax(-1)
    return torch.stack(centroids, 1)


def patch_deterministic_fps(enable: bool = True) -> None:
    """Swap the sampler used by every SetAbstraction layer.

    Both call sites (sample_and_group, and the MSG path) look the name up as a
    module global at call time, so rebinding it here covers both.
    """
    import pointnet2_utils

    if not hasattr(pointnet2_utils, "_original_fps"):
        pointnet2_utils._original_fps = pointnet2_utils.farthest_point_sample
    pointnet2_utils.farthest_point_sample = (
        _deterministic_fps if enable else pointnet2_utils._original_fps
    )


DETERMINISTIC_FPS = True
if DETERMINISTIC_FPS:
    patch_deterministic_fps(True)


# --------------------------------------------------------------------------
# Backbone
# --------------------------------------------------------------------------

class PointNet2Backbone(nn.Module):
    """Pretrained SSG backbone truncated before the classification head.

    forward(x) -> (B, 1024) global feature, the exact tensor that
    pointnet2_cls_ssg.get_model.forward computes as `l3_points.view(B, 1024)`
    and feeds to fc1.

    The head (fc1/fc2/fc3) is retained on `.full` so classification logits stay
    available for sanity checks, but it is not used by forward().

    Fully differentiable w.r.t. the input, so input-gradient attacks work.
    """

    def __init__(self, ckpt_path: os.PathLike | str = CKPT, device: str | None = None):
        super().__init__()
        from pointnet2_cls_ssg import get_model  # noqa: E402  (needs sys.path above)

        ckpt_path = Path(ckpt_path)
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")

        self.device = torch.device(
            device if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        model = get_model(num_class=NUM_CATEGORY, normal_channel=USE_NORMALS)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state = ckpt["model_state_dict"]
        model.load_state_dict(state, strict=True)
        self.ckpt_meta = {
            "epoch": ckpt.get("epoch"),
            "instance_acc": ckpt.get("instance_acc"),
            "class_acc": ckpt.get("class_acc"),
        }

        self.full = model.to(self.device).eval()
        # Frozen: no grads for weights, but activations still carry input grads.
        for prm in self.full.parameters():
            prm.requires_grad_(False)

    def train(self, mode: bool = True):
        # Guard against accidentally putting BatchNorm into batch-statistics
        # mode, which would make features depend on the rest of the batch.
        return super().train(False)

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        """xyz: (B, 3, N) float tensor -> (B, 1024) global feature."""
        if xyz.dim() != 3 or xyz.shape[1] != 3:
            raise ValueError(f"expected (B, 3, N), got {tuple(xyz.shape)}")
        B = xyz.shape[0]
        m = self.full
        l1_xyz, l1_points = m.sa1(xyz, None)          # (B,3,512),  (B,128,512)
        l2_xyz, l2_points = m.sa2(l1_xyz, l1_points)  # (B,3,128),  (B,256,128)
        l3_xyz, l3_points = m.sa3(l2_xyz, l2_points)  # (B,3,1),    (B,1024,1)
        return l3_points.view(B, FEATURE_DIM)

    def classify(self, xyz: torch.Tensor) -> torch.Tensor:
        """Log-softmax logits over the 40 classes. Sanity checks only."""
        return self.full(xyz)[0]


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------

def pc_normalize(pc: np.ndarray) -> np.ndarray:
    """Unit-sphere normalization, identical to the repo's dataloader."""
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    return pc / m


class ModelNet40(torch.utils.data.Dataset):
    """ModelNet40 with the exact preprocessing the checkpoint was trained under.

    Returns (points, label):
        points : (1024, 3) float32, unit-sphere normalized
        label  : int64 in [0, 40)

    Also exposes the untouched 10000-point cloud via `raw(i)` so perturbations
    can be applied before or after subsampling/normalization.

    No train-time augmentation (dropout/scale/shift) is applied here; those live
    in the repo's provider.py and were only used during training.
    """

    def __init__(self, root=DATA_ROOT, split="train", npoints=NUM_POINT,
                 uniform=USE_UNIFORM_SAMPLE, normalize=True):
        self.root, self.npoints = Path(root), npoints
        self.uniform, self.normalize = uniform, normalize
        assert split in ("train", "test")

        if not self.root.is_dir():
            raise FileNotFoundError(f"dataset not found: {self.root}")

        self.cat = [l.rstrip() for l in open(self.root / "modelnet40_shape_names.txt")]
        self.classes = dict(zip(self.cat, range(len(self.cat))))
        ids = [l.rstrip() for l in open(self.root / f"modelnet40_{split}.txt")]
        names = ["_".join(x.split("_")[:-1]) for x in ids]
        self.datapath = [(names[i], self.root / names[i] / f"{ids[i]}.txt")
                         for i in range(len(ids))]

    def __len__(self):
        return len(self.datapath)

    def raw(self, index: int) -> np.ndarray:
        """Full (10000, 6) cloud straight off disk: x,y,z,nx,ny,nz."""
        return np.loadtxt(self.datapath[index][1], delimiter=",").astype(np.float32)

    def __getitem__(self, index):
        label = self.classes[self.datapath[index][0]]
        pts = self.raw(index)

        if self.uniform:
            from ModelNetDataLoader import farthest_point_sample
            pts = farthest_point_sample(pts, self.npoints)
        else:
            pts = pts[0:self.npoints, :]   # first N points, as in training

        pts = pts[:, 0:3]                  # drop normals
        if self.normalize:
            pts = pc_normalize(pts)
        return torch.from_numpy(pts.astype(np.float32)), label


def to_model_input(points: torch.Tensor) -> torch.Tensor:
    """(B, N, 3) -> (B, 3, N). The transpose train_classification.py applies."""
    return points.transpose(2, 1).contiguous()
