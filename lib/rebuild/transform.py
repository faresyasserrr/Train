"""Step 4: fit and save the one feature transform, and provide the only loader.

    raw 1024-D PointNet++ feature
      -> drop dead dims        -> 927   [mask from TRAIN]
      -> subtract mean         -> 927   [mean from TRAIN]
      -> project onto PCA basis-> k     [basis from TRAIN]
      -> NO whitening                   [components keep natural scale]

ONE DEFINITION
--------------
This module both fits the transform and loads it. Nothing else in the pipeline
may compute a basis. The old pipeline stored a standardized 307-dim PCA in
feature_bank.npz while ebm_space.py silently recomputed a centered 195-dim one
and used that instead. It was self-consistent in practice -- the recomputed
basis was deterministic -- but two coexisting definitions with one silently
unused is a reproducibility hazard. A single fitted artifact removes it.

WHY k=195
---------
Chosen on two independent criteria, not on "95% of variance":

  probe_dims.py     linear class information saturates by k=64 and declines
                    above 128. Taken alone this argued for k=64.
  recon_ceiling.py  OVERTURNED that. The best 1-NN any generator could reach
                    is 62.35 at k=64 -- worse than the 60.85 the existing
                    model already achieves -- against 54.04 at k=195.
                    Recall collapses to 58.11 at k=64 versus 87.66 at k=195.

The discarded directions carry little linear class signal, which is why the
probe saw nothing, and carry the within-class variation, which is why
generation needs them. The ceiling criterion wins: it measures the property we
want and has no regularization confound.

At k=195 truncation costs 3.43 points of 1-NN while the model sits 6.8 points
above its ceiling, so the bottleneck is NOT the binding constraint -- the
modelling error is.

FITTED ON TRAIN ONLY
--------------------
Mask, mean and basis use all 9,843 training features. The split protocols in
probe_dims.py and recon_ceiling.py existed to choose k honestly; that choice is
now made, so the shipped basis uses all available training data. Test is not
read by this module at all, and must not be until the final evaluation.

Run:  python rebuild/transform.py        # fit and save transform.npz
"""

import hashlib
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
BANK = ROOT / "bank.npz"
OUT = ROOT / "transform.npz"

K = 195
FPS_POLICY = "deterministic (patch_deterministic_fps(True)); index-zero start"


def _sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


class Transform:
    """The fitted map between raw 1024-D features and the k-D EBM space.

    Load it; never refit it anywhere else:

        t = Transform.load()
        Z = t.encode(F)      # (N, 1024) -> (N, k)
        F = t.decode(Z)      # (N, k)    -> (N, 1024)
    """

    def __init__(self, alive, mean, components, sd, meta):
        self.alive = alive
        self.mean = mean
        self.components = components        # (k, 927), orthonormal rows
        self.sd = sd                        # (k,) per-component std on TRAIN
        self.meta = meta
        self.k = components.shape[0]

    @classmethod
    def load(cls, path=OUT):
        d = np.load(path, allow_pickle=True)
        meta = {key: d[key].item() if d[key].shape == () else d[key]
                for key in ("k", "variance_explained", "fps_policy",
                            "checkpoint_sha256", "bank_sha256", "fitted_utc",
                            "n_train")}
        return cls(d["alive_mask"], d["mean"], d["components"], d["sd"], meta)

    def encode(self, F):
        """raw (N, 1024) -> (N, k). No whitening: sd is NOT divided out."""
        return (F[:, self.alive] - self.mean) @ self.components.T

    def decode(self, Z):
        """(N, k) -> raw (N, 1024).

        Clipped at zero because the raw features are ReLU outputs and cannot be
        negative. That clip makes decode a non-linear map, so decode is a right
        inverse of encode only up to the discarded variance and the clip.
        """
        A = Z @ self.components + self.mean
        out = np.zeros((len(Z), len(self.alive)), dtype=np.float64)
        out[:, self.alive] = A
        return np.clip(out, 0, None)


def fit():
    d = np.load(BANK, allow_pickle=True)
    F = d["train_features"].astype(np.float64)
    n_train = len(F)

    # Dead dimensions are ReLU units no TRAINING cloud ever activates. They
    # have zero std, which breaks whitening and covariance inversion, and they
    # cost input units for nothing.
    alive = F.max(0) > 0
    stored = d["alive_mask"]
    if not np.array_equal(alive, stored):
        sys.exit("alive mask disagrees with the bank -- investigate before use")
    X = F[:, alive]
    print(f"train features : {F.shape}  ->  {X.shape} alive "
          f"({(~alive).sum()} dead dims dropped)")

    mean = X.mean(0)
    _, S, Vt = np.linalg.svd(X - mean, full_matrices=False)
    cum = np.cumsum(S ** 2 / (S ** 2).sum())
    components = Vt[:K]
    Z = (X - mean) @ components.T
    sd = Z.std(0)
    var = float(cum[K - 1] * 100)

    print(f"k              : {K}   ({var:.2f}% of variance)")
    print(f"component sd   : {sd.min():.3f} to {sd.max():.3f}  "
          f"({sd.max()/sd.min():.0f}x spread, unwhitened)")
    print(f"preconditioner : gradient x sd^2 ({sd.min()**2:.3f} to "
          f"{sd.max()**2:.3f}), noise x sd;  eta tuned separately")

    # ---- mechanical verification, on TRAIN. This checks that the code is
    # correct -- that the inverse inverts and the basis is orthonormal -- not
    # how much information is lost. recon_ceiling.py answered that on held-out
    # data: 7.5% relative error at k=195.
    ortho = np.abs(components @ components.T - np.eye(K)).max()
    Zr = (Z @ components) @ components.T          # project twice
    idem = np.abs(Zr - Z).max()
    recon = np.clip((Z @ components) + mean, 0, None)
    rel = np.abs(recon - X).mean() / np.abs(X).mean() * 100

    print(f"\nmechanical checks (on train -- test is not read):")
    print(f"  orthonormal basis   : max |PP^T - I| = {ortho:.2e}")
    print(f"  projection idempotent: max err       = {idem:.2e}")
    print(f"  round-trip rel error : {rel:.2f}%   "
          f"(held-out was 7.5% -- train is fitted, so lower is expected)")
    if ortho > 1e-8 or idem > 1e-8:
        sys.exit("basis failed its mechanical checks -- do not ship this")

    meta = dict(
        k=K, variance_explained=var, fps_policy=FPS_POLICY,
        checkpoint_sha256=_sha256(Path(__file__).resolve().parents[1] /
                                  "Pointnet_Pointnet2_pytorch/log/classification/"
                                  "pointnet2_ssg_wo_normals/checkpoints/best_model.pth"),
        bank_sha256=_sha256(BANK),
        fitted_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        n_train=n_train,
    )
    np.savez(OUT, alive_mask=alive, mean=mean, components=components, sd=sd,
             **meta)
    print(f"\nprovenance:")
    for key in ("k", "variance_explained", "n_train", "fitted_utc"):
        print(f"  {key:<20} {meta[key]}")
    print(f"  {'fps_policy':<20} {meta['fps_policy']}")
    print(f"  {'checkpoint_sha256':<20} {meta['checkpoint_sha256'][:16]}...")
    print(f"  {'bank_sha256':<20} {meta['bank_sha256'][:16]}...")
    print(f"\nsaved {OUT}")


if __name__ == "__main__":
    fit()
