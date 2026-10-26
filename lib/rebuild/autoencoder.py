"""Jointly trained point-cloud autoencoder: the latent the EBM will live in.

    cloud -> PointNet++ encoder -> 128-D latent -> decoder -> cloud
                    \\____________ Chamfer ____________/

WHY THIS REPLACES THE FROZEN-BACKBONE DECODER
---------------------------------------------
The previous decoder read features from a backbone trained to CLASSIFY.
Nothing in that training ever required the features to be invertible, and the
measurements said so: cycle error 0.79 against a 0.23 floor, class accuracy
96.88% -> 67.58%. A feature-matching loss patches that from the outside and
opens a hacking surface, because only part of the cloud receives feature
gradient at all.

Training the encoder WITH the decoder removes the problem instead of fighting
it. The encoder is under direct pressure to keep whatever the decoder needs,
so the latent is reconstructable by construction.

This is Achlioptas et al.'s structure -- autoencoder first, generative model
in the frozen latent afterwards -- with an EBM in place of their GAN.

WHAT IT ALSO BUYS
-----------------
  no hacking     encoder and decoder cooperate; there is no frozen network to
                 game, so the anti-hacking gates stop being load-bearing
  independent    class purity is later measured by decoding to clouds and
    evaluation   classifying THE CLOUDS with the frozen PointNet++ classifier,
                 which is a different network from the encoder. Today's purity
                 number is scored by the same network that made the features
  smooth latent  Langevin runs in the latent, never on coordinates, so the
                 discrete FPS / ball-query gradient question never arises

CHOICES
-------
  latent 128     Achlioptas' size. Measured intrinsic dimension of the old
                 feature space was ~10-12, so 128 is generous rather than
                 tight; it is the one knob worth a second arm later.
  from scratch   the encoder is NOT initialised from the classifier
                 checkpoint. The entire point is a latent that is not shaped
                 by classification.
  deterministic  FPS is seeded at index 0 throughout, matching build_bank.py,
    FPS          so the latent bank this produces is reproducible.
  no latent      the EBM will use a frozen fit-split `sd` exactly as it does
    constraint   now, so the latent needs no normalisation layer -- but its
                 scale is reported after training, because the sampler's
                 init and noise are both built from that sd.
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent
PARENT = ROOT.parent
REPO = PARENT / "Pointnet_Pointnet2_pytorch"
for p in (str(ROOT), str(PARENT), str(REPO), str(REPO / "models")):
    if p not in sys.path:
        sys.path.insert(0, p)

from pointnet2_utils import PointNetSetAbstraction  # noqa: E402

LATENT = 128
N_POINTS = 1024


class Encoder(nn.Module):
    """PointNet++ SSG trunk, trained from scratch, ending at a latent vector.

    Identical set-abstraction configuration to the classifier so the
    comparison stays honest, but the classification head is replaced by a
    single linear map to the latent.
    """

    def __init__(self, latent=LATENT):
        super().__init__()
        self.sa1 = PointNetSetAbstraction(
            npoint=512, radius=0.2, nsample=32, in_channel=3,
            mlp=[64, 64, 128], group_all=False)
        self.sa2 = PointNetSetAbstraction(
            npoint=128, radius=0.4, nsample=64, in_channel=128 + 3,
            mlp=[128, 128, 256], group_all=False)
        self.sa3 = PointNetSetAbstraction(
            npoint=None, radius=None, nsample=None, in_channel=256 + 3,
            mlp=[256, 512, 1024], group_all=True)
        self.fc = nn.Sequential(
            nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Linear(512, latent))

    def forward(self, xyz):
        """xyz: (B, 3, N) -- the transposed form to_model_input produces."""
        b = xyz.shape[0]
        l1_xyz, l1_pts = self.sa1(xyz, None)
        l2_xyz, l2_pts = self.sa2(l1_xyz, l1_pts)
        _, l3_pts = self.sa3(l2_xyz, l2_pts)
        return self.fc(l3_pts.view(b, 1024))


class Decoder(nn.Module):
    """Fully connected decoder, after Achlioptas et al., 2048x3 -> 1024x3."""

    def __init__(self, latent=LATENT, n_points=N_POINTS, hidden=(256, 256)):
        super().__init__()
        self.n_points = n_points
        layers, d = [], latent
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        layers += [nn.Linear(d, n_points * 3)]
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z).view(-1, self.n_points, 3)


def canonical_order(X):
    """Sort a cloud's points by their coordinates, so order carries no info.

    PointNet++ is NOT order-invariant in practice. query_ball_point keeps the
    LOWEST-INDEX neighbours when more than nsample fall inside the radius,
    so which points get grouped depends on the file's ordering. Measured on
    real clouds: shuffling the input moves features by up to 1.00 with an
    index-0 FPS start and 0.89 with a centroid start -- the FPS seed is a
    small part of it.

    Sorting lexicographically by (x, y, z) makes the input a function of the
    point SET alone, so any permutation gives an identical tensor. The same
    measurement drops to ~0.003, which is float non-associativity in the
    reductions rather than order dependence.

    This matters most for the latent-cycle gate: the decoder emits points in
    its own arbitrary order, and re-encoding that without canonicalising
    would charge the autoencoder for reshuffling it did not cause.
    """
    idx = torch.arange(len(X), device=X.device).unsqueeze(1)
    order = torch.argsort(X[:, :, 2], dim=1, stable=True)
    X = X[idx, order]
    order = torch.argsort(X[:, :, 1], dim=1, stable=True)
    X = X[idx, order]
    order = torch.argsort(X[:, :, 0], dim=1, stable=True)
    return X[idx, order]


def fps_points(X, n):
    """Farthest-point sampling of n of the points of each cloud, starting at
    point 0 (deterministic). The choice carries no gradient; the kept points
    do. X: (B, N, 3) -> (B, n, 3)."""
    with torch.no_grad():
        B, N, _ = X.shape
        idx = torch.zeros(B, n, dtype=torch.long, device=X.device)
        d = torch.full((B, N), float("inf"), device=X.device)
        far = torch.zeros(B, dtype=torch.long, device=X.device)
        ar = torch.arange(B, device=X.device)
        for i in range(n):
            idx[:, i] = far
            d = torch.minimum(d, ((X - X[ar, far][:, None]) ** 2).sum(2))
            far = d.argmax(1)
    return torch.gather(X, 1, idx[..., None].expand(-1, -1, 3))


class PointAE(nn.Module):
    """Encoder + decoder. The encoder always sees canonically ordered points
    and always uses the centroid FPS start, so neither the file's point order
    nor the decoder's output order can influence the latent.

    `decoder` selects the architecture and is the ONLY thing that differs
    between the two joint arms:

        fc        flat, after Achlioptas: latent -> 256 -> 256 -> n*3
        folding   FoldingNet: deform a fixed 2D grid, twice, conditioned on
                  the latent

    The loss stays untouched across arms on purpose. There is already a joint
    FC baseline trained with the current Chamfer, so holding the loss fixed
    makes the decoder the single changed variable and any difference
    attributable. FoldingNet's published advantage was measured with a
    different Chamfer (max of the two directions, unsquared) -- so if this
    arm ties, that is the next variable to change, NOT evidence that the
    decoder architecture does not matter.
    """

    def __init__(self, latent=LATENT, n_points=N_POINTS, canonical=True,
                 decoder="fc", hidden=None):
        super().__init__()
        if hidden is None:
            hidden = (1024, 512, 256, 128) if decoder == "sphere" else (256, 256)
        self.encoder = Encoder(latent)
        if decoder == "cond":
            from cond_decoder import ClassConditionalDecoder
            self.decoder = ClassConditionalDecoder(latent, n_points,
                                                   hidden=tuple(hidden))
        elif decoder == "sphere":
            from sphere_decoder import SpherePriorDecoder
            self.decoder = SpherePriorDecoder(latent, n_points,
                                              hidden=tuple(hidden))
        elif decoder == "folding":
            from folding import FoldingDecoder
            self.decoder = FoldingDecoder(latent, n_points,
                                          hidden=int(hidden[0]))
        elif decoder == "fc":
            self.decoder = Decoder(latent, n_points, hidden=tuple(hidden))
        else:
            raise ValueError(f"unknown decoder {decoder!r}")
        self.decoder_kind = decoder
        self.latent = latent
        self.canonical = canonical
        self.hidden = tuple(hidden)

    def encode(self, X):
        """X: (B, N, 3) -> (B, latent)."""
        import pointnet2_features as P
        if self.canonical:
            X = canonical_order(X)
        with P.fps_start("centroid"):
            return self.encoder(X.transpose(2, 1).contiguous())

    @property
    def conditional(self):
        return getattr(self.decoder, "conditional", False)

    def decode(self, z, y=None):
        """Decode, passing the class label only to a conditional decoder.
        If `out_points` is set (from a checkpoint's "out_points", e.g. the
        2,048-point paper autoencoder used by the 1,024-point classifier), the
        decoded cloud is reduced to that many points by farthest-point
        sampling, which spreads them like a real scan. Unset: unchanged."""
        X = self.decoder(z, y) if self.conditional else self.decoder(z)
        n = getattr(self, "out_points", None)
        if n and X.shape[1] > n:
            X = fps_points(X, int(n))
        return X

    def forward(self, X, y=None):
        z = self.encode(X)
        return self.decode(z, y), z
