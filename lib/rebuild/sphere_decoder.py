"""Per-point decoder with a learnable sphere prior, class-conditional.

    label y -> Embedding(40, 32) --+
    latent z (195) ----------------+--> shared MLP -> refine -> that point's xyz
    this point's own anchor (3) ---+        (identical weights for every point)

WHY THIS AND NOT THE FC DECODER
-------------------------------
The FC decoder emits all n*3 coordinates from one final Linear layer. Point i
is output units 3i, 3i+1, 3i+2 and nothing whatsoever ties it to a distinct
place -- the decoder has no notion of "points" at all. Piling many outputs
onto the same location is therefore free, and Chamfer barely charges for it:
a clump costs almost nothing in the reconstruction->target direction, and the
reverse term is an average a few well-placed points can carry.

Our own measurement caught exactly that. On 480 held-out objects:

    CD (all three variants)   0.88 - 0.98 of retrieval    decoder looks better
    EMD                       1.41 of retrieval           decoder looks worse

EMD forces a one-to-one matching, so every reconstructed point must account
for exactly one target point. CD and EMD disagreeing in that direction is
the signature of point AGGREGATION, which is precisely the failure Lee et
al. attribute to FC decoders ("FC Dec. exhibits poor performance regardless
of which encoder it is combined with, despite the large training time with
numerous parameters to train").

Here each point is decoded from its own anchor through weights shared with
every other point. This is a structural prior, not an injectivity constraint:
the network can ignore anchors or map different anchors to the same point.
Density and diversity therefore still require measurement. This is
FoldingNet's argument with a 3D sphere prior in place of a 2D grid, the form
used by Generative3DCounterfactuals (dzem2912), reimplemented here because
that repository publishes no licence.

MATCHED BY CONSTRUCTION
-----------------------
At hidden = (1024, 512, 256, 128) this decoder has ~954K parameters against
the FC decoder's 914,944 -- within 4%. The comparison against the 85.21%
run is therefore a test of STRUCTURE at a fixed parameter budget, not of
capacity. That is deliberate: capacity is the other hypothesis and it gets
its own arm.

DEVIATION FROM THE SOURCE
-------------------------
The reference initialises its sphere by sampling angles uniformly at random,
which clusters anchors and leaves gaps. Anchors are initialised on a
Fibonacci sphere instead: deterministic, reproducible under our seed, and
near-uniform, so the prior starts with the spread the whole design exists to
provide. The anchors stay learnable, exactly as in the reference, so the
model can still move them.
"""

import math

import torch
import torch.nn as nn

LATENT = 195
N_CLASSES = 40
EMBED = 32
N_POINTS = 1024
HIDDEN = (1024, 512, 256, 128)


def fibonacci_sphere(n):
    """n near-uniform points on the unit sphere. Deterministic."""
    i = torch.arange(n, dtype=torch.float32)
    # z sweeps [-1, 1]; the golden angle keeps successive points maximally
    # spread in longitude, so no seed and no clustering.
    z = 1.0 - 2.0 * (i + 0.5) / n
    r = torch.sqrt(torch.clamp(1.0 - z * z, min=0.0))
    phi = i * math.pi * (3.0 - math.sqrt(5.0))
    return torch.stack([r * torch.cos(phi), r * torch.sin(phi), z], dim=1)


class SpherePriorDecoder(nn.Module):
    """Class-conditional per-point decoder. (B, latent), (B,) -> (B, n, 3)."""

    conditional = True

    def __init__(self, latent=LATENT, n_points=N_POINTS, n_classes=N_CLASSES,
                 embed=EMBED, hidden=HIDDEN):
        super().__init__()
        self.n_points = n_points
        self.latent = latent
        self.embed_dim = embed
        self.embed = nn.Embedding(n_classes, embed)
        self.anchors = nn.Parameter(fibonacci_sphere(n_points))

        d = latent + embed + 3
        layers = []
        for h in hidden:
            layers += [nn.Conv1d(d, h, 1), nn.ReLU(inplace=True)]
            d = h
        self.mlp = nn.Sequential(*layers)
        # A short per-point head, as in the reference. Kept separate from the
        # trunk so path_contributions can read the trunk's first layer.
        self.refine = nn.Sequential(
            nn.Conv1d(d, 128, 1), nn.ReLU(inplace=True),
            nn.Conv1d(128, 64, 1), nn.ReLU(inplace=True),
            nn.Conv1d(64, 3, 1))

    def forward(self, z, y):
        b, n = z.shape[0], self.n_points
        e = self.embed(y)                                     # (B, embed)
        cond = torch.cat([z, e], dim=1).unsqueeze(2).expand(-1, -1, n)
        grid = self.anchors.t().unsqueeze(0).expand(b, -1, -1)  # (B, 3, n)
        x = torch.cat([cond, grid], dim=1)                    # (B, d, n)
        return self.refine(self.mlp(x)).transpose(2, 1)       # (B, n, 3)

    @torch.no_grad()
    def path_contributions(self, z, y):
        """||W_z z|| against ||W_y e_y|| at the first shared layer.

        Same test as the FC conditional decoder: a large embedding proves
        nothing if the weights consuming it are small. The anchor columns are
        excluded -- they carry no object identity, so including them would
        flatter both paths equally and tell us nothing.
        """
        w = self.mlp[0].weight.squeeze(2)          # (h, latent + embed + 3)
        wz = w[:, :self.latent]
        wy = w[:, self.latent:self.latent + self.embed_dim]
        return (float((z @ wz.t()).norm(dim=1).mean()),
                float((self.embed(y) @ wy.t()).norm(dim=1).mean()))
