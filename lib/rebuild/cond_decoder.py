"""Class-conditioned FC decoder: [z, embed(y)] -> 1024 x 3.

    class label y -> Embedding(40, 32) ----+
                                           |-> concat(160) -> 256 -> 256 -> 3072
    128-D AE latent z ---------------------+

WHY
---
Reconstructed clouds are recognised ~66-73% of the time against 96.5% for
real ones, and three decoder architectures with three Chamfer variants and a
feature-consistency loss all failed to close it. This hands the decoder the
class explicitly, which is legitimate for the pipeline: the EBM is already
E(z, y), so a label exists at generation time.

WHAT IT COSTS
-------------
Class accuracy stops asking "did the latent and decoder preserve THIS
object's identity" and starts asking "can a decoder that was told the answer
draw a recognisable member of that class". The second is far easier, so the
number will very likely rise, and that rise is NOT evidence the
reconstruction improved.

The failure it invites is the one rejected at the start of this work: learn
"for class y, draw the average chair" and stop reading z. Chamfer resists
only weakly, because a class-average chair is not far from any chair.

Two measurements catch it, and the second is the decisive one:

    first-layer contribution   ||W_z z||  against  ||W_y e_y||
        The embedding's own norm proves nothing -- a large embedding can be
        cancelled by small first-layer weights. What matters is how much each
        path actually contributes after the weights that consume it.

    same-class latent swap     decode(z_j, y) for a DIFFERENT object j of the
        same class. If that reconstructs object i nearly as well as its own
        latent does, the decoder is drawing a class prototype and z is
        decoration.
"""

import torch
import torch.nn as nn

LATENT = 128
N_CLASSES = 40
EMBED = 32
N_POINTS = 1024


class ClassConditionalDecoder(nn.Module):
    conditional = True

    def __init__(self, latent=LATENT, n_points=N_POINTS, n_classes=N_CLASSES,
                 embed=EMBED, hidden=(256, 256)):
        super().__init__()
        self.n_points = n_points
        self.latent = latent
        self.embed = nn.Embedding(n_classes, embed)
        layers, d = [], latent + embed
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        layers += [nn.Linear(d, n_points * 3)]
        self.net = nn.Sequential(*layers)

    def forward(self, z, y):
        e = self.embed(y)
        return self.net(torch.cat([z, e], dim=1)).view(-1, self.n_points, 3)

    @torch.no_grad()
    def path_contributions(self, z, y):
        """||W_z z|| against ||W_y e_y|| at the first layer.

        The first Linear consumes [z, e] as one 160-vector, so its weight
        splits column-wise: the first `latent` columns act on z, the rest on
        the embedding. Comparing the two products measures how much each
        path actually drives the network, which the raw embedding norm does
        not.
        """
        w = self.net[0].weight                    # (hidden, latent + embed)
        wz, wy = w[:, :self.latent], w[:, self.latent:]
        cz = (z @ wz.t()).norm(dim=1).mean()
        cy = (self.embed(y) @ wy.t()).norm(dim=1).mean()
        return float(cz), float(cy)
