"""FoldingNet decoder, and the cached-latent harness both arms share.

WHAT THIS TESTS
---------------
Reconstruction is good on Chamfer (0.00689) while reconstructed clouds are
recognised only 64.79% of the time against 96.67% for the originals. The
matched linear probe located the loss:

    old PCA-195 features        93.97%
    PointAE latent-128          87.71%     -6.26  at the encoder
    after decode -> re-encode   75.25%    -12.46  at the decoder

So the decoder is the larger bottleneck. This asks whether its ARCHITECTURE
is the reason, by holding everything else fixed.

WHY TWO ARMS AND NOT ONE
------------------------
The encoder was trained jointly WITH the FC decoder, so its latent is adapted
to that decoder. Dropping FoldingNet onto those latents and comparing against
the original jointly-trained FC decoder would handicap FoldingNet and confound
the result.

Both arms therefore train from scratch on the SAME cached latents:

    Arm A   cached latent -> a NEW fully connected decoder
    Arm B   cached latent -> FoldingNet

Same data, same latents, same Chamfer loss, same optimizer, same budget. The
decoder architecture is the only difference, and both are equally
disadvantaged by the co-adaptation.

WHAT A TIE WOULD AND WOULD NOT MEAN
-----------------------------------
A tie rules out this decoder change UNDER FROZEN LATENTS. It does not show
that Chamfer is the problem, because both arms inherit a latent shaped for
the FC decoder. It only makes a density-aware loss the next justified test.

ABOUT FOLDINGNET
----------------
It deforms a fixed 2D grid onto the surface in two folds, each conditioned on
the latent. The grid gives a useful bias toward locally coherent structure --
neighbouring grid points tend to stay neighbours on the surface. It does NOT
guarantee even spacing: the folding map can stretch or collapse regions, and
nothing in the loss forbids it.
"""

import numpy as np
import torch
import torch.nn as nn


class FoldingDecoder(nn.Module):
    """Two-fold FoldingNet. latent -> (n_points, 3).

    Fold 1 maps (grid_2d, z) to a first 3-D guess; fold 2 refines it from
    (guess_3d, z). Both folds are shared MLPs applied per point, so the
    parameter count does not scale with n_points -- unlike the flat decoder,
    which spends 256 x 3072 parameters on its output layer alone.
    """

    def __init__(self, latent=128, n_points=1024, hidden=512, grid_range=0.5):
        super().__init__()
        self.n_points = n_points
        side = int(np.ceil(np.sqrt(n_points)))
        lin = np.linspace(-grid_range, grid_range, side, dtype=np.float32)
        grid = np.stack(np.meshgrid(lin, lin), axis=-1).reshape(-1, 2)
        # register_buffer: fixed, moves with .to(device), saved in state_dict
        self.register_buffer("grid",
                             torch.from_numpy(grid[:n_points]).contiguous())

        def mlp(d_in):
            return nn.Sequential(
                nn.Conv1d(d_in, hidden, 1), nn.ReLU(),
                nn.Conv1d(hidden, hidden, 1), nn.ReLU(),
                nn.Conv1d(hidden, 3, 1))

        self.fold1 = mlp(latent + 2)
        self.fold2 = mlp(latent + 3)

    def forward(self, z):
        b = z.shape[0]
        zr = z.unsqueeze(2).expand(-1, -1, self.n_points)        # (B, L, N)
        g = self.grid.t().unsqueeze(0).expand(b, -1, -1)          # (B, 2, N)
        p1 = self.fold1(torch.cat([zr, g], 1))                    # (B, 3, N)
        p2 = self.fold2(torch.cat([zr, p1], 1))                   # (B, 3, N)
        return p2.transpose(2, 1).contiguous()                    # (B, N, 3)
