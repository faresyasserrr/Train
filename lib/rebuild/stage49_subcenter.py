"""A cosine head with SEVERAL directions per class (sub-centres).

WHY
Each class currently owns one direction in the 256-D feature space:

    f_y(x) = s * cos(h, W_y) + b_y

so a class is a point, and every object of that class must point the same way.
Classes with more than one visual mode -- a flat writing desk and a boxy desk
with drawers -- are forced onto a single arrow. Sub-centre ArcFace (Deng et
al., 2020) is the standard answer: K directions per class, and an object needs
to match any one of them.

THE HEAD

    f_y(x) = logsumexp_m [ s * cos(h, W_y,m) ] - log M

logsumexp rather than max, so every sub-centre receives gradient and the class
field stays differentiable. Subtracting log M keeps the scale comparable to the
single-centre head at M = 1, where this reduces exactly to it.

WHAT IS PRESERVED
Both properties the rest of the project rests on survive unchanged:

    scale invariance   every term is a cosine, so feature magnitude still
                       carries no information -- that is where +0.065 of the
                       OOD came from
    the marginal       E(x) = -logsumexp_y f_y(x) is still the marginal of the
                       joint density, now summing over classes AND sub-centres

EXPECTATION, STATED BEFORE THE RUN
Low. Accuracy on this trio has been measured at roughly 85 +/- 2 regardless of
what is added, and the same protocol reaches ~90 at 160 objects per class, so
the binding constraint is data, not per-class capacity. At M = 4 each class
owns 1024 parameters against 48 fit objects, which is memorising territory. The
run is worth its 14 minutes to convert an open question into a measured one,
not because a gain is expected.

NO TEST DATA IS OPENED.
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
PARENT = ROOT.parent
REPO = PARENT / "Pointnet_Pointnet2_pytorch"
for _p in (str(ROOT), str(PARENT), str(REPO), str(REPO / "models")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


class SubCenterEnergyNet(nn.Module):
    """PN++ trunk, M cosine sub-centres per class.

    forward(x) returns f_y(x) = -E(x, y), matching EnergyNet's contract, so
    argmax classifies and -logsumexp gives E(x) exactly as before.
    """

    def __init__(self, n_classes, dim=256, scale=16.0, centres=4):
        super().__init__()
        from pointnet2_cls_ssg import get_model
        self.b = get_model(num_class=n_classes, normal_channel=False)
        self.w = nn.Parameter(torch.randn(n_classes, centres, dim) * 0.05)
        self.bias = nn.Parameter(torch.zeros(n_classes))
        self.scale = scale
        self.centres = centres

    def features(self, x):
        b = self.b
        xyz = x.transpose(2, 1)
        l1_xyz, l1_points = b.sa1(xyz, None)
        l2_xyz, l2_points = b.sa2(l1_xyz, l1_points)
        _, l3_points = b.sa3(l2_xyz, l2_points)
        h = l3_points.view(len(x), 1024)
        h = b.drop1(F.relu(b.bn1(b.fc1(h))))
        return b.drop2(F.relu(b.bn2(b.fc2(h))))

    def forward(self, x):
        h = F.normalize(self.features(x), dim=1)
        w = F.normalize(self.w, dim=2)
        cos = torch.einsum("bd,cmd->bcm", h, w) * self.scale
        import math
        return torch.logsumexp(cos, dim=2) - math.log(self.centres) + self.bias
