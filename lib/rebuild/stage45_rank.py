"""Arm P: arm L, plus energy ordered by MEASURED distance from the data.

WHAT THE TWO PREVIOUS ATTEMPTS GOT WRONG
Arm N and arm N2 enforced monotonicity ALONG A PATH: for each ladder, E must
rise from rung to rung. Both failed, and in the same way:

    arm N   ladders from arm A   OOD 0.875 -> 0.845   mono 0.069 -> 0.620
    arm N2  ladders from arm L   OOD 0.875 -> 0.800   mono 0.068 -> 0.283

The mono term GREW in both runs -- it never converged -- while the mean ladder
profile was already monotone (the printed dip was 0.00 throughout). So the
average ordering held and the loss was chasing per-path wobble: a single ladder
is five points decoded from one sample, any of which can sit a little out of
line for reasons that have nothing to do with distance from the data. Those
wobbles are unsatisfiable, so the gradient never stops, and it reshapes the
shared features until detection degrades. Arm N2 also raised classification
accuracy by 3.3 while losing 0.075 of AUROC, which says exactly where the
capacity went.

TWO STRUCTURAL FAULTS, NOT ONE TUNING PROBLEM

  THE UNIT WAS THE PATH.  A path imposes an ordering on five specific points
  and demands it hold for every one of them individually. Nothing averages.

  THE REAL OBJECTS WERE NOT IN IT.  Both losses ordered negatives among
  themselves only. Nothing anchored the low end, so the whole set could drift
  upward and satisfy the constraint without ever expressing "real objects sit
  lowest" -- which is the property detection actually needs. The constraint ran
  ACROSS the wall loss instead of along it.

THE CHANGE
Order by measured geometry, across the whole pool, with the real data in it.

    pool      real objects          d = 0            (they are the reference)
              decoded real objects  d measured
              ladder rungs          d measured
              holes                 d measured

    sample pairs (i, j) with  d_j > d_i + delta
    require                   E_i <= E_j

    L_rank = mean relu( E_i - E_j ) / sd_real

Why each part answers a measured fault:

  PAIRS, NOT PATHS.  Pairs are drawn across different objects and different
  paths, so a single wobbling sample appears in many pairs with both signs and
  averages out instead of becoming a permanent violation. The loss can reach
  zero.

  THE DELTA GAP.  No ordering is ever demanded between two points at nearly the
  same distance -- which is precisely the case that was unsatisfiable. delta is
  in class rulers, the same unit every distance in this project uses.

  REAL OBJECTS INCLUDED AT d = 0.  Every pair whose low side is a real object
  states "this real object must sit below that fake one" -- the wall loss's own
  claim. The ranking objective therefore CONTAINS detection as its first case
  and reinforces it, rather than competing for the same features.

  NO ABSOLUTE LEVEL.  Only differences appear, divided by the in-batch real
  spread, so the term is shift-invariant and scale-free and cannot ratchet the
  way the path loss did (arm N inflated every rung by about 10 into the cosine
  head's bounded range).

NOTHING NEW IS GENERATED
Every distance used here was measured when the holes and ladders were made and
is stored in those files. No search runs, and the hole set is arm L's,
unchanged.

EVERYTHING ELSE IS ARM L
Same holes, same split, same head, same real slots, same step count, same
learning rate. The ranking term is the only difference.

NO TEST DATA IS OPENED.
"""

import argparse
import json
import sys
import time
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

from stage0_real_only import auroc                               # noqa: E402
from stage13_decoded_positives import CosineEnergyNet            # noqa: E402


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--holes", type=Path, default=ROOT / "holes_joint_armA.npz")
    ap.add_argument("--ladders", type=Path, default=ROOT / "ladders_armA.npz",
                    help="rungs enter the pool as distance-labelled points; "
                         "their path structure is not used")
    ap.add_argument("--arm-a", type=Path, default=ROOT / "stage0_hard60_s0.pth")
    ap.add_argument("--baseline", type=Path, default=ROOT / "stage24_armL_s0.pth")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--hole-slots", type=int, default=8)
    ap.add_argument("--rung-slots", type=int, default=8,
                    help="distance-labelled points drawn from the ladder pool")
    ap.add_argument("--delta", type=float, default=0.10,
                    help="minimum distance gap, in class rulers, before a pair "
                         "is asked to be ordered; skips the pairs that were "
                         "unsatisfiable under the path loss")
    ap.add_argument("--centres", type=int, default=1,
                    help="cosine sub-centres per class (stage49). 1 is the "
                         "single-direction head every arm so far has used and "
                         "is bit-identical to it.")
    ap.add_argument("--augment", action="store_true",
                    help="rotate about the up-axis and rescale every cloud in "
                         "the batch, independently per sample. Nothing in this "
                         "project has ever been augmented: load_clouds returns "
                         "one cached array, so the model sees the same 180 "
                         "objects in the same pose at every one of 2000 steps, "
                         "which is what memorising 60 objects per class looks "
                         "like. Jitter is deliberately NOT included -- the "
                         "pilot in build_views.py measured it at 94.1% label "
                         "accuracy against 97.4% for the alternatives and "
                         "rejected it; rotation and scaling were never tested. "
                         "EVERY cloud is transformed, negatives included: "
                         "augmenting only the positives would make 'was "
                         "rotated' a perfect cue for 'real', the same kind of "
                         "free shortcut the raw-vs-decoded cue was.")
    ap.add_argument("--anchor", action="store_true",
                    help="add alpha * mean(E(real)^2), holding the energy of real "
                         "objects at 0; alpha is set at the first step so the "
                         "anchor term equals the classification term")
    ap.add_argument("--jitter-real", type=float, default=0.0,
                    help="add Gaussian point jitter to the raw real objects of "
                         "every batch, strength drawn per object uniformly in "
                         "[0, this]. They keep their class, so slightly noisy "
                         "known objects are trained as known. 0 = off.")
    ap.add_argument("--rot-axis", type=int, default=1,
                    help="up-axis for the rotation; ModelNet clouds are y-up")
    ap.add_argument("--scale-lo", type=float, default=0.9)
    ap.add_argument("--scale-hi", type=float, default=1.1)
    ap.add_argument("--positives", type=Path, default=None,
                    help="extra in-distribution samples to use as positives "
                         "(stage48). They take a SHARE of the existing real "
                         "slots rather than adding slots, so batch size and "
                         "compute stay identical to the arm without them and "
                         "only the content of the positive pool changes.")
    ap.add_argument("--min-d", type=float, default=0.0,
                    help="only order pairs where BOTH points already lie this "
                         "far from the data, in class rulers. Arm P ordered "
                         "everything including the real objects at d = 0, "
                         "which states 'close to the data means low energy' -- "
                         "and the categories it then lost were exactly the "
                         "geometrically nearest ones (mantel -0.257, table "
                         "-0.220, bookshelf -0.219, laptop -0.219, bed -0.218, "
                         "all flat boxy furniture). Near-OOD detection needs "
                         "the opposite of that claim. Setting this above the "
                         "real-object neighbourhood keeps the ordering where "
                         "the defect actually is -- the dip runs from d 0.468 "
                         "to 0.604 -- while saying nothing about near points. "
                         "The wall loss still anchors the low end.")
    ap.add_argument("--per-class-wall", action="store_true",
                    help="measure each negative against the real objects of "
                         "ITS OWN class, instead of against the pooled real "
                         "mean. Measured: the spread of per-class real "
                         "energies is 0.52 sd on the three-class trio and "
                         "0.58 on the six-class set that worked, but 2.04 on "
                         "the six-class set that did not -- bottle at -0.92 "
                         "against flower_pot at +1.12. A pooled 2-sd margin "
                         "then asks flower_pot's negatives to clear 0.88 sd "
                         "above their own class while bottle's must clear "
                         "2.92, so the same rule is three times harder for "
                         "one class than another. Per class, the margin means "
                         "the same thing everywhere. Prediction with a "
                         "built-in control: this should help where the spread "
                         "is large and do almost nothing where it is small.")
    ap.add_argument("--freeze-head-in-rank", action="store_true",
                    help="compute the ranking term through a DETACHED copy of "
                         "the class vectors, so the ordering loss shapes the "
                         "features but never moves W_y; classification and the "
                         "wall loss still move it normally. Motivated by a "
                         "measurement: cos(W_y, mu_y), the alignment between "
                         "each class vector and its own feature cluster, is "
                         "0.321 -> 0.318 when the ordering is added at THREE "
                         "classes (no effect) but 0.352 -> 0.245 at SIX "
                         "(a 30% drop), while ||W|| grows 1.067 -> 1.278. The "
                         "class vectors are being driven off their own data. "
                         "That splits the two readouts exactly as observed: "
                         "kNN reads h alone and improves to 0.862, while the "
                         "energy reads cos(h, W_y) and falls to 0.794. If the "
                         "drift is the cause, freezing W_y here should recover "
                         "the energy while leaving kNN where it is.")
    ap.add_argument("--head", type=str, default="cosine",
                    choices=["cosine", "distance", "linear"],
                    help="readout. 'cosine' scores by ANGLE only: "
                         "f_y = s*cos(h, W_y) + b_y, so a class's accepted "
                         "region is a cone reaching outward forever and "
                         "nothing depends on how FAR a point sits from the "
                         "class. Measured at six classes, the energy scores "
                         "0.794 while Mahalanobis and kNN on the SAME frozen "
                         "features reach 0.853 and 0.862 -- and both of those "
                         "use distance. 'distance' is the bounded quadratic "
                         "head (stage4), which has a radial term. It was "
                         "tried once, as arm C, and scored 0.597 -- but that "
                         "run had the wall loss comparing a borrowed med_real "
                         "against quadratic energies, so the wall term read "
                         "0.0000 for its entire duration. That bug was fixed "
                         "later and this head was never retested, so the one "
                         "readout addressing the measured weakness has never "
                         "had a valid trial.")
    ap.add_argument("--wide", action="store_true",
                    help="read the 512-D trunk layer instead of the 256-D one. "
                         "Tests whether the OOD/ordering conflict at six "
                         "classes is a capacity limit: the correlation between "
                         "them is +0.99 / +0.98 / +0.84 on the three trios and "
                         "-0.56 on six classes. If capacity is the cause the "
                         "correlation should return to positive; if it stays "
                         "negative the claim is bounded at three classes, with "
                         "a measured reason rather than a guess.")
    ap.add_argument("--same-class", action="store_true",
                    help="order only pairs drawn from the SAME class. Every "
                         "distance in this project is divided by a ruler "
                         "averaged over the classes, so a bottle's 0.3 and a "
                         "flower_pot's 0.3 do not stand for the same real "
                         "distance -- the rulers span 4.29x on the six-class "
                         "set. A cross-class pair therefore asserts an "
                         "ordering that the numbers do not support, and a loss "
                         "enforcing an unsupported ordering is a loss chasing "
                         "noise, which arms N and N2 measured as costing "
                         "0.030-0.075 of AUROC. Restricting to same-class "
                         "pairs makes the unit cancel: both points are divided "
                         "by the same thing, whatever it is. The smoothness "
                         "measurement (stage46) has always paired within a "
                         "class for exactly this reason; the loss did not, "
                         "which went unnoticed while every trio had nearly "
                         "equal rulers. With six classes five pairs in six are "
                         "cross-class.")
    ap.add_argument("--min-d-rel", type=float, default=0.0,
                    help="per-class cut, replacing the single --min-d. Each "
                         "class gets its own threshold: this multiple of that "
                         "class's OWN near-data floor, measured as the median "
                         "distance of rung x1. A single absolute cut leaves "
                         "whole classes out of the ordering -- at 0.40 on the "
                         "vessel trio bottle contributed 18% of its points "
                         "against flower_pot's 45%, and normalising by "
                         "per-class rulers only flipped which class was left "
                         "out (53% / 19%). The floor is per class because the "
                         "real-object neighbourhood is per class, so a cut "
                         "expressed in it means the same thing everywhere. "
                         "1.4 reproduces the 0.40 already used on the "
                         "furniture trio, whose x1 floor is 0.286.")
    ap.add_argument("--smooth-tol", type=float, default=-1.0,
                    help="slack, in standard deviations of the real energies, "
                         "before an OVERSHOOT is penalised under --smooth. "
                         "The hard two-sided form fought the wall loss "
                         "directly: the wall never reached zero (0.052 at "
                         "step 1500, against 0.0002 in every working arm) and "
                         "inverted categories climbed 1 -> 2 -> 4, because a "
                         "ceiling on how far apart two far points may sit is "
                         "a ceiling on how high the wall may push a hole. "
                         "With slack the constraint only catches runaway "
                         "gaps, which is the regularised form (penalise "
                         "excessive change, not any change). Default -1 uses "
                         "the wall's own margin, so nothing new is chosen.")
    ap.add_argument("--smooth", action="store_true",
                    help="two-sided version of --prop-margin: the energy gap "
                         "must be ABOUT alpha*(d_j - d_i), not merely at least "
                         "it. This is the smoothness constraint -- equal "
                         "distance steps should cost equal energy -- and it "
                         "implies the one-sided ordering. It was unsafe while "
                         "real objects were inside the ordering, because a "
                         "two-sided constraint puts a CEILING on energy gaps "
                         "and would have capped the real/unseen gap that "
                         "detection depends on. --min-d-rel excludes real "
                         "objects from the ordering, so the ceiling now only "
                         "ever applies between two far points. Implies "
                         "--prop-margin; alpha is the same calibration, the "
                         "wall's margin at the median hole distance.")
    ap.add_argument("--prop-margin", action="store_true",
                    help="require the energy gap to grow WITH the distance "
                         "gap, instead of merely being positive. Arm P showed "
                         "why this matters: a margin-free ordering is "
                         "satisfied by an even ramp, which spends the head's "
                         "range on fine distinctions between fake points and "
                         "leaves the real/unseen gap no larger than any other. "
                         "The required gap is margin*sd at the median hole "
                         "distance and scales linearly, so the pair "
                         "(real, hole) asks for exactly what the wall loss "
                         "already asks. No new quantity is chosen.")
    ap.add_argument("--lam-rank", type=float, default=1.0)
    ap.add_argument("--scale", type=float, default=16.0)
    ap.add_argument("--ae", type=Path,
                    default=ROOT / "pointae_sphere_unsquared_L195_"
                                   "h1024x512x256x128_s0.pth")
    ap.add_argument("--dec-bank", type=Path, default=None,
                    help="take the latents of the decoded positives from this "
                         "latent bank (same object order as the train clouds) "
                         "instead of encoding the 1,024-point clouds. Needed "
                         "for an autoencoder trained on 2,048 points, whose "
                         "encoder gives wrong latents for 1,024-point input. "
                         "Default: encode, as before.")
    ap.add_argument("--lam-in", type=float, default=0.0,
                    help="weight on the IN-DISTRIBUTION energy hinge, the "
                         "half of Liu et al.'s (NeurIPS 2020) two-sided "
                         "formulation this project never had. Their fine-"
                         "tuning is relu(E(x_in) - m_in)^2 pushing real "
                         "energies DOWN, together with relu(m_out - E(x_out))^2 "
                         "pushing negatives UP. We only ever had the second: "
                         "the wall. Cross-entropy constrains differences "
                         "between logits, not the energy level, so nothing "
                         "in the objective has ever pushed a real object's "
                         "energy down. That level is what FPR@95TPR reads -- "
                         "the threshold is the 95th percentile of the real "
                         "energies, so the worst 5% set it alone. IMPORTANT "
                         "DISTINCTION: this constrains a LEVEL, one point at "
                         "a time. Every constraint that has cost detection in "
                         "this project constrained a GRADIENT -- how fast the "
                         "energy changes between two points -- which flattens "
                         "the rise away from the data, and that rise is the "
                         "detector. Whether a level constraint behaves "
                         "differently is untested, and that is the point of "
                         "the run. 0 disables it.")
    ap.add_argument("--m-in", type=float, default=1.0,
                    help="the hinge threshold, in real standard deviations "
                         "ABOVE the batch's mean real energy, so the loss is "
                         "relu(E - (mu + m_in*sd))^2. At 1.0 it acts only on "
                         "real objects more than one sd above the mean -- "
                         "roughly the worst 16%, the tail that sets FPR@95TPR "
                         "-- and is exactly zero for the rest. A first "
                         "attempt used -1.0, which puts the threshold one sd "
                         "BELOW the mean and so fires on about 84% of real "
                         "objects: accuracy collapsed to 41% and the term "
                         "never converged. Expressed relative to the batch's "
                         "own statistics rather than as an absolute energy, "
                         "so it means the same thing whatever the head's "
                         "scale.")
    ap.add_argument("--margin", type=float, default=2.0)
    ap.add_argument("--lam-lip", type=float, default=0.0,
                    help="weight on a LOCAL LIPSCHITZ penalty: for pairs of "
                         "clouds that are CLOSE to each other, cap how fast "
                         "the energy may change between them. The ordering "
                         "term sets a floor on the rise between well-"
                         "separated pairs and says nothing about nearby "
                         "ones, and the measurement shows the cost: on the "
                         "trio 4.6% of local pairs jump more than 10 real sd "
                         "per class ruler, against 47% on the six-class set "
                         "A and 38% on set B. The field orders correctly and "
                         "is still full of cliffs. This penalises "
                         "relu(|E_i - E_j| / d_ij - tol) over pairs whose "
                         "Chamfer distance is at most --lip-window, which "
                         "leaves the large data-to-negative gap that "
                         "detection depends on completely untouched -- that "
                         "gap is not a close pair. Two earlier two-sided "
                         "constraints failed (0.918 -> 0.730 and -> 0.787), "
                         "but both capped every pair including the far ones, "
                         "and both ran on the trio where the cliff rate was "
                         "already 4.6% and there was nothing to gain. "
                         "0 disables it.")
    ap.add_argument("--lip-tol", type=float, default=30.0,
                    help="the allowed slope, in real standard deviations of "
                         "energy per class ruler of Chamfer distance. Read "
                         "off the trained models rather than chosen: under "
                         "this construction the slope runs p50 27.2, p75 "
                         "56.4, p95 184.2 on set A and p50 19.1, p95 138.6 "
                         "on set B, so 30 sits at about the median and the "
                         "loss acts on the tail while leaving half the "
                         "points alone. A first attempt used 10, taken from "
                         "the PAIRWISE measurement, which is a different "
                         "construction on a different scale: it sat below "
                         "the median, fired on nearly every point, and "
                         "reached 92 against a classification loss of 2.3.")
    ap.add_argument("--lam-tan", type=float, default=0.0,
                    help="weight on a TANGENT smoothness penalty. Every "
                         "smoothness term tried so far perturbed the cloud "
                         "isotropically, and almost every such direction "
                         "leaves the data manifold -- which is the direction "
                         "the energy MUST rise in, because that rise is the "
                         "detector. Measured cost: 0.08 AUROC at both 0.5 "
                         "and 1.0 weight, with the damage falling on the "
                         "energy readout while Mahalanobis on the same "
                         "feature stayed at 0.95. This term instead moves "
                         "ALONG the manifold: the latent of a real object is "
                         "nudged toward a same-class neighbour and decoded, "
                         "so the twin is another plausible object of the "
                         "same class. Energy should not vary between them, "
                         "and nothing about detection requires that it does. "
                         "This is tangent propagation (Simard et al.), which "
                         "was introduced to improve CLASSIFICATION, and the "
                         "confusion matrix says our errors are confident "
                         "rather than borderline -- the kind a tighter class "
                         "manifold acts on. 0 disables it.")
    ap.add_argument("--freeze-head-in-tan", action="store_true",
                    help="let the tangent term reach the backbone but not the "
                         "class vectors. The measurement is specific: under "
                         "the tangent penalty the ENERGY readout falls to "
                         "0.89 while Mahalanobis on the same feature stays "
                         "at 0.95, so the representation survives and the "
                         "head is what degrades. The ordering loss had the "
                         "identical symptom and --freeze-head-in-rank fixed "
                         "it, 0.794 -> 0.818. Detaching W and the bias here "
                         "lets the term flatten the feature along the "
                         "manifold without moving the vectors that decide "
                         "where the energy sits.")
    ap.add_argument("--tan-clearance", type=float, default=0.0,
                    help="apply the tangent penalty only to the fraction of "
                         "real objects that sit FURTHEST from any negative, "
                         "between 0 and 1. Flattening widens the basin, and "
                         "widening hurts most where the basin edge is "
                         "already close -- an object with a negative nearby "
                         "has no room to spare. Objects deep inside their "
                         "class have room, and flattening there costs "
                         "nothing the detector needs. 0.5 keeps the roomier "
                         "half; 0 disables the filter and flattens "
                         "everywhere, which is what the first tangent runs "
                         "did.")
    ap.add_argument("--tan-eps", type=float, default=0.15,
                    help="how far along the segment toward the same-class "
                         "neighbour the twin sits, as a fraction. Small "
                         "enough that the twin is unambiguously the same "
                         "class, large enough that the decoded shape "
                         "actually differs.")
    ap.add_argument("--lip-where", choices=["batch", "rungs", "generated"],
                    default="batch",
                    help="which clouds get a perturbed twin. 'batch' is any "
                         "of them, which is what the first attempt did and "
                         "what cost 0.075 AUROC on set B. Two measurements "
                         "say that was the wrong target: the steepest points "
                         "in the field are the REAL objects (gradient norm "
                         "median 22, p95 117), and that steepness is the "
                         "detector -- a real object must sit in a narrow low "
                         "basin. Meanwhile the roughness being complained "
                         "about is measured among GENERATED points, on the "
                         "rung and hole pool, not at real objects at all. "
                         "'rungs' restricts the penalty to the rung slots, "
                         "the corridor between data and negatives, where "
                         "smoothness is the property wanted and where "
                         "neither the classification loss nor the wall has a "
                         "stake. 'generated' widens it to the holes as well.")
    ap.add_argument("--lip-n", type=int, default=8,
                    help="how many clouds of the batch get a perturbed twin "
                         "each step. Each one costs two extra forward passes, "
                         "so this trades directly against step time.")
    ap.add_argument("--lip-eps", type=float, default=0.01,
                    help="size of the perturbation, the same 0.01 the "
                         "smoothness measurement uses for its first metric, "
                         "so the training signal and the reported number are "
                         "the same quantity.")
    ap.add_argument("--lip-window", type=float, default=0.25,
                    help="how close two clouds must be, in class rulers, to "
                         "count as a local pair. The same window the "
                         "smoothness measurement uses.")
    ap.add_argument("--lip-pairs", action="store_true",
                    help="fill the rung slots with ADJACENT rungs of the same "
                         "ladder instead of drawing them independently from "
                         "the flattened pool. The Lipschitz penalty needs "
                         "pairs of clouds that are close to each other, and "
                         "two clouds picked at random almost never are: a "
                         "first attempt found the term at 0.0000 from step "
                         "40 onward because no pair in the batch fell inside "
                         "the window. Rungs t and t+1 of one ladder are close "
                         "by construction -- they are consecutive points on a "
                         "single segment -- so drawing them together supplies "
                         "exactly the local pairs the measurement forms. Only "
                         "the rung draw changes; the holes and real objects "
                         "are untouched, and the ordering term sees the same "
                         "material it always did.")
    ap.add_argument("--lip-floor", type=float, default=0.05,
                    help="pairs closer together than this, in class rulers, "
                         "are dropped. A raw real object and its own decoded "
                         "reconstruction are almost the same cloud, so their "
                         "d_ij is near zero and the ratio |dE|/d_ij explodes "
                         "for arithmetic reasons rather than because the "
                         "field is rough -- a first attempt without this "
                         "floor saw the term reach 1051 within 40 steps. The "
                         "smoothness measurement never forms such a pair: it "
                         "compares distinct generated points, not an object "
                         "against its own reconstruction.")
    ap.add_argument("--lip-points", type=int, default=64,
                    help="points subsampled per cloud for the in-batch "
                         "Chamfer distances. The full pairwise distance over "
                         "1024 points every step would dominate the run; 64 "
                         "is enough to rank which pairs are local.")
    ap.add_argument("--pos-from-dec", action="store_true",
                    help="take the generated positives' batch slots from the "
                         "DECODED share instead of from the raw real objects. "
                         "Without this, supplying --positives silently drops "
                         "the raw real share from half the real slots to a "
                         "third: the three-way split gives raw, decoded and "
                         "positives a third each. The one run that used "
                         "positives lost 5 points of accuracy on the trio "
                         "(arm V 82.5% against arm P3's 87.5%) while OOD "
                         "barely moved, which is what replacing a third of "
                         "the genuine training signal would do on its own. "
                         "With this flag the raw share stays at half and the "
                         "positives come out of the decoded objects, which "
                         "are reconstructions of those same raw objects and "
                         "carry no label information the raw ones do not.")
    ap.add_argument("--balanced", action="store_true",
                    help="draw the real slots one class at a time instead of "
                         "uniformly over objects. Set A is imbalanced 3:1 -- "
                         "vase has 374 fit objects against flower_pot's 121 "
                         "-- and the errors run along exactly that axis: 8 "
                         "flower_pots are called vase against 5 the other "
                         "way. The loss already carries inverse-frequency "
                         "class weights, but weighting scales a gradient it "
                         "still only receives a third as often; balanced "
                         "SAMPLING fixes the count rather than the size. "
                         "Because the two corrections would otherwise "
                         "compound, this flag also sets the cross-entropy "
                         "weights to 1 -- the balance is applied once, in "
                         "the sampler. Holes and rungs are untouched: they "
                         "are already capped per class by --ratio.")
    ap.add_argument("--lam-comp", type=float, default=0.0,
                    help="weight on the COMPLEMENTARY-LABEL term borrowed "
                         "from Lyu & Beksi (2026). Every negative was "
                         "searched from a known class's latent and class "
                         "embedding, so the one label it is NOT is known. "
                         "The wall only moves logsumexp, a single scalar: "
                         "all logits can rise together, leaving the wall "
                         "satisfied while the negative still reads as a "
                         "confident member of its source class. This term "
                         "acts per logit instead -- -log(1 - p_y) on the "
                         "source class -- and drives that one probability "
                         "down. Expected to be small here, because the "
                         "logits are already strongly peaked and 1 - p_y is "
                         "near zero for the top class, so the gradient is "
                         "thin. 0 disables it, which is the default and "
                         "leaves every earlier run reproducible.")
    ap.add_argument("--energy-head", choices=("jem", "margin"), default="jem",
                    help="how the per-class outputs are read and trained. "
                         "'jem': they are logits -- classification by "
                         "softmax cross-entropy, the input energy is "
                         "-logsumexp, the complementary term uses softmax "
                         "probabilities (Grathwohl et al. 2020). 'margin': "
                         "they are class energies E(x,y) = -f_y and nothing "
                         "is ever normalised -- classification by a margin "
                         "loss requiring E(x,true) to sit --cls-margin below "
                         "every other class's energy, the input energy is "
                         "the best class's energy min_y E(x,y), and the "
                         "complementary term requires a negative's SOURCE "
                         "class energy to sit --cls-margin above the best "
                         "other class's (LeCun et al. 2006). No softmax "
                         "anywhere in training or readout")
    ap.add_argument("--energy-readout", choices=("auto", "free", "best"),
                    default="auto",
                    help="the scalar input energy used by the wall, the "
                         "ranking and the evaluation. 'free': the free "
                         "energy over class energies, -log sum_y "
                         "exp(-E(x,y)) (LeCun et al. 2006) -- every class "
                         "receives gradient. 'best': the best class's energy "
                         "min_y E(x,y) -- only the top class receives "
                         "gradient. 'auto': free for the jem head, best for "
                         "the margin head. The free energy is not softmax: "
                         "it is the normaliser softmax divides by, used here "
                         "as an energy, and no probability is ever formed")
    ap.add_argument("--cls-smooth", action="store_true",
                    help="margin head only. Replace the hinge max(0, m - gap) "
                         "by its smooth form log(1 + exp(m - gap)): the same "
                         "pairwise margin, but the push never drops to "
                         "exactly zero once the margin is met, as the "
                         "cross-entropy push never does. Still no softmax: "
                         "each true/wrong pair is scored on its own, with "
                         "no normalisation over classes")
    ap.add_argument("--cls-loss", choices=("closest", "all"),
                    default="closest",
                    help="margin head only. 'closest': the true class's "
                         "energy must sit --cls-margin below the closest "
                         "wrong class (Crammer & Singer 2001) -- only that "
                         "one class is trained per example. 'all': below "
                         "EVERY wrong class (Weston & Watkins 1999), so all "
                         "class energies are trained on every example, as "
                         "with cross-entropy, but with no softmax")
    ap.add_argument("--cls-margin", type=float, default=9.7,
                    help="energy gap the margin head demands. 9.7 is the "
                         "median gap between the true class and the best "
                         "other class that the jem-trained production model "
                         "(stage155) achieves on held-out objects, so the "
                         "margin head is asked for exactly the separation "
                         "the model it replaces already delivers")
    ap.add_argument("--lam-wall", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--ood-cap", type=int, default=120)
    ap.add_argument("--out", type=Path, default=ROOT / "stage45_armP_s0.pth")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if args.out.exists() and not args.overwrite:
        raise SystemExit(f"refusing to overwrite {args.out}; pass --overwrite")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    import pointnet2_features as P
    P.patch_deterministic_fps(True)
    from decoder import load_clouds
    with (P.DATA_ROOT / "modelnet40_shape_names.txt").open() as handle:
        names = [line.rstrip() for line in handle]

    a = torch.load(args.arm_a, map_location="cpu", weights_only=False)
    classes, cname = a["classes"], a["class_names"]
    C = len(classes)
    remap = {c: i for i, c in enumerate(classes)}
    fit_src = np.array(a["fit_sources"])
    val_src = np.array(a["val_sources"])

    d = np.load(args.holes, allow_pickle=True)
    prov = json.loads(str(d["provenance_json"]))
    if prov["fit_sources"] != fit_src.tolist():
        raise SystemExit("holes come from a different split than arm A")
    keep = np.where(d["energy"] < float(prov["median_real_energy"]))[0]
    H = torch.from_numpy(d["clouds"][keep]).float()
    H_d = d["d_real"][keep].astype(np.float32)
    H_y = d["label"][keep]

    ld = np.load(args.ladders, allow_pickle=True)
    lprov = json.loads(str(ld["provenance_json"]))
    if lprov["fit_sources"] != fit_src.tolist():
        raise SystemExit("ladders come from a different split than arm A")
    # rungs enter as a flat, distance-labelled pool; x0 is the decoded real
    # object and is kept, because it is the low end the ordering needs
    R = torch.from_numpy(ld["clouds"].reshape(-1, ld["clouds"].shape[2], 3)).float()
    R_d = ld["d_real"].reshape(-1).astype(np.float32)
    R_y = np.repeat(ld["label"], ld["d_real"].shape[1])
    n_rungs = int(ld["d_real"].shape[1])

    Xtr, ytr = load_clouds("train")
    Xtr = torch.as_tensor(np.asarray(Xtr)).float()
    ytr = np.asarray(ytr)
    y_fit = np.array([remap[int(v)] for v in ytr[fit_src]])
    y_val = np.array([remap[int(v)] for v in ytr[val_src]])
    X_val = Xtr[val_src]

    from autoencoder import PointAE, canonical_order
    ack = torch.load(args.ae, map_location="cpu", weights_only=False)
    ae = PointAE(ack["latent"], n_points=ack["n_points"],
                 decoder=ack["decoder"], hidden=tuple(ack["hidden"])).to(dev).eval()
    ae.load_state_dict(ack["state_dict"])
    ae.out_points = ack.get("out_points")
    ae.requires_grad_(False)
    dec = []
    Zbank = (torch.from_numpy(np.load(args.dec_bank)["latents"]).float()
             if args.dec_bank is not None else None)
    with torch.no_grad():
        for i in range(0, len(fit_src), 48):
            sl = fit_src[i:i + 48]
            z = (Zbank[sl].to(dev) if Zbank is not None
                 else ae.encode(Xtr[sl].to(dev)))
            dec.append(canonical_order(
                ae.decode(z, torch.from_numpy(ytr[sl]).long().to(dev))).cpu())
    X_dec = torch.cat(dec)
    y_dec = y_fit.copy()

    # tangent twins: the same object's latent nudged toward a same-class
    # neighbour, then decoded. Both ends are decoded, so the only difference
    # between a twin and its object is the latent move -- a step ALONG the
    # class manifold rather than off it. Built once, like X_dec.
    X_tan = None
    if args.lam_tan > 0:
        with torch.no_grad():
            Z_fit = torch.cat([ae.encode(Xtr[fit_src[i:i + 48]].to(dev)).cpu()
                               for i in range(0, len(fit_src), 48)])
        nb = np.empty(len(fit_src), dtype=np.int64)
        for k in range(C):
            idx = np.where(y_fit == k)[0]
            zk = Z_fit[idx].to(dev)
            dm = torch.cdist(zk, zk)
            dm.fill_diagonal_(float("inf"))
            nb[idx] = idx[dm.argmin(1).cpu().numpy()]
        tan = []
        with torch.no_grad():
            for i in range(0, len(fit_src), 48):
                sl = slice(i, i + 48)
                z0 = Z_fit[sl].to(dev)
                z1 = Z_fit[nb[sl]].to(dev)
                zt = z0 + args.tan_eps * (z1 - z0)
                tan.append(canonical_order(ae.decode(
                    zt, torch.from_numpy(ytr[fit_src[sl]]).long().to(dev))).cpu())
        X_tan = torch.cat(tan)
        tan_ok = np.ones(len(fit_src), bool)
        if args.tan_clearance > 0:
            # distance from each real object to its nearest negative, in
            # rulers; keep only the objects with the most room
            sub = (np.arange(len(H)) if len(H) <= 1500
                   else np.random.default_rng(args.seed).choice(
                       len(H), 1500, replace=False))
            clr = nearest_cd(Xtr[fit_src], H[sub], dev, 256) / ruler
            cut = float(np.quantile(clr, 1.0 - args.tan_clearance))
            tan_ok = clr >= cut
            print(f"  tangent    applied to {int(tan_ok.sum())} of "
                  f"{len(tan_ok)} objects, those at least {cut:.3f} rulers "
                  f"from any negative")

    X_pos = y_pos = None
    if args.positives is not None:
        pd_ = np.load(args.positives, allow_pickle=True)
        pprov = json.loads(str(pd_["provenance_json"]))
        if pprov["fit_sources"] != fit_src.tolist():
            raise SystemExit("positives come from a different split")
        X_pos = torch.from_numpy(pd_["clouds"]).float()
        y_pos = np.array([remap[int(v)] for v in pd_["label"]])
    # the decoded real objects sit at the distance measured for rung x0
    dec_d = float(np.median(ld["d_real"][:, 0]))
    # each class's near-data floor: how far rung x1 sits from the real data
    # for objects of that class. Measured, not chosen.
    floor = {int(c): float(np.median(ld["d_real"][ld["label"] == c, 1]))
             for c in np.unique(ld["label"])}
    cut_of_R = np.array([args.min_d_rel * floor[int(c)]
                         for c in np.repeat(ld["label"], ld["d_real"].shape[1])],
                        np.float32)
    cut_of_H = np.array([args.min_d_rel * floor[int(c)]
                         for c in d["label"][keep]], np.float32)
    # the distance at which the required energy gap equals the wall's margin
    d_ref = float(np.median(H_d))
    # the class ruler the hole distances were divided by, so an in-batch
    # Chamfer can be put in the same unit as H_d and the measurement
    ruler = float(prov["ruler"])
    unseen = [c for c in sorted(set(int(v) for v in ytr))
              if c not in classes and (ytr == c).sum() >= 40]
    y_hole = np.array([remap[int(v)] for v in H_y])
    y_rung = np.array([remap[int(v)] for v in R_y])
    n_real = args.batch - args.hole_slots

    bl = None
    if args.baseline.is_file():
        bl = torch.load(args.baseline, map_location="cpu",
                        weights_only=False)["log"][-1]

    print("=" * 78)
    print("STAGE 45   ARM P   energy ordered by MEASURED distance, not by path")
    print("=" * 78)
    print(f"  classes    {cname}")
    print(f"  real       {len(fit_src)} fit objects, holes {len(H)}")
    print(f"  rung pool  {len(R)} distance-labelled points, "
          f"d_real {R_d.min():.2f}-{R_d.max():.2f}")
    print(f"  hole pool  {len(H)} points, d_real {H_d.min():.2f}-{H_d.max():.2f}")
    print(f"  raw real   d = 0.000    decoded real   d = {dec_d:.3f}")
    if args.smooth:
        tol = args.margin if args.smooth_tol < 0 else args.smooth_tol
        print(f"  ordering   SMOOTH: gap tracks distance ({args.margin}*sd at "
              f"d = {d_ref:.3f}), overshoot free up to {tol}*sd")
    elif args.prop_margin:
        print(f"  ordering   gap must reach {args.margin}*sd at "
              f"d = {d_ref:.3f} rulers, scaling linearly with distance")
    else:
        print(f"  ordering   relu(E_i - E_j)/sd_real over pairs with "
              f"d_j > d_i + {args.delta}   (no margin: any rise satisfies it)")
    print(f"  batch      {n_real} real + {args.hole_slots} holes + "
          f"{args.rung_slots} rungs   {args.steps} steps, lr {args.lr}")
    if X_pos is not None:
        # stage48 records the ball as "radius", stage23 as "rz"; hard
        # positives come from the latter
        rad = pprov.get("radius", pprov.get("rz", "?"))
        kind = pprov.get("kind", "perturbed")
        split = ("raw kept at half, generated taken from the decoded share"
                 if args.pos_from_dec else
                 "the real slots split three ways (raw / decoded / generated)")
        print(f"  positives  {len(X_pos)} {kind}, d_real median "
              f"{float(np.median(pd_['d_real'])):.3f}, radius "
              f"{rad} x D_class; {split}")
    if args.same_class:
        print("  pairs      SAME CLASS only; the averaged ruler then cancels")
    if args.min_d_rel > 0:
        nr = int((R_d >= cut_of_R).sum())
        nh = int((H_d >= cut_of_H).sum())
        print(f"  per-class cut  {args.min_d_rel} x each class's own x1 floor")
        for c in sorted(floor):
            mR = np.repeat(ld["label"], ld["d_real"].shape[1]) == c
            eR = int((R_d[mR] >= cut_of_R[mR]).sum())
            print(f"    {names[c]:<14}floor {floor[c]:.3f}  cut "
                  f"{args.min_d_rel*floor[c]:.3f}  eligible "
                  f"{eR:>4}/{int(mR.sum()):<4} ({100*eR/max(int(mR.sum()),1):.0f}%)")
        print(f"             eligible: {nr} of {len(R_d)} rungs, "
              f"{nh} of {len(H_d)} holes")
        print("  the loss makes no claim that near points have low energy;")
        print("  the wall loss alone anchors the low end, as in arm L")
    elif args.min_d > 0:
        nr = int((R_d >= args.min_d).sum())
        nh = int((H_d >= args.min_d).sum())
        print(f"  min_d      {args.min_d} rulers; real objects and near points "
              f"are OUTSIDE the ordering")
        print(f"             eligible: {nr} of {len(R_d)} rungs, "
              f"{nh} of {len(H_d)} holes")
        print("  the loss makes no claim that near points have low energy;")
        print("  the wall loss alone anchors the low end, as in arm L")
    else:
        print("  the real objects are IN the ordering, at d = 0; every pair "
              "whose")
        print("  low side is real restates the wall loss, so the two agree")
    if bl:
        print("")
        print(f"  ARM L was   acc {bl['val_acc']:.1f}%   OOD {bl['ood']:.3f}"
              f"   inverted {bl['inverted']}")
        print("  arm N (path loss, arm A ladders)   OOD 0.845")
        print("  arm N2 (path loss, arm L ladders)  OOD 0.800")
    print("")

    counts = np.array([(y_fit == i).sum() for i in range(C)], dtype=np.float64)
    w = torch.tensor(counts.sum() / (C * np.maximum(counts, 1)),
                     dtype=torch.float32, device=dev)
    if args.balanced:
        # the sampler already equalises the classes; weighting on top would
        # correct the same imbalance twice
        w = torch.ones(C, dtype=torch.float32, device=dev)

    if args.head == "distance":
        from stage4_distance_head import DistanceEnergyNet
        net = DistanceEnergyNet(C).to(dev)
    elif args.head == "linear":
        # the helper's network: PointNet++ with its plain fc3 as the class
        # readout, f_y = W_y . h + b_y, no normalisation and no scale
        from train_energy_classifier import EnergyNet
        net = EnergyNet(C).to(dev)
    elif args.centres > 1:
        from stage49_subcenter import SubCenterEnergyNet
        net = SubCenterEnergyNet(C, scale=args.scale,
                                 centres=args.centres).to(dev)
    else:
        net = CosineEnergyNet(C, scale=args.scale, wide=args.wide).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)

    def readout(f):
        """Input energy from the per-class outputs: -logsumexp for the
        jem head, the best class's energy min_y E(x,y) = -max_y f_y for
        the margin head."""
        mode = args.energy_readout
        if mode == "auto":
            mode = "best" if args.energy_head == "margin" else "free"
        if mode == "best":
            return -f.max(dim=1).values
        return -torch.logsumexp(f, dim=1)

    @torch.no_grad()
    def fwd(X, b=48):
        return torch.cat([net(X[i:i + b].to(dev)).cpu()
                          for i in range(0, len(X), b)])

    def evaluate():
        net.eval()
        fv = fwd(X_val)
        acc = float((fv.argmax(1).numpy() == y_val).mean()) * 100
        e_in = readout(fv).numpy()
        per, pool = {}, []
        for c in unseen:
            eo = readout(fwd(Xtr[np.where(ytr == c)[0][:args.ood_cap]])).numpy()
            per[names[c]] = auroc(e_in, eo)
            pool.append(eo)
        eh = float(readout(fwd(H[:200])).mean())
        # how well the energy actually tracks distance, on the whole rung pool:
        # the fraction of well-separated pairs that come out in the right order
        sub = np.arange(0, len(R), max(1, len(R) // 400))
        er = readout(fwd(R[sub])).numpy()
        dr = R_d[sub]
        gap = dr[None, :] - dr[:, None]
        m = gap > args.delta
        conc = float(((er[None, :] - er[:, None])[m] > 0).mean()) if m.any() else 0.0
        net.train()
        return acc, auroc(e_in, np.concatenate(pool)), per, eh, conc

    def augment(x, gen_):
        """One random up-axis rotation and one scale per sample, applied to
        the whole batch so orientation carries no label information."""
        n = len(x)
        th = torch.rand(n, generator=gen_) * (2 * np.pi)
        c, s_ = torch.cos(th), torch.sin(th)
        R = torch.zeros(n, 3, 3)
        a = args.rot_axis
        u, v = [k for k in (0, 1, 2) if k != a]
        R[:, a, a] = 1.0
        R[:, u, u] = c
        R[:, u, v] = -s_
        R[:, v, u] = s_
        R[:, v, v] = c
        sc = (torch.rand(n, 1, 1, generator=gen_)
              * (args.scale_hi - args.scale_lo) + args.scale_lo)
        return torch.bmm(x, R.transpose(1, 2)) * sc

    g = torch.Generator().manual_seed(args.seed)
    rpool = torch.from_numpy(fit_src)
    # positions within the fit split, grouped by class. X_dec shares the fit
    # ordering and y_dec is y_fit, so one grouping serves both draws.
    by_cls = [np.where(y_fit == k)[0] for k in range(C)]
    if args.balanced and min(len(v) for v in by_cls) == 0:
        raise SystemExit("--balanced needs every class present in the fit split")

    def draw_balanced(n):
        """n positions in the fit split, class picked first, then object."""
        ks = torch.randint(C, (n,), generator=g).numpy()
        return torch.from_numpy(np.array(
            [by_cls[k][torch.randint(len(by_cls[k]), (1,),
                                     generator=g).item()]
             for k in ks], dtype=np.int64))
    t0, log = time.time(), []
    cw = (f"{'comp':>9}" if args.lam_comp > 0 else "") +          (f"{'lip':>9}" if args.lam_lip > 0 else "") +          (f"{'tan':>9}" if args.lam_tan > 0 else "") +          (f"{'in':>9}" if args.lam_in > 0 else "")
    print(f"  {'step':>7}{'cls':>9}{'wall':>9}{'rank':>9}{cw}{'val acc':>10}"
          f"{'OOD':>8}{'inv':>6}{'order':>8}{'min':>7}")
    print("  " + "-" * 73)
    rc = rw = rr = rp = rl = rt = ri_ = 0.0
    anchor_alpha = None
    for st in range(1, args.steps + 1):
        if X_pos is None:
            n_raw = n_real // 2
            n_dec, n_pos = n_real - n_raw, 0
        elif args.pos_from_dec:
            # raw stays at half; the positives come out of the decoded share
            n_raw = n_real // 2
            n_dec = (n_real - n_raw) // 2
            n_pos = n_real - n_raw - n_dec
        else:
            n_raw = n_real // 3
            n_dec = n_real // 3
            n_pos = n_real - n_raw - n_dec
        if args.balanced:
            ri = rpool[draw_balanced(n_raw)]
            di = draw_balanced(n_dec)
        else:
            ri = rpool[torch.randint(len(rpool), (n_raw,), generator=g)]
            di = torch.randint(len(X_dec), (n_dec,), generator=g)
        pi = (torch.randint(len(X_pos), (n_pos,), generator=g)
              if n_pos else torch.zeros(0, dtype=torch.long))
        hi = torch.randint(len(H), (args.hole_slots,), generator=g)
        if args.lip_pairs and args.rung_slots >= 2:
            # half as many ladders, each contributing one adjacent pair
            nl = args.rung_slots // 2
            li = torch.randint(len(R) // n_rungs, (nl,), generator=g)
            ti = torch.randint(n_rungs - 1, (nl,), generator=g)
            gi = torch.stack([li * n_rungs + ti,
                              li * n_rungs + ti + 1], dim=1).reshape(-1)
            if len(gi) < args.rung_slots:
                gi = torch.cat([gi, torch.randint(
                    len(R), (args.rung_slots - len(gi),), generator=g)])
        else:
            gi = torch.randint(len(R), (args.rung_slots,), generator=g)
        pos_parts = [Xtr[ri], X_dec[di]]
        if n_pos:
            pos_parts.append(X_pos[pi])
        xb = torch.cat(pos_parts + [H[hi], R[gi]])
        if args.jitter_real > 0 and n_raw:
            sj = torch.rand(n_raw, 1, 1, generator=g) * args.jitter_real
            xb[:n_raw] = xb[:n_raw] + sj * torch.randn(
                xb[:n_raw].shape, generator=g)
        if args.augment:
            xb = augment(xb, g)
        xb = xb.to(dev)
        # The perturbed twins for the smoothness term ride along in the SAME
        # forward pass. A separate pass is not cheap: profiling gives 223 ms
        # for 40 clouds and 104 ms for 8, because the set-abstraction layers
        # cost the same per call whatever the batch size, so an 8-cloud pass
        # costs half a step on its own.
        n_main = len(xb)
        di_tan = di[torch.from_numpy(tan_ok[di.numpy()])]             if X_tan is not None else None
        if X_tan is not None and len(di_tan):
            xb = torch.cat([xb, X_tan[di_tan].to(dev)])
        if args.lam_lip > 0:
            if args.lip_where == "rungs":
                lo, hi_ = n_real + args.hole_slots, n_main
            elif args.lip_where == "generated":
                lo, hi_ = n_real, n_main
            else:
                lo, hi_ = 0, n_main
            pool_ = torch.arange(lo, hi_)
            k = min(args.lip_n, len(pool_))
            sidx = pool_[torch.randperm(len(pool_), generator=g)[:k]].to(dev)
            xs = xb[sidx]
            xp = xs + (torch.randn(xs.shape, generator=g)
                       * args.lip_eps).to(dev)
            with torch.no_grad():
                d2 = torch.cdist(xs, xp).pow(2)
                dij = ((d2.min(2).values.mean(1)
                        + d2.min(1).values.mean(1)) / ruler)
            xb = torch.cat([xb, xp])
        if args.freeze_head_in_rank and args.head == "linear":
            # the same split as the cosine branch: the live fc3 for
            # classification and the wall, a detached copy for the ranking
            bb = net.b
            l1x, l1p = bb.sa1(xb.transpose(2, 1), None)
            l2x, l2p = bb.sa2(l1x, l1p)
            _, l3p = bb.sa3(l2x, l2p)
            h_ = l3p.view(len(xb), 1024)
            h_ = bb.drop1(F.relu(bb.bn1(bb.fc1(h_))))
            h_ = bb.drop2(F.relu(bb.bn2(bb.fc2(h_))))
            f = bb.fc3(h_)
            e = readout(f)
            e_rank = readout(F.linear(h_, bb.fc3.weight.detach(),
                                      bb.fc3.bias.detach()))
        elif args.freeze_head_in_rank:
            if args.head != "cosine" or args.centres > 1:
                raise SystemExit("--freeze-head-in-rank expects the cosine head")
            # one backbone pass, two readouts off the same features: the live
            # head for classification and the wall, a detached copy of the
            # class vectors for the ranking term
            h_ = F.normalize(net.features(xb), dim=1)
            f = args.scale * (h_ @ F.normalize(net.w, dim=1).T) + net.bias
            e = readout(f)
            e_rank = readout(
                args.scale * (h_ @ F.normalize(net.w.detach(), dim=1).T)
                + net.bias.detach())
        else:
            f = net(xb)
            e = readout(f)
            e_rank = e
        # the twins are read off and removed, so every slice below sees
        # exactly the batch it always did
        n_lip = args.lip_n if args.lam_lip > 0 else 0
        if args.freeze_head_in_tan and X_tan is not None and len(di_tan):
            # recompute the twin energies against a DETACHED head, so the
            # term shapes the backbone and leaves the class vectors alone
            h_t = F.normalize(net.features(xb[n_main + n_lip:]), dim=1)
            e[n_main + n_lip:] = readout(
                args.scale * (h_t @ F.normalize(net.w.detach(), dim=1).T)
                + net.bias.detach())
        e_twin = e[n_main:n_main + n_lip]
        e_tan = e[n_main + n_lip:]
        e, e_rank, f = e[:n_main], e_rank[:n_main], f[:n_main]
        yparts = [y_fit[np.searchsorted(fit_src, ri.numpy())],
                  y_dec[di.numpy()]]
        if n_pos:
            yparts.append(y_pos[pi.numpy()])
        yb = torch.from_numpy(np.concatenate(yparts)).to(dev)
        if args.energy_head == "margin":
            # class energies E = -f; the true class must sit cls_margin
            # below the best other class. No normalisation over classes.
            fr = f[:n_real]
            f_true = fr.gather(1, yb[:, None]).squeeze(1)
            is_true = F.one_hot(yb, fr.shape[1]).bool()
            act = F.softplus if args.cls_smooth else F.relu
            if args.cls_loss == "all":
                # Weston & Watkins: EVERY wrong class must sit cls_margin
                # above the true class, so every class energy is trained on
                # every example, as with cross-entropy, without softmax
                viol = act(args.cls_margin - (f_true[:, None] - fr))
                hinge = viol.masked_fill(is_true, 0.0).sum(1) \
                    / (fr.shape[1] - 1)
            else:
                # Crammer & Singer: only the closest wrong class
                f_other = fr.masked_fill(is_true, float("-inf")).max(1).values
                hinge = act(args.cls_margin - (f_true - f_other))
            l_cls = ((hinge * w[yb]).sum() / w[yb].sum() if w is not None
                     else hinge.mean())
        else:
            l_cls = F.cross_entropy(f[:n_real], yb, weight=w)
        e_real = e[:n_real]
        mu_r = e_real.mean().detach()
        sd_r = e_real.std().detach().clamp_min(1e-6)
        e_hole = e[n_real:n_real + args.hole_slots]
        if args.hole_slots == 0:
            # no negatives in the batch (the no-samples control): no wall
            l_wall = torch.zeros((), device=dev)
        elif args.per_class_wall:
            # each negative against its own class's real objects in this batch;
            # classes absent from the batch fall back to the pooled mean
            yb_np = np.concatenate(yparts)
            ref = mu_r.expand(args.hole_slots).clone()
            yh = y_hole[hi.numpy()]
            for k in range(C):
                m_r = yb_np == k
                m_h = yh == k
                if m_r.any() and m_h.any():
                    ref[torch.from_numpy(m_h).to(dev)] =                         e_real[torch.from_numpy(m_r).to(dev)].mean().detach()
            l_wall = F.relu(args.margin * sd_r - (e_hole - ref)).mean() / sd_r
        else:
            l_wall = F.relu(args.margin * sd_r - (e_hole - mu_r)).mean() / sd_r

        # every point in the batch carries a measured distance from the data:
        # raw real objects are the reference at 0
        dist = torch.from_numpy(np.concatenate([
            np.zeros(n_raw, np.float32),
            np.full(n_dec + n_pos, dec_d, np.float32),
            H_d[hi.numpy()], R_d[gi.numpy()]])).to(dev)
        gap = dist[None, :] - dist[:, None]
        mask = gap > args.delta
        if args.same_class:
            cls_of = torch.cat([yb,
                                torch.from_numpy(
                                    np.concatenate([
                                        y_hole[hi.numpy()],
                                        y_rung[gi.numpy()]])).to(dev)])
            mask = mask & (cls_of[:, None] == cls_of[None, :])
        if args.min_d_rel > 0:
            cut = torch.from_numpy(np.concatenate([
                np.zeros(n_raw + n_dec + n_pos, np.float32),
                cut_of_H[hi.numpy()], cut_of_R[gi.numpy()]])).to(dev)
            # a real object has no cut of its own; keep it out of the ordering
            far = (dist >= cut) & (dist > 0)
            mask = mask & far[:, None] & far[None, :]
        elif args.min_d > 0:
            far = dist >= args.min_d
            mask = mask & far[:, None] & far[None, :]
        if bool(mask.any()):
            # NOT detached, deliberately. Detaching the nearer point was
            # tried (stage60) on the reasoning that half the gradient
            # otherwise pushes a negative DOWN against the wall loss. It made
            # things worse: OOD 0.918 -> 0.839, inverted 0 -> 4, and the rank
            # term stopped converging (0.035 -> 0.063, rising), exactly as the
            # path loss did in arm N. With the lower point frozen the term can
            # only push upward and so ratchets without limit. The two-sided
            # form is self-limiting, which is why it converges.
            rise = (e_rank[None, :].expand_as(gap)
                    - e_rank[:, None].expand_as(gap))
            if args.prop_margin or args.smooth:
                # required rise grows with the distance gap, calibrated so
                # that a pair spanning the median hole distance asks for the
                # wall's own margin -- nothing new is chosen
                need = args.margin * sd_r * (gap / d_ref)
            else:
                need = torch.zeros((), device=dev)
            short = F.relu(need - rise)            # rose too little
            if args.smooth:
                # and penalise rising too much, symmetrically: the gap should
                # track the distance, not merely exceed it
                tol = (args.margin if args.smooth_tol < 0
                       else args.smooth_tol) * sd_r
                over = F.relu(rise - need - tol)
                l_rank = (short + over)[mask].mean() / sd_r
            else:
                l_rank = short[mask].mean() / sd_r
        else:
            l_rank = torch.zeros((), device=dev)

        if args.lam_lip > 0:
            # A local pair is MADE, not looked for. Pairing inside the batch
            # was tried first and fails: two clouds drawn independently are
            # 0.8-3.1 rulers apart and even two adjacent rungs of one ladder
            # are 0.8, while the measurement's window is 0.25, so 33 batches
            # in 40 contained no local pair at all and the term read 0.0000.
            # A cloud against a slightly perturbed copy of ITSELF is local by
            # construction, and it is safe in the way that matters: a pair is
            # one point and its own neighbour, so the large data-to-negative
            # gap that detection depends on is never inside a pair and cannot
            # be capped.
            # relative excess, so the term is O(1) like the others and
            # --lam-lip means the same thing as --lam-wall and --lam-rank
            slope = ((e_twin - e[sidx]).abs()
                     / dij.clamp_min(1e-6)) / sd_r
            l_lip = F.relu(slope / args.lip_tol - 1.0).mean()
        else:
            l_lip = torch.zeros((), device=dev)

        if args.lam_comp > 0:
            # the class each negative was searched FROM is the class it is
            # not; push that one probability down, per logit
            yh_t = torch.from_numpy(y_hole[hi.numpy()]).long().to(dev)
            fh = f[n_real:n_real + args.hole_slots]
            if args.energy_head == "margin":
                # the negative's energy UNDER ITS SOURCE CLASS, E(x, y_src)
                # = -f_src, must sit above the true-class energy of the real
                # objects, by the wall's own margin in the same sd units.
                # Only the source class is moved, and only upward: no other
                # class is asked to claim the negative, which would lower
                # its energy and undo the wall.
                f_src = fh.gather(1, yh_t[:, None]).squeeze(1)
                ec_real = -f[:n_real].gather(1, yb[:, None]).squeeze(1)
                mu_c = ec_real.mean().detach()
                sd_c = ec_real.std().detach().clamp_min(1e-6)
                l_comp = F.relu(args.margin * sd_c
                                - (-f_src - mu_c)).mean() / sd_c
            else:
                p_src = F.softmax(fh, dim=1).gather(
                    1, yh_t[:, None]).squeeze(1)
                l_comp = -torch.log((1.0 - p_src).clamp_min(1e-6)).mean()
        else:
            l_comp = torch.zeros((), device=dev)

        if X_tan is not None and len(di_tan):
            # the decoded objects occupy [n_raw, n_raw + n_dec); the twins at
            # the end of the batch are those of the KEPT objects, in order
            keep_d = torch.from_numpy(tan_ok[di.numpy()]).to(dev)
            base = e[n_raw:n_raw + n_dec][keep_d]
            l_tan = (((e_tan - base) / sd_r) ** 2).mean()
        else:
            l_tan = torch.zeros((), device=dev)

        if args.lam_in > 0:
            # push the upper tail of the REAL energies down. One point at a
            # time: no pair, no gradient, no relation between points.
            thr = mu_r + args.m_in * sd_r
            l_in = (F.relu(e_real - thr) ** 2).mean() / (sd_r ** 2)
        else:
            l_in = torch.zeros((), device=dev)

        loss = (l_cls + args.lam_wall * l_wall + args.lam_rank * l_rank
                + args.lam_comp * l_comp + args.lam_lip * l_lip
                + args.lam_tan * l_tan + args.lam_in * l_in)
        if args.anchor:
            l_anchor = e_real.square().mean()
            if anchor_alpha is None:
                anchor_alpha = float(l_cls.detach()
                                     / l_anchor.detach().clamp_min(1e-8))
                print(f"  anchor alpha {anchor_alpha:.4g}  (E(real) at start "
                      f"{float(e_real.mean()):+.2f})", flush=True)
            loss = loss + anchor_alpha * l_anchor
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        rc += float(l_cls.detach())
        rw += float(l_wall.detach())
        rr += float(l_rank.detach())
        rp += float(l_comp.detach())
        rl += float(l_lip.detach())
        rt += float(l_tan.detach())
        ri_ += float(l_in.detach())
        if st % args.eval_every == 0 or st == args.steps:
            acc, ood, per, eh, conc = evaluate()
            inv = sum(1 for v in per.values() if v < 0.5)
            mins = (time.time() - t0) / 60
            log.append({"step": st, "cls": rc / args.eval_every,
                        "wall": rw / args.eval_every,
                        "rank": rr / args.eval_every,
                        "comp": rp / args.eval_every,
                        "lip": rl / args.eval_every,
                        "tan": rt / args.eval_every,
                        "in": ri_ / args.eval_every, "val_acc": acc,
                        "ood": ood, "inverted": inv, "e_hole": eh,
                        "order": conc, "per_category": per, "minutes": mins})
            cv = ((f"{rp/args.eval_every:>9.4f}"
                   if args.lam_comp > 0 else "")
                  + (f"{rl/args.eval_every:>9.4f}"
                     if args.lam_lip > 0 else "")
                  + (f"{rt/args.eval_every:>9.4f}"
                     if args.lam_tan > 0 else "")
                  + (f"{ri_/args.eval_every:>9.4f}"
                     if args.lam_in > 0 else ""))
            print(f"  {st:>7}{rc/args.eval_every:>9.4f}"
                  f"{rw/args.eval_every:>9.4f}{rr/args.eval_every:>9.4f}{cv}"
                  f"{acc:>9.1f}%{ood:>8.3f}{inv:>6}{conc:>8.3f}{mins:>7.1f}",
                  flush=True)
            rc = rw = rr = rp = rl = rt = ri_ = 0.0
            torch.save({"state_dict": net.state_dict(), "classes": classes,
                        "class_names": cname, "val_sources": val_src.tolist(),
                        "fit_sources": fit_src.tolist(),
                        "head": (args.head if args.head != "cosine"
                                 else ("subcenter" if args.centres > 1
                                       else "cosine")),
                        "centres": args.centres, "scale": args.scale,
                        "wide": bool(args.wide),
                        "holes": args.holes.name, "ladders": args.ladders.name,
                        "config": {k: str(v) for k, v in vars(args).items()},
                        "log": log}, args.out)

    r = log[-1]
    print("")
    print(f"  saved {args.out.name}")
    if bl:
        print("")
        print("  " + "=" * 62)
        print(f"  {'':<16}{'ARM L':>12}{'ARM P':>12}{'change':>12}")
        print("  " + "-" * 62)
        print(f"  {'val accuracy':<16}{bl['val_acc']:>11.1f}%"
              f"{r['val_acc']:>11.1f}%{r['val_acc']-bl['val_acc']:>+12.1f}")
        print(f"  {'OOD AUROC':<16}{bl['ood']:>12.3f}{r['ood']:>12.3f}"
              f"{r['ood']-bl['ood']:>+12.3f}")
        print(f"  {'inverted':<16}{bl['inverted']:>12}{r['inverted']:>12}"
              f"{r['inverted']-bl['inverted']:>+12}")
        print("  " + "=" * 62)
    print("")
    print(f"  pairs ordered correctly, whole rung pool   {r['order']:.3f}")
    print("  (1.000 would mean the energy is a strictly increasing function of")
    print("   measured distance from the data over every well-separated pair)")
    print("")
    print("  WHAT SEPARATES THIS FROM ARM N AND ARM N2")
    print("    both of those had a term that GREW every eval and never")
    print("    converged. If the rank column falls instead, the objective is")
    print("    satisfiable and the previous failures were the path unit, not")
    print("    the property. If it grows too, the property itself conflicts")
    print("    with detection in this model, measured two independent ways.")
    print("")
    print("  Validation numbers. The TEST split is still unopened.")


if __name__ == "__main__":
    main()
