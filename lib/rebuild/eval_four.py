"""The four metrics that decide whether this decoder serves the thesis.

    Chamfer distance        overall geometric reconstruction
    Earth Mover distance    correct point distribution; catches aggregation
    feature-cycle error     does the reconstructed cloud still carry the
                            PRODUCTION feature the EBM actually samples in
    adversarial validity    does a latent attack survive feature -> cloud

WHY THESE FOUR AND NOT THE EIGHT-GATE SUITE
-------------------------------------------
The gate suite answers "is this a good autoencoder". These four answer "can
the pipeline this thesis needs be built on it", which is a different and
narrower question. Two of them are new.

FEATURE-CYCLE IS NOT THE LATENT CYCLE
-------------------------------------
The gate suite's cycle error uses the autoencoder's OWN encoder:

    |E_ae(D(E_ae(X))) - E_ae(X)|

That measures self-consistency and nothing more -- an encoder and decoder can
agree perfectly with each other while both drift away from the feature space
the EBM lives in. The metric here uses the FROZEN PRODUCTION backbone and the
fitted 1024 -> 195 transform instead:

    |T(f_prod(D(E_ae(X)))) - T(f_prod(X))|

This measures preservation of production features during reconstruction.
The AE latent and production PCA coordinates are different spaces, even
when both have dimension 195. It does not validate decoding production
EBM samples directly; that requires an explicitly compatible decoder.

ADVERSARIAL SUCCESS IS MEASURED AGAINST THE DECODED BASELINE
------------------------------------------------------------
ASR compares the attacked reconstruction against the UNATTACKED
reconstruction, never against the original cloud:

    wrong                                     right
    ------------------------------------      -----------------------------
    classify(X)  vs  classify(D(z + dz))      classify(D(z))  vs
                                                    classify(D(z + dz))

Report prediction flips over all reconstructions separately from ASR on
the subset correctly classified before attack. Neither is a guaranteed
lower bound on attack success in another pipeline. A radius screen alone
does not establish plausible geometry or preserved semantic identity.

The attack is projected gradient ascent in the latent, scaled per dimension
by the AE fit-split `sd`. This is an L-infinity budget in standardized AE
coordinates, not the norm or distribution of a Gaussian EBM noise step.
"""

import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parent
PARENT = ROOT.parent
REPO = PARENT / "Pointnet_Pointnet2_pytorch"
for _p in (str(ROOT), str(PARENT), str(REPO), str(REPO / "models")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from decoder import (chamfer_pairs, chamfer_pairs_unsquared,      # noqa: E402
                     chamfer_pairs_unsquared_max)

_BACKBONE = None
_TRANSFORM = None
_SAME_CLASS_EMD = {}     # model-independent, so cache across gate checks


def backbone(dev):
    """The frozen production classifier. Loaded once per process."""
    global _BACKBONE
    if _BACKBONE is None:
        import pointnet2_features as P
        _BACKBONE = P.PointNet2Backbone(device=dev)
    return _BACKBONE


def transform():
    """The fitted 1024 -> k map the EBM's space is defined by."""
    global _TRANSFORM
    if _TRANSFORM is None:
        from transform import Transform
        _TRANSFORM = Transform.load()
    return _TRANSFORM


def prod_feature(X, dev, chunk=64):
    """(B, N, 3) clouds -> (B, 1024) production features. No grad."""
    import pointnet2_features as P
    bb = backbone(dev)
    with torch.no_grad():
        return torch.cat([bb(P.to_model_input(X[i:i + chunk]))
                          for i in range(0, len(X), chunk)])


def emd_pairs(A, B, chunk=16):
    """Exact EMD for aligned pairs: mean per-point distance under the
    optimal one-to-one assignment."""
    out = np.empty(len(A))
    for s in range(0, len(A), chunk):
        d = torch.cdist(A[s:s + chunk], B[s:s + chunk]).cpu().numpy()
        for j in range(len(d)):
            r, c = linear_sum_assignment(d[j])
            out[s + j] = d[j][r, c].mean()
    return float(out.mean())


def pgd_latent(model, z, y, sd, eps, steps=10, chunk=64):
    """Projected gradient ascent on the latent against the frozen classifier.

    Maximises cross-entropy of classify(decode(z + dz, y)) away from the
    label the UNATTACKED reconstruction receives. dz is bounded per
    dimension by eps * sd, an L-inf ball in sd units.

    Returns (attacked clouds, baseline predictions, attacked predictions).
    """
    import pointnet2_features as P
    bb = backbone(z.device)
    lim = (eps * sd)[None, :]

    base_pred, adv_cloud = [], []
    for s in range(0, len(z), chunk):
        zb, yb = z[s:s + chunk], y[s:s + chunk]
        with torch.no_grad():
            b_cloud = model.decode(zb, yb)
            b_pred = bb.classify(P.to_model_input(b_cloud)).argmax(1)
        base_pred.append(b_pred)

        dz = torch.zeros_like(zb, requires_grad=True)
        for _ in range(steps):
            cloud = model.decode(zb + dz, yb)
            logit = bb.classify(P.to_model_input(cloud))
            loss = F.nll_loss(logit, b_pred)      # classify returns log-probs
            g, = torch.autograd.grad(loss, [dz])
            with torch.no_grad():
                dz += (lim / 4.0) * g.sign()
                dz.clamp_(-lim.expand_as(dz), lim.expand_as(dz))
            dz.requires_grad_(True)
        with torch.no_grad():
            adv_cloud.append(model.decode(zb + dz.detach(), yb))
    return (torch.cat(adv_cloud), torch.cat(base_pred))


def report(model, X, y, Xf, dev, epoch=None, emd_n=256, eps=(0.1, 0.25),
           pgd_steps=10, tag="", sd=None):
    """Print the four metrics. X, y: held-out clouds and labels (on `dev`).

    Xf is the fit split: it supplies the retrieval reference AND the
    per-dimension latent sd that scales the attack. That sd is the FIT
    split's, never the eval subset's, so the attack budget matches the
    scaling the EBM's own sampler uses.
    """
    import pointnet2_features as P
    from autoencoder import canonical_order

    t0 = time.time()
    head = f"epoch {epoch}" if epoch is not None else "final"
    print("=" * 76)
    print(f"FOUR-METRIC EVALUATION  --  {head}   {len(X)} held-out clouds")
    print("=" * 76)

    Xc = canonical_order(X)
    with torch.no_grad():
        Zf = torch.cat([model.encode(Xf[i:i + 64])
                        for i in range(0, len(Xf), 64)])
        z = torch.cat([model.encode(X[i:i + 64]) for i in range(0, len(X), 64)])
        rec = torch.cat([model.decode(z[i:i + 64], y[i:i + 64])
                         for i in range(0, len(X), 64)])
    if sd is None:
        sd = Zf.std(0)

    # ---------------------------------------------------------------- 1. CD
    cd_u = float(chamfer_pairs_unsquared(rec, Xc).mean())
    cd_s = float(chamfer_pairs(rec, Xc).mean())
    cd_m = float(chamfer_pairs_unsquared_max(rec, Xc).mean())
    print("\n  1. CHAMFER DISTANCE            (geometric reconstruction)")
    print(f"     unsquared sum  {cd_u:.5f}   <- the training objective")
    print(f"     squared sum    {cd_s:.5f}")
    print(f"     unsquared max  {cd_m:.5f}   <- FoldingNet's form")

    # --------------------------------------------------------------- 2. EMD
    m = min(emd_n, len(X))
    Xfc = canonical_order(Xf)
    nn64 = torch.cdist(z[:m], Zf).topk(64, largest=False).indices
    retr = torch.empty_like(Xc[:m])
    with torch.no_grad():
        for i in range(m):
            cand = Xfc[nn64[i]]
            d = chamfer_pairs_unsquared(cand, Xc[i].expand_as(cand))
            retr[i] = cand[int(d.argmin())]

    emd_rec = emd_pairs(rec[:m], Xc[:m])
    emd_ret = emd_pairs(retr, Xc[:m])
    key = (tag, m)
    if key not in _SAME_CLASS_EMD:
        # Real vs another real of the same class. Independent of the model,
        # so it is computed once and reused by every later gate check.
        yo = y[:m].cpu().numpy()
        partner = np.full(m, -1, dtype=np.int64)
        for c in np.unique(yo):
            idx = np.where(yo == c)[0]
            if len(idx) > 1:
                partner[idx] = idx[np.roll(np.arange(len(idx)), 1)]
        have = partner >= 0
        _SAME_CLASS_EMD[key] = (
            emd_pairs(Xc[:m][torch.from_numpy(partner[have]).to(dev)],
                      Xc[:m][torch.from_numpy(np.where(have)[0]).to(dev)])
            if have.any() else float("nan"))
    emd_cls = _SAME_CLASS_EMD[key]

    print(f"\n  2. EARTH MOVER DISTANCE        (point distribution, n = {m})")
    print(f"     reconstruction        {emd_rec:.5f}")
    print(f"     retrieval, best of 64 {emd_ret:.5f}")
    print(f"     other real same class {emd_cls:.5f}")
    print(f"     EMD / retrieval       {emd_rec / emd_ret:.3f}"
          "   <= 1 means decoding beats lookup")
    print(f"     CD  / retrieval       "
          f"{float(chamfer_pairs_unsquared(rec[:m], Xc[:m]).mean()) / float(chamfer_pairs_unsquared(retr, Xc[:m]).mean()):.3f}"
          "   disagreement is diagnostic, not proof of aggregation")

    # ------------------------------------------------------ 3. feature cycle
    t = transform()
    f_real = prod_feature(X, dev).cpu().numpy().astype(np.float64)
    f_rec = prod_feature(rec, dev).cpu().numpy().astype(np.float64)
    T_real = t.encode(f_real)
    T_rec = t.encode(f_rec)
    err = np.linalg.norm(T_rec - T_real, axis=1)
    rel = err / np.maximum(np.linalg.norm(T_real, axis=1), 1e-9)
    inter = float(np.linalg.norm(
        T_real[:, None, :][:64] - T_real[None, :, :][:, :64], axis=2).mean())
    print("\n  3. FEATURE-CYCLE ERROR         (production feature preserved)")
    print(f"     |T(f(rec)) - T(f(real))| / |T(f(real))|   {rel.mean():.4f}")
    print(f"     same, / mean inter-object distance        "
          f"{err.mean() / inter:.4f}")
    print(f"     (production backbone + fitted {T_real.shape[1]}-D transform,"
          " NOT the AE's own encoder)")

    # ------------------------------------------------- 4. adversarial / ASR
    print("\n  4. LATENT ATTACK  (radius screening does not establish semantic validity)")
    print(f"     {'eps':<8}{'flip %':>9}{'ASR correct':>12}"
          f"{'max radius':>13}{'>1.05 %':>10}")
    bb = backbone(dev)
    attacks = {}
    for e in eps:
        adv, base_pred = pgd_latent(model, z, y, sd, e, steps=pgd_steps)
        with torch.no_grad():
            adv_pred = torch.cat(
                [bb.classify(P.to_model_input(adv[i:i + 64])).argmax(1)
                 for i in range(0, len(adv), 64)])
            rad = adv.norm(dim=2).max(1).values
        flipped = adv_pred != base_pred
        ok = rad <= 1.10
        asr = float(flipped.float().mean()) * 100
        eligible = base_pred == y
        count = int(eligible.sum())
        successful = eligible & (adv_pred != y)
        vasr = 100.0 * int(successful.sum()) / count if count else None
        radius_asr = 100.0 * int((successful & ok).sum()) / count if count else None
        attacks[str(e)] = {
            "eps_sd_linf": float(e), "steps": pgd_steps,
            "n": len(y), "n_correct_baseline": count,
            "prediction_flip_pct": asr, "asr_correct_pct": vasr,
            "radius_screened_asr_correct_pct": radius_asr,
            "radius_threshold": 1.10, "mean_max_radius": float(rad.mean()),
            "clouds_over_105_pct": float((rad > 1.05).float().mean()) * 100,
        }
        asr_text = f"{vasr:.1f}%" if vasr is not None else "N/A"
        print(f"     {e:<8.2f}{asr:>8.1f}%{asr_text:>12}"
              f"{float(rad.mean()):>13.3f}"
              f"{float((rad > 1.05).float().mean()) * 100:>9.2f}")
    print("     ASR uses only correctly classified unperturbed reconstructions.")
    print("     This attacks the AE latent, not the production PCA/EBM space.")

    print(f"\n  [{time.time() - t0:.0f}s]")
    print("=" * 76, flush=True)

    return {"cd_unsquared": cd_u, "cd_squared": cd_s, "cd_unsq_max": cd_m,
            "emd": emd_rec, "emd_retrieval": emd_ret,
            "emd_same_class": emd_cls, "emd_n": m,
            "feature_cycle_rel": float(rel.mean()),
            "feature_cycle_ratio": float(err.mean() / inter),
            "attacks": attacks}
