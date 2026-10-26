"""Energy-based open-set classifier on ModelNet10, full training data.

One script, every step in order:

  1. helper      PointNet++ energy classifier (no softmax) trained on REAL
                 objects only: all ModelNet10 fit objects of the fixed split
                 (official train minus a stratified 20 % validation, seed 0)
  2. settings    derived by their rules before anything is generated:
                   lam   the helper's and the generator's pulls are balanced
                   cut   90th percentile of real-to-nearest-real distance
                   r_z   smallest search radius whose median negative lies
                         beyond the cut
                 all other settings are the audited values (see SETTINGS)
  3. samples     search in the pretrained autoencoder's latent space, guided by
                 the helper and kept realistic by the pretrained generator;
                 decoded to clouds: negatives (beyond the cut), hard positives,
                 ladders
  4. final       classifier trained with the samples (3 seeds)
  5. control     the same classifier without the samples (3 seeds)
  6. compare     validation accuracy, OOD AUROC and FPR@95 against the 30 other
                 ModelNet40 categories; rule fixed in advance: the samples help
                 if AUROC is higher AND FPR@95 lower than the control, both
                 beyond the seed spread. The official test split is not used.

    python run_pipeline.py                 full run  -> runs/full/
    python run_pipeline.py --smoke         a few minutes, checks everything runs

Requirements: Python 3.10+, a CUDA GPU, torch, numpy, scipy, scikit-learn
    (pip install torch numpy scipy scikit-learn). Data and checkpoints are
    stored with Git LFS: run `git lfs pull` after cloning.

Inputs (nothing else is read):
    pretrained/autoencoder.pth   point-cloud autoencoder (1,024 points)
    pretrained/generator.pth     latent energy-based generator
    data/clouds_train.npz        ModelNet40 train clouds, 1,024 points, labels
    data/latent_bank.npz         the autoencoder's latents of those clouds
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
LIB = HERE / "lib" / "rebuild"
CLASSES = [1, 2, 8, 12, 14, 22, 23, 30, 33, 35]   # ModelNet10 inside ModelNet40

# Audited settings (parameter audit, 1,024 points); per-object budgets are the
# 60-per-class values divided by its 600 fit objects.
SETTINGS = dict(r_e=1.0, steps=60, eta=0.1, restarts=18, neg_per_object=5.0,
                hp_rz=0.05, hp_re=0.1, hp_restarts=6, hp_per_object=3.85,
                rungs=8, min_d_rel=1.4, delta=0.1, margin=3.0, cls_margin=9.7,
                lam_comp=1.0, jitter=0.02, helper_steps=2000, train_steps=6000)


def run(cmd, log, env):
    """Run one step of the pipeline from lib/rebuild, output to a log file."""
    t0 = time.time()
    with open(log, "w") as fh:
        p = subprocess.run([sys.executable, "-u"] + [str(c) for c in cmd], cwd=LIB, env=env,
                           stdout=fh, stderr=subprocess.STDOUT)
    if p.returncode:
        print(open(log).read()[-3000:])
        sys.exit(f"step failed: {' '.join(str(c) for c in cmd[:1])} (log {log})")
    return time.time() - t0


def say(msg):
    print(f"[{time.strftime('%H:%M')}] {msg}", flush=True)


def cut_from_real(helper, env):
    """90th percentile of each validation object's distance to its nearest real
    object of the reference set, in class rulers (as in the stage180 pipeline)."""
    code = f"""
import sys, numpy as np, torch
sys.path.insert(0, "."); sys.path.insert(0, "..")
import pointnet2_features as P
P.patch_deterministic_fps(True)
from decoder import load_clouds
from stage11_can_we_cross import nearest_cd, ruler_of
dev = "cuda"
Xtr, ytr = load_clouds("train"); Xtr = torch.as_tensor(np.asarray(Xtr)).float(); ytr = np.asarray(ytr)
ck = torch.load(r"{helper}", map_location="cpu", weights_only=False)
cls = ck["classes"]
ref_idx = np.concatenate([np.where(ytr == c)[0][:40] for c in cls])
ruler = float(np.mean([ruler_of(Xtr[np.where(ytr == c)[0][:40]], dev, 256) for c in cls]))
val = np.array([i for i in ck["val_sources"] if i not in set(ref_idx.tolist())])
val = np.random.default_rng(0).choice(val, min(300, len(val)), replace=False)
d = np.asarray(nearest_cd(Xtr[val], Xtr[ref_idx], dev, 256)) / ruler
print(f"CUT {{np.quantile(d, 0.9):.3f}}")
"""
    out = subprocess.run([sys.executable, "-c", code], cwd=LIB, env=env, capture_output=True, text=True)
    m = re.search(r"CUT ([0-9.]+)", out.stdout)
    if not m:
        sys.exit(out.stdout[-2000:] + out.stderr[-2000:])
    return float(m.group(1))


def keep_beyond_cut(src, dst, cut):
    d = np.load(src, allow_pickle=True)
    m = d["d_real"] > cut
    o = {k: (d[k][m] if k != "provenance_json" else d[k]) for k in d.files}
    p = json.loads(str(d["provenance_json"]))
    p["filtered"] = f"d_real > {cut}"
    o["provenance_json"] = np.array(json.dumps(p))
    np.savez(dst, **o)
    return int(m.sum()), len(m)


def evaluate(groups, env):
    """Validation accuracy, OOD AUROC and FPR@95 (unknowns: up to 120 objects of
    each of the 30 other ModelNet40 categories, from the train split)."""
    code = f"""
import sys, json, numpy as np, torch
sys.path.insert(0, "."); sys.path.insert(0, "..")
import pointnet2_features as P
P.patch_deterministic_fps(True)
from decoder import load_clouds
from train_energy_classifier import EnergyNet
from stage0_real_only import auroc
dev = "cuda"
Xtr, ytr = load_clouds("train"); Xtr = torch.as_tensor(np.asarray(Xtr)).float(); ytr = np.asarray(ytr)
groups = json.loads(r'''{json.dumps(groups)}''')
res = {{}}
for g, files in groups.items():
    rows = []
    for f in files:
        ck = torch.load(f, map_location="cpu", weights_only=False)
        net = EnergyNet(len(ck["classes"])).to(dev); net.load_state_dict(ck["state_dict"]); net.eval()
        def logits(X):
            with torch.no_grad():
                return torch.cat([net(X[i:i + 48].to(dev)).cpu() for i in range(0, len(X), 48)])
        val = np.array(ck["val_sources"]); cmap = {{c: i for i, c in enumerate(ck["classes"])}}
        lv = logits(Xtr[val])
        acc = 100 * float((lv.argmax(1).numpy() == np.array([cmap[int(c)] for c in ytr[val]])).mean())
        si = (-torch.logsumexp(lv, 1)).numpy(); thr = np.quantile(si, 0.95)
        un = [c for c in sorted(set(int(v) for v in ytr)) if c not in ck["classes"] and (ytr == c).sum() >= 40]
        so = (-torch.logsumexp(logits(Xtr[np.concatenate([np.where(ytr == c)[0][:120] for c in un])]), 1)).numpy()
        rows.append([acc, float(auroc(si, so)), 100 * float((so <= thr).mean())])
    res[g] = rows
print("RESULT " + json.dumps(res))
"""
    out = subprocess.run([sys.executable, "-c", code], cwd=LIB, env=env, capture_output=True, text=True)
    m = re.search(r"RESULT (.*)", out.stdout)
    if not m:
        sys.exit(out.stdout[-2000:] + out.stderr[-2000:])
    return {g: np.array(v) for g, v in json.loads(m.group(1)).items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=HERE / "runs" / "full")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--per-class", type=int, default=0, help="cap on fit objects per class (0 = all)")
    ap.add_argument("--smoke", action="store_true", help="tiny settings, only checks that every step runs")
    args = ap.parse_args()

    S = dict(SETTINGS)
    rz_grid = [1.5, 2.0, 2.5, 3.0, 3.5]
    if args.smoke:
        S.update(steps=3, restarts=1, hp_restarts=1, helper_steps=30, train_steps=30)
        rz_grid = [1.5, 2.5]
        args.seeds = args.seeds[:1]
        if args.out == HERE / "runs" / "full":
            args.out = HERE / "runs" / "smoke"
    O = args.out.resolve()
    O.mkdir(parents=True, exist_ok=True)
    AE, GEN = HERE / "pretrained" / "autoencoder.pth", HERE / "pretrained" / "generator.pth"
    for f in (AE, GEN, HERE / "data" / "clouds_train.npz", HERE / "data" / "latent_bank.npz"):
        if not f.exists():
            sys.exit(f"missing {f}")
    env = dict(os.environ, CLOUDS_NPZ=str(HERE / "data" / "clouds_train.npz"),
               LATENT_BANK=str(HERE / "data" / "latent_bank.npz"))
    env.pop("SHUFFLE_POINTS", None)
    record = {"settings": S, "seeds": args.seeds, "per_class": args.per_class}

    # 1. helper -------------------------------------------------------------
    helper = O / "helper.pth"
    say("1. helper on real objects only")
    run(["stage0_real_only.py", "--classes", *CLASSES, "--steps", S["helper_steps"], "--per-class", args.per_class,
         "--seed", 0, "--eval-every", 500, "--cls-loss", "margin", "--cls-margin", S["cls_margin"],
         "--out", helper, "--overwrite"], O / "1_helper.log", env)
    import torch
    n_fit = len(torch.load(helper, map_location="cpu", weights_only=False)["fit_sources"])
    say(f"   {n_fit} fit objects")

    # 2. settings by their rules ----------------------------------------------
    say("2. settings: lam, cut, r_z")
    run(["stage147_lam_balance.py", "--ckpt", helper, "--ae", AE, "--ebm", GEN], O / "2_lam.log", env)
    m = re.search(r"whole chain\s+\S+\s+\S+\s+(\S+)", open(O / "2_lam.log").read())
    if not m:
        sys.exit("lam balance gave no value (2_lam.log)")
    lam = float(m.group(1))
    cut = cut_from_real(helper, env)
    meds = {}
    for rz in rz_grid:
        run(["stage23_joint_holes.py", "--ckpt", helper, "--ae", AE, "--ebm", GEN, "--rz", rz, "--re", S["r_e"],
             "--steps", S["steps"], "--restarts", 1, "--langevin", "--eta", S["eta"], "--lam", lam,
             "--out", O / f"rz_check_{rz}.npz"], O / f"2_rz_{rz}.log", env)
        meds[rz] = float(np.median(np.load(O / f"rz_check_{rz}.npz", allow_pickle=True)["d_real"]))
    rz = next((r for r in rz_grid if meds[r] > cut), rz_grid[-1])
    say(f"   lam {lam:.3f}   cut {cut:.3f}   r_z {rz}   (medians {meds})")
    record.update(lam=lam, cut=cut, r_z=rz, rz_medians=meds, n_fit=n_fit)

    # 3. samples ---------------------------------------------------------------
    neg_cap, hp_cap = int(round(S["neg_per_object"] * n_fit)), int(round(S["hp_per_object"] * n_fit))
    say(f"3. samples: negatives (cap {neg_cap}), hard positives (cap {hp_cap}), ladders")
    run(["stage23_joint_holes.py", "--ckpt", helper, "--ae", AE, "--ebm", GEN, "--rz", rz, "--re", S["r_e"],
         "--steps", S["steps"], "--restarts", S["restarts"], "--langevin", "--eta", S["eta"], "--lam", lam,
         "--max-keep", neg_cap, "--save-clouds", "--out", O / "negatives_all.npz"], O / "3_negatives.log", env)
    run(["stage23_joint_holes.py", "--ckpt", helper, "--ae", AE, "--ebm", GEN, "--rz", S["hp_rz"], "--re", S["hp_re"],
         "--steps", S["steps"], "--restarts", S["hp_restarts"], "--langevin", "--eta", S["eta"], "--lam", lam,
         "--hard-positives", "--max-keep", hp_cap, "--save-clouds", "--out", O / "hard_positives.npz"],
        O / "3_hard_positives.log", env)
    run(["stage42_ladder_probe.py", "--ckpt", helper, "--ae", AE, "--ebm", GEN, "--rz", rz, "--rungs", S["rungs"],
         "--langevin", "--lam", lam, "--out", O / "ladders.npz"], O / "3_ladders.log", env)
    kept, total = keep_beyond_cut(O / "negatives_all.npz", O / "negatives.npz", cut)
    say(f"   negatives kept beyond the cut: {kept} of {total}")
    record.update(negatives_kept=kept, negatives_found=total)

    # 4./5. final and control ---------------------------------------------------
    common = ["--arm-a", helper, "--head", "linear", "--steps", S["train_steps"], "--eval-every", 1000,
              "--energy-head", "margin", "--cls-margin", S["cls_margin"], "--cls-loss", "all",
              "--energy-readout", "free", "--jitter-real", S["jitter"], "--ae", AE, "--overwrite"]
    for s in args.seeds:
        say(f"4. final classifier (with samples), seed {s}")
        run(["stage45_rank.py", "--holes", O / "negatives.npz", "--ladders", O / "ladders.npz",
             "--positives", O / "hard_positives.npz", "--lam-comp", S["lam_comp"], "--min-d-rel", S["min_d_rel"],
             "--delta", S["delta"], "--margin", S["margin"], "--prop-margin", "--freeze-head-in-rank",
             "--pos-from-dec", "--seed", s, "--out", O / f"final_s{s}.pth", *common], O / f"4_final_s{s}.log", env)
    for s in args.seeds:
        say(f"5. control (no samples), seed {s}")
        run(["stage45_rank.py", "--holes", O / "negatives.npz", "--ladders", O / "ladders.npz", "--batch", 24,
             "--hole-slots", 0, "--rung-slots", 0, "--lam-wall", 0, "--lam-rank", 0, "--lam-comp", 0,
             "--seed", s, "--out", O / f"control_s{s}.pth", *common], O / f"5_control_s{s}.log", env)

    # 6. compare ------------------------------------------------------------------
    say("6. final vs control on validation")
    res = evaluate({"final": [str(O / f"final_s{s}.pth") for s in args.seeds],
                    "control": [str(O / f"control_s{s}.pth") for s in args.seeds]}, env)
    lines = []
    for g in ("final", "control"):
        r = res[g]
        mean, sd = r.mean(0), r.std(0)
        record[g] = {"per_seed": r.tolist(), "mean": mean.tolist(), "sd": sd.tolist()}
        lines.append(f"  {g:<8} acc {mean[0]:5.1f} +- {sd[0]:.1f}   AUROC {mean[1]:.3f} +- {sd[1]:.3f}   "
                     f"FPR@95 {mean[2]:5.1f} +- {sd[2]:.1f}")
    f_, c_ = res["final"], res["control"]
    d_auc, d_fpr = f_[:, 1].mean() - c_[:, 1].mean(), c_[:, 2].mean() - f_[:, 2].mean()
    s_auc, s_fpr = max(f_[:, 1].std(), c_[:, 1].std()), max(f_[:, 2].std(), c_[:, 2].std())
    helps = bool(d_auc > s_auc and d_fpr > s_fpr)
    lines.append(f"  AUROC gain {d_auc:+.3f} (spread {s_auc:.3f})   FPR@95 drop {d_fpr:+.1f} (spread {s_fpr:.1f})")
    lines.append(f"  -> by the fixed rule the samples {'HELP' if helps else 'DO NOT help'}")
    record.update(auroc_gain=d_auc, fpr_drop=d_fpr, samples_help=helps)
    print("\n".join(lines), flush=True)
    (O / "summary.txt").write_text("\n".join(lines) + "\n")
    (O / "record.json").write_text(json.dumps(record, indent=2, default=float))
    say(f"done -> {O}")


if __name__ == "__main__":
    main()
