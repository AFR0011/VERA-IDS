#!/usr/bin/env python3
"""C03 RF Stage-2 shared-fit precondition; NO FITTING or test-row access.

Run at VERA-IDS root after the approved C03 bundle/candidate preflight:
    python -B c03_rf_stage2_sharing_gate.py

Imports the pinned historical training helpers, binds three archived profile
specifications and checks the two RF Stage-2 paths using a tiny SYNTHETIC train
frame. The stand-in feature matrix is intentionally NOT a check of the full
1.2M-row training tensor, full-preprocessor bytes, or a model fit. A production
runner must construct one Stage-2 tensor and weight array, commit their hashes,
and pass those SAME objects to the shared RF Stage-2 fit.

Never opens the actual train/val/test Parquet files, model predictions or test
labels; never calls fit, reads or modifies the Git repository, or writes files.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import inspect
import json
import subprocess
import sys
from pathlib import Path

from c03_paths import ROOT, POLICY
PINNED = "ca13f1097f851ce6693658ccc0c6dd35480fabd6"
POLICY_SHA = "bf56f938d1fb193d2b383deeeec29f9d84759a4be829f6a0011f1bc20fe5e099"
LABELS = ("BruteForce", "DDoS", "DoS", "Other", "Scan/Recon")
RF_A = "rf_inv_family_clipped"
RF_B = "rf_class_weight_balanced"


def ensure(condition: bool, description: str):
    if not bool(condition):
        raise RuntimeError("BLOCKED: " + description)


def digest(*arrays):
    h = hashlib.sha256()
    for arr in arrays:
        import numpy as np
        a = np.ascontiguousarray(arr)
        h.update(str(a.dtype).encode())
        h.update(str(a.shape).encode())
        h.update(a.tobytes(order="C"))
    return h.hexdigest()


def csr_digest(x):
    from scipy.sparse import csr_matrix
    m = csr_matrix(x)
    m.sort_indices()
    return digest(m.indptr, m.indices, m.data)


def modules():
    p = ROOT / "src" / "ids_eval_framework" / "_native" / "protocol_b_grid.py"
    ensure(p.is_file(), "must run at the VERA-IDS repository root")
    src = str(ROOT / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    grid = importlib.import_module("ids_eval_framework._native.protocol_b_grid")
    engine = importlib.import_module("ids_eval_framework.src.two_stage_engine")
    for module in (grid, engine):
        source = Path(inspect.getfile(module)).resolve()
        ensure(source.is_relative_to(ROOT / "src"), "historical module imported from outside repo: " + str(source))
    return grid, engine


def bind_contract():
    import c03_candidate_fit_contract as fit_contract
    ensure(subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip() == PINNED,
           "Git HEAD differs from the pinned audited commit")
    ensure(hashlib.sha256(POLICY.read_bytes()).hexdigest() == POLICY_SHA,
           "scientific policy hash changed")
    path = ROOT / fit_contract.AGG
    import csv
    with path.open(encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    contract = fit_contract.candidate_contract(rows)
    profiles = contract["profiles"]
    ensure(len(profiles) == 3, "not exactly three historical candidate profiles")
    ensure(profiles[RF_A]["stage2_params"] == profiles[RF_B]["stage2_params"],
           "RF Stage-2 hyperparameters not identical")
    ensure(profiles[RF_A]["stage2_weight"] == profiles[RF_B]["stage2_weight"] == "balanced",
           "RF Stage-2 weighting not identical")
    ensure(profiles[RF_A]["apply_loao_stage1"] and profiles[RF_B]["apply_loao_stage1"],
           "RF Stage1 LOAO must be strict")
    return profiles


def sample_frame():
    import numpy as np
    import pandas as pd
    rng = np.random.RandomState(123)
    labels = (["Benign"] * 17 + [v for v in LABELS for _ in range(12)] + ["Botnet"] * 16)
    order = rng.permutation(len(labels))
    fam = np.asarray(labels, dtype=object)[order]
    frame = pd.DataFrame({
        "f0": rng.normal(size=len(fam)).astype("float64"),
        "f1": rng.uniform(size=len(fam)).astype("float64"),
        "y_stage1_attack": (fam != "Benign").astype("int8"),
        "y_stage2_family": fam,
    })
    return frame


def stage2_payload(frame, prep, grid, profile):
    import numpy as np
    known = (frame["y_stage1_attack"].astype(int) == 1) & frame["y_stage2_family"].astype(str).isin(LABELS)
    selected = frame.loc[known].reset_index(drop=True)
    mapping = {x: i for i, x in enumerate(LABELS)}
    y = np.asarray([mapping[x] for x in selected["y_stage2_family"]], dtype=np.int32)
    x = prep.transform(selected.drop(columns=["y_stage1_attack", "y_stage2_family"]))
    w = grid.stage2_sample_weights(y, len(LABELS), profile["stage2_weight"])
    ensure(w is not None, "balanced stage2 sample weights unexpectedly missing")
    return {"x": x, "y": y, "weights": w,
            "row_identity": digest(np.flatnonzero(known.to_numpy(dtype=bool)).astype(np.int64))}


def identical_fit_inputs(a, b, model_a, model_b):
    import numpy as np
    ensure(a["row_identity"] == b["row_identity"], "training subset row identities differ")
    ensure(a["x"].shape == b["x"].shape, "feature matrix shapes differ")
    ensure(csr_digest(a["x"]) == csr_digest(b["x"]), "feature tensors differ")
    ensure(digest(a["y"]) == digest(b["y"]), "class indices differ")
    ensure(digest(a["weights"]) == digest(b["weights"]), "sample weights differ")
    pa, pb = model_a.get_params(deep=False), model_b.get_params(deep=False)
    ensure(pa == pb, "model constructor params/RNG differ")
    ensure(pa["random_state"] == 133, "Stage2 RF random seed not seed+10=133")
    ensure(pa["n_estimators"] == 300 and pa["max_depth"] is None,
           "historical RF tree count/depth changed")
    ensure(np.isfinite(a["weights"]).all() and np.all(a["weights"] > 0),
           "Stage2 weights are not positive finite")
    return csr_digest(a["x"]), digest(a["y"]), digest(a["weights"])


def main():
    import numpy as np
    print("=== C03 STAGE-2 RF SHARED-FIT IDENTITY CONTRACT (NO FIT) ===")
    profiles = bind_contract()
    grid, engine = modules()
    ensure(callable(grid.stage2_sample_weights) and callable(grid.build_stage2_model),
           "missing historical Stage2 helpers")
    # Pin critical old function structure: original source must derive Stage2
    # training from train_df, not the Stage1 profile-dependent frame.
    src = inspect.getsource(grid.run_one_combo)
    for needle in ("attack_train_df = train_df.loc[", "y2_train_idx =", "X2_train = prep.transform(",
                   "s2_weights = stage2_sample_weights(", "seed=seed + 10"):
        ensure(needle in src, "historical Stage2 code contract missing: " + needle)
    ensure("attack_train_df = s1_train_df" not in src,
           "Stage2 source incorrectly depends on Stage1 filtering frame")
    prep = engine.Preprocessor(num_cols=["f0", "f1"], cat_cols=[],
                               num_mean=np.zeros(2, dtype=np.float64),
                               num_std=np.ones(2, dtype=np.float64),
                               cat_maps={}, cat_offsets={}, n_num=2, n_cat=0)
    frame = sample_frame()
    a = stage2_payload(frame, prep, grid, profiles[RF_A])
    b = stage2_payload(frame, prep, grid, profiles[RF_B])
    model_a = grid.build_stage2_model("rf", dict(profiles[RF_A]["stage2_params"]), len(LABELS), seed=133)
    model_b = grid.build_stage2_model("rf", dict(profiles[RF_B]["stage2_params"]), len(LABELS), seed=133)
    hx, hy, hw = identical_fit_inputs(a, b, model_a, model_b)
    ensure(a["x"].shape[0] == 60 and len(a["y"]) == 60,
           "unexpected synthetic known-family selection")
    ensure((a["y"] >= 0).all() and (a["y"] < 5).all(), "invalid known class encodings")
    # NEGATIVE CONTROLS: unchanged comparator must reject each single-factor edit.
    negative = []
    for kind in ("features", "labels", "weights", "seed", "row_identity"):
        faulty = dict(b)
        different_model = model_b
        if kind == "features":
            faulty["x"] = b["x"].copy()
            faulty["x"].data[0] += 0.123
        elif kind == "labels":
            faulty["y"] = b["y"].copy(); faulty["y"][0] = (int(faulty["y"][0]) + 1) % 5
        elif kind == "weights":
            faulty["weights"] = b["weights"].copy(); faulty["weights"][0] += .001
        elif kind == "seed":
            different_model = grid.build_stage2_model("rf", dict(profiles[RF_B]["stage2_params"]), len(LABELS), seed=134)
        elif kind == "row_identity":
            faulty["row_identity"] = "0" * 64
        try:
            identical_fit_inputs(a, faulty, model_a, different_model)
        except RuntimeError:
            negative.append(kind)
        else:
            raise RuntimeError("Negative-control difference escaped contract: " + kind)
    print("Historical source: actual Stage2 helper / training derivation verified")
    print("Historical RF Stage2: 300 trees, unlimited max depth, seed=133, balanced weights")
    print("Synthetic known-family rows: 60, shared feature count: 2")
    print("Identical input fingerprints (synthetic):")
    print("  feature CSR SHA256:", hx)
    print("  label SHA256:      ", hy)
    print("  weights SHA256:    ", hw)
    print("Difference detection:", ", ".join(negative), "(5/5 negative controls)")
    print("FIT CALLS: 0; REAL TRAIN/VAL/TEST OBSERVATIONS READ: 0")
    print("SCOPE: common RF Stage2 is justified BY DESIGN, provided production runner supplies")
    print("one exact sampled training feature tensor, label vector, weight array and seed.")
    print("LIMIT: This check DOES NOT prove real sampled tensor identity or fitted model equivalence.")
    print("RESULT: PASS RF STAGE2 COMMON-FIT PRECONDITION; REAL FITTING NOT AUTHORIZED")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("BLOCKED:", exc, file=sys.stderr)
        sys.exit(2)
