#!/usr/bin/env python3
"""
4.ProtocolB_SupportAudit.py
===========================

Purpose
-------
Audit already-processed Protocol B datasets before you spend GPU/CPU time on model
training. The script answers the question:

    "Which LOAO / open-set scenarios are even valid enough to train?"

It scans processed train/val/test splits, computes support per family per split, and
builds a scoreboard of candidate held-out families (LOAO scenarios). Only scenarios
that satisfy explicit support rules are exported as eligible manifests.

Why this script exists
----------------------
Your current pipeline already does a lot of model-side work (XGB grid search,
family-aware thresholding, tau tuning, LOAO logic). The weak point is not a lack of
knobs; it is that some Protocol B scenarios are structurally weak or invalid.

This script separates *data validity* from *model tuning*.

Outputs
-------
For each dataset, the script writes:
    - family_support_by_file.csv
    - family_support_by_split.csv
    - eligible_holdouts.csv
    - scenario_scoreboard.csv
    - manifests/<dataset>__holdout_<family>.json

Notes
-----
1) This script assumes datasets were already prepared by your PrepareDatasets_V4-style
   pipeline and therefore contain at least:
       - y_stage1_attack
       - y_stage2_family
2) It audits the *current* processed split. It does not rebuild day/file splits.
3) It is intentionally strict. A scenario that is "probably okay" but hard to defend
   should fail here rather than later in the thesis.
"""

from __future__ import annotations

import json
import os
import glob
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from ids_eval_framework.src.paths import repo_path


# =============================================================================
# CONFIGURATION
# =============================================================================
CFG: Dict[str, object] = {
    # Root folder that contains processed/<protocol>/<dataset>/train,val,test/...
    # Example:
    #   processed_V4_stratified
    "processed_root": repo_path("processed_V5_cicids17_recovery"),

    # Which subfolder under processed_root to audit.
    "protocol": "B_day_file",

    # Datasets to audit. Add or remove names here.
    "datasets": ["CICIDS2017"],

    # Output root for audit artifacts.
    "out_root": "protocolB_support_audit_out_cicids17_recovery",

    # Column names expected in the processed data.
    "y_stage1_col": "y_stage1_attack",
    "y_stage2_col": "y_stage2_family",
    "benign_label": "Benign",

    # Split names to inspect.
    "splits": ["train", "val", "test"],

    # Optional file pattern overrides. By default, the script looks for both parquet
    # and csv parts in each split directory.
    "part_patterns": ["*.parquet", "*.csv", "*.csv.gz"],

    # ---------------------------
    # Scenario validity rules
    # ---------------------------
    "support_rules": {
        # A held-out family must exist in validation if tau is tuned on validation.
        "require_unknown_in_val_when_tuning_tau": True,

        # Minimum benign rows required for stable benign-side error estimates.
        "min_benign_val": 200,
        "min_benign_test": 200,

        # Minimum holdout-family rows required to treat unknown detection as meaningful.
        "min_unknown_val": 200,
        "min_unknown_test": 200,

        # Minimum support required for each known family that remains after removing the
        # held-out family.
        "min_train_per_known_family": 200,
        "min_val_per_known_family": 200,
        "min_test_per_known_family": 200,

        # After removing the held-out family, you still want a non-trivial known-family
        # problem, not a 1-family toy problem.
        "min_known_families_after_holdout": 2,

        # If True, every known family present in train after removing the holdout must
        # also pass the support rule in val/test. This is stricter but easier to defend.
        "require_all_remaining_train_families_to_be_valid": True,
    },

    # ---------------------------
    # Manifest defaults
    # ---------------------------
    # These fields are not model-specific training decisions. They are simply bundled into
    # the scenario manifest so the grid runner can start from a clean, explicit record.
    "manifest_defaults": {
        "support_mode": "strict_drop",
        "apply_loao_stage1_values": [False, True],
        "optimize_tau_for_unknown": True,
        "unknown_label": "Unknown",
    },
}


# =============================================================================
# SMALL DATA STRUCTURES
# =============================================================================
@dataclass
class ScenarioAudit:
    dataset: str
    holdout_family: str
    scenario_valid: bool
    reasons: List[str]
    benign_val: int
    benign_test: int
    unknown_val: int
    unknown_test: int
    n_train_attack_families_total: int
    n_known_families_after_holdout: int
    n_valid_known_families: int
    valid_known_families: List[str]
    invalid_known_families: List[str]
    known_family_support_table: Dict[str, Dict[str, int]]


# =============================================================================
# IO HELPERS
# =============================================================================
def safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def canonical_col(name: str) -> str:
    """Normalize column names so csv/parquet variants behave consistently."""
    s = str(name).strip().lower()
    s = re.sub(r"\s+", "_", s)
    return s


def read_part(path: str, usecols: Optional[List[str]] = None) -> pd.DataFrame:
    """Read one processed part. Supports parquet or csv."""
    if path.lower().endswith(".parquet"):
        return pd.read_parquet(path, columns=usecols)
    if path.lower().endswith(".csv") or path.lower().endswith(".csv.gz"):
        return pd.read_csv(path, usecols=usecols)
    raise RuntimeError(f"Unsupported file type: {path}")


def list_split_parts(dataset_dir: str, split: str) -> List[str]:
    """
    Return all data parts for a split.

    The script supports either:
        <dataset>/<split>/*.parquet
        <dataset>/<split>/*.csv
    or flat patterns like:
        <dataset>/train_*.parquet
    """
    split_dir = os.path.join(dataset_dir, split)
    out: List[str] = []

    for pat in CFG["part_patterns"]:
        out.extend(sorted(glob.glob(os.path.join(split_dir, pat))))
        out.extend(sorted(glob.glob(os.path.join(dataset_dir, f"{split}_*{pat[1:]}"))))

    # Deduplicate while preserving order.
    seen = set()
    final = []
    for p in out:
        if p not in seen:
            final.append(p)
            seen.add(p)
    return final


def iter_split_parts(dataset_dir: str, split: str, usecols: Optional[List[str]] = None) -> Iterable[Tuple[str, pd.DataFrame]]:
    """Yield (basename, dataframe) for each split part."""
    for part in list_split_parts(dataset_dir, split):
        df = read_part(part, usecols=usecols)
        df.columns = [canonical_col(c) for c in df.columns]
        yield os.path.basename(part), df


# =============================================================================
# COUNTING HELPERS
# =============================================================================
def count_family_support_in_frame(
    df: pd.DataFrame,
    y_stage1_col: str,
    y_stage2_col: str,
    benign_label: str,
) -> Counter:
    """
    Count family labels in one frame.

    Benign rows are normalized to benign_label. Attack rows use y_stage2_col.
    This protects against missing / blank family labels on benign rows.
    """
    y1 = df[y_stage1_col].astype(int).to_numpy()
    y2 = df[y_stage2_col].astype(str).fillna("").to_numpy(dtype=object)

    fam = np.where(y1 == 0, benign_label, y2)
    fam = np.array([str(x).strip() if str(x).strip() else "__EMPTY__" for x in fam], dtype=object)
    return Counter(fam.tolist())


def build_support_tables(dataset: str, dataset_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build:
        1) per-file support table
        2) per-split aggregated support table
    """
    y1 = canonical_col(str(CFG["y_stage1_col"]))
    y2 = canonical_col(str(CFG["y_stage2_col"]))
    benign_label = str(CFG["benign_label"])

    file_rows: List[Dict[str, object]] = []
    split_rows: List[Dict[str, object]] = []

    for split in CFG["splits"]:
        split_counter = Counter()
        split_rows_total = 0

        for part_name, df in iter_split_parts(dataset_dir, split, usecols=[y1, y2]):
            cnt = count_family_support_in_frame(df, y1, y2, benign_label)
            rows_this_part = int(sum(cnt.values()))
            split_rows_total += rows_this_part
            split_counter.update(cnt)

            for fam, n in sorted(cnt.items()):
                file_rows.append(
                    {
                        "dataset": dataset,
                        "split": split,
                        "part": part_name,
                        "family": fam,
                        "count": int(n),
                    }
                )

        for fam, n in sorted(split_counter.items()):
            split_rows.append(
                {
                    "dataset": dataset,
                    "split": split,
                    "family": fam,
                    "count": int(n),
                    "split_rows_total": int(split_rows_total),
                }
            )

    return pd.DataFrame(file_rows), pd.DataFrame(split_rows)


# =============================================================================
# SCENARIO AUDIT LOGIC
# =============================================================================
def support_lookup(split_df: pd.DataFrame) -> Dict[str, Dict[str, int]]:
    """Convert the split support table into {split: {family: count}}."""
    out: Dict[str, Dict[str, int]] = defaultdict(dict)
    if split_df.empty:
        return out
    for _, row in split_df.iterrows():
        out[str(row["split"])][str(row["family"])] = int(row["count"])
    return out


def audit_holdout_scenario(dataset: str, split_support_df: pd.DataFrame, holdout_family: str) -> ScenarioAudit:
    """
    Decide whether one LOAO scenario is valid.

    The rule is intentionally explicit:
      - there must be enough benign data in val/test
      - there must be enough unknown (held-out) data in val/test
      - after removing the holdout, enough known families must remain
      - every remaining known family must have enough train/val/test support
    """
    rules = dict(CFG["support_rules"])
    benign_label = str(CFG["benign_label"])
    counts = support_lookup(split_support_df)

    train_counts = counts.get("train", {})
    val_counts = counts.get("val", {})
    test_counts = counts.get("test", {})

    train_attack_families = sorted(
        f for f in train_counts.keys()
        if f not in {benign_label, "__EMPTY__"}
    )

    remaining_known = [f for f in train_attack_families if f != holdout_family]
    known_support_table: Dict[str, Dict[str, int]] = {}
    valid_known: List[str] = []
    invalid_known: List[str] = []

    for fam in remaining_known:
        row = {
            "train": int(train_counts.get(fam, 0)),
            "val": int(val_counts.get(fam, 0)),
            "test": int(test_counts.get(fam, 0)),
        }
        known_support_table[fam] = row

        ok = (
            row["train"] >= int(rules["min_train_per_known_family"])
            and row["val"] >= int(rules["min_val_per_known_family"])
            and row["test"] >= int(rules["min_test_per_known_family"])
        )
        if ok:
            valid_known.append(fam)
        else:
            invalid_known.append(fam)

    benign_val = int(val_counts.get(benign_label, 0))
    benign_test = int(test_counts.get(benign_label, 0))
    unknown_val = int(val_counts.get(holdout_family, 0))
    unknown_test = int(test_counts.get(holdout_family, 0))

    reasons: List[str] = []
    if benign_val < int(rules["min_benign_val"]):
        reasons.append("benign_val_below_min")
    if benign_test < int(rules["min_benign_test"]):
        reasons.append("benign_test_below_min")

    if bool(rules["require_unknown_in_val_when_tuning_tau"]) and unknown_val < int(rules["min_unknown_val"]):
        reasons.append("unknown_val_below_min")
    if unknown_test < int(rules["min_unknown_test"]):
        reasons.append("unknown_test_below_min")

    if len(valid_known) < int(rules["min_known_families_after_holdout"]):
        reasons.append("not_enough_valid_known_families_after_holdout")

    if bool(rules["require_all_remaining_train_families_to_be_valid"]) and len(invalid_known) > 0:
        reasons.append("remaining_known_families_failed_support_rule")

    scenario_valid = len(reasons) == 0

    return ScenarioAudit(
        dataset=dataset,
        holdout_family=holdout_family,
        scenario_valid=scenario_valid,
        reasons=reasons,
        benign_val=benign_val,
        benign_test=benign_test,
        unknown_val=unknown_val,
        unknown_test=unknown_test,
        n_train_attack_families_total=len(train_attack_families),
        n_known_families_after_holdout=len(remaining_known),
        n_valid_known_families=len(valid_known),
        valid_known_families=valid_known,
        invalid_known_families=invalid_known,
        known_family_support_table=known_support_table,
    )


# =============================================================================
# MANIFEST WRITING
# =============================================================================
def write_manifest(out_dir: str, audit: ScenarioAudit, dataset_dir: str) -> str:
    """
    Write one scenario manifest.

    The manifest is deliberately data-centric. It records:
      - the processed dataset path
      - which family is held out
      - which remaining families are valid known families
      - the exact support rules used to declare this scenario valid

    Model grids stay in the grid runner, not here.
    """
    safe_mkdir(out_dir)
    defaults = dict(CFG["manifest_defaults"])

    manifest = {
        "dataset": audit.dataset,
        "protocol": str(CFG["protocol"]),
        "processed_dir": dataset_dir,
        "y_stage1_col": str(CFG["y_stage1_col"]),
        "y_stage2_col": str(CFG["y_stage2_col"]),
        "benign_label": str(CFG["benign_label"]),
        "holdout_family": audit.holdout_family,
        "valid_known_families": audit.valid_known_families,
        "invalid_known_families": audit.invalid_known_families,
        "scenario_valid": bool(audit.scenario_valid),
        "scenario_reasons": audit.reasons,
        "support_rules": dict(CFG["support_rules"]),
        "support_mode": defaults["support_mode"],
        "apply_loao_stage1_values": defaults["apply_loao_stage1_values"],
        "optimize_tau_for_unknown": defaults["optimize_tau_for_unknown"],
        "unknown_label": defaults["unknown_label"],
        "support_snapshot": audit.known_family_support_table,
    }

    fname = f"{audit.dataset}__holdout_{audit.holdout_family.replace('/', '_')}.json"
    path = os.path.join(out_dir, fname)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return path


# =============================================================================
# MAIN DRIVER
# =============================================================================
def main() -> None:
    out_root = str(CFG["out_root"])
    safe_mkdir(out_root)

    processed_root = str(CFG["processed_root"])
    protocol = str(CFG["protocol"])

    all_scoreboard_rows: List[Dict[str, object]] = []
    all_eligible_rows: List[Dict[str, object]] = []

    for dataset in CFG["datasets"]:
        dataset_dir = os.path.join(processed_root, protocol, str(dataset))
        if not os.path.isdir(dataset_dir):
            print(f"[WARN] Skipping {dataset}: dataset_dir not found -> {dataset_dir}")
            continue

        ds_out = os.path.join(out_root, str(dataset))
        manifests_out = os.path.join(ds_out, "manifests")
        safe_mkdir(ds_out)
        safe_mkdir(manifests_out)

        # Clean stale manifest JSON files from previous audit runs so the grid runner does
        # not accidentally pick up outdated or malformed leftovers. This folder is meant
        # to contain only the manifests for the *current* audit pass.
        for stale_json in glob.glob(os.path.join(manifests_out, "*.json")):
            try:
                os.remove(stale_json)
            except Exception:
                pass

        print(f"\n=== Auditing dataset: {dataset} ===")
        print(f"dataset_dir={dataset_dir}")

        file_df, split_df = build_support_tables(str(dataset), dataset_dir)
        file_csv = os.path.join(ds_out, "family_support_by_file.csv")
        split_csv = os.path.join(ds_out, "family_support_by_split.csv")
        file_df.to_csv(file_csv, index=False)
        split_df.to_csv(split_csv, index=False)
        print(f"  wrote: {file_csv}")
        print(f"  wrote: {split_csv}")

        benign_label = str(CFG["benign_label"])
        train_families = sorted(
            f for f in split_df.loc[split_df["split"] == "train", "family"].astype(str).unique().tolist()
            if f not in {benign_label, "__EMPTY__"}
        )

        dataset_scoreboard: List[Dict[str, object]] = []
        dataset_eligible: List[Dict[str, object]] = []

        for holdout_family in train_families:
            audit = audit_holdout_scenario(str(dataset), split_df, holdout_family)
            manifest_path = None
            if audit.scenario_valid:
                manifest_path = write_manifest(manifests_out, audit, dataset_dir)

            row = {
                "dataset": audit.dataset,
                "holdout_family": audit.holdout_family,
                "scenario_valid": bool(audit.scenario_valid),
                "reasons": "|".join(audit.reasons),
                "benign_val": audit.benign_val,
                "benign_test": audit.benign_test,
                "unknown_val": audit.unknown_val,
                "unknown_test": audit.unknown_test,
                "n_train_attack_families_total": audit.n_train_attack_families_total,
                "n_known_families_after_holdout": audit.n_known_families_after_holdout,
                "n_valid_known_families": audit.n_valid_known_families,
                "valid_known_families": "|".join(audit.valid_known_families),
                "invalid_known_families": "|".join(audit.invalid_known_families),
                "manifest_path": manifest_path or "",
            }
            dataset_scoreboard.append(row)
            all_scoreboard_rows.append(row)

            if audit.scenario_valid:
                eligible_row = dict(row)
                dataset_eligible.append(eligible_row)
                all_eligible_rows.append(eligible_row)

        scoreboard_csv = os.path.join(ds_out, "scenario_scoreboard.csv")
        eligible_csv = os.path.join(ds_out, "eligible_holdouts.csv")
        pd.DataFrame(dataset_scoreboard).sort_values(
            ["scenario_valid", "n_valid_known_families", "unknown_test", "unknown_val"],
            ascending=[False, False, False, False],
        ).to_csv(scoreboard_csv, index=False)
        columns = ["dataset", "holdout_family", "scenario_valid", "reasons", "benign_val", "benign_test", "unknown_val", "unknown_test", "n_train_attack_families_total", "n_known_families_after_holdout", "n_valid_known_families", "valid_known_families", "invalid_known_families", "manifest_path"]
        df_eligible = pd.DataFrame(dataset_eligible, columns=columns)
        df_eligible.sort_values(
            ["n_valid_known_families", "unknown_test", "unknown_val"],
            ascending=[False, False, False],
        ).to_csv(eligible_csv, index=False)

        print(f"  wrote: {scoreboard_csv}")
        print(f"  wrote: {eligible_csv}")
        print(f"  valid scenarios: {sum(r['scenario_valid'] for r in dataset_scoreboard)} / {len(dataset_scoreboard)}")

    # Global index files are useful when the grid runner wants to sweep across datasets.
    scoreboard_columns = [
        "dataset", "holdout_family", "scenario_valid", "reasons", "benign_val", "benign_test",
        "unknown_val", "unknown_test", "n_train_attack_families_total", "n_known_families_after_holdout",
        "n_valid_known_families", "valid_known_families", "invalid_known_families", "manifest_path",
    ]
    pd.DataFrame(all_scoreboard_rows, columns=scoreboard_columns).to_csv(
        os.path.join(out_root, "scenario_scoreboard_all.csv"),
        index=False,
    )
    pd.DataFrame(all_eligible_rows, columns=scoreboard_columns).to_csv(
        os.path.join(out_root, "eligible_holdouts_all.csv"),
        index=False,
    )

    with open(os.path.join(out_root, "audit_config_snapshot.json"), "w", encoding="utf-8") as f:
        json.dump(CFG, f, indent=2)

    print("\nDone.")


if __name__ == "__main__":
    main()
