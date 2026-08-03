from __future__ import annotations

import copy
import json
import math
import os
import re
from functools import lru_cache
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


from ids_eval_framework.src.paths import REPO_ROOT as _PATHS_REPO_ROOT, resolve_repo_path

REPO_ROOT = str(_PATHS_REPO_ROOT)


def safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_module(path: str, module_name: str):
    """Resolve the finite set of package-local modules used by native lanes."""
    from ids_eval_framework._native import (
        prepare_datasets,
        protocol_a_core,
        protocol_b_grid,
        protocol_b_open_set,
        protocol_b_support_audit,
    )

    name = os.path.basename(str(path)).lower()
    if "preparedatasets" in name or "prepare_datasets" in name or name.startswith("2."):
        return prepare_datasets
    if "supportaudit" in name or "support_audit" in name or name.startswith("4."):
        return protocol_b_support_audit
    if "corerunner" in name or "protocol_a_core" in name or name.startswith("7."):
        return protocol_a_core
    if "openset" in name or "open_set" in name or name.startswith("14."):
        return protocol_b_open_set
    if "gridrunner" in name or "protocol_b_grid" in name or name.startswith("5."):
        return protocol_b_grid
    raise ValueError(f"Unsupported native module request: {path}")


def primary_raw_dataset_configs() -> Dict[str, Dict[str, object]]:
    return {
        "CICIDS2017": {
            "root": resolve_repo_path(r"Datasets\CICIDS 2017"),
            "file_glob": "*.csv",
            "exclude_if_contains": ["features", "_plus"],
            "label_col": "Label",
            "benign_value": "BENIGN",
            "split_stratify_on": "family",
            "drop_families": ["Other"],
            "drop_cols": ["attempted_category"],
            "protocol_b_subfiles_per_file": 4,
        },
        "CICIoT2023": {
            "root": resolve_repo_path(r"Datasets\CIC IoT Dataset 2023"),
            "file_glob": "part-*.csv",
            "exclude_if_contains": ["features"],
            "label_col": "label",
            "benign_value": "BenignTraffic",
            "split_stratify_on": "fine",
            "drop_cols": ["attempted_category"],
        },
    }


def external_protocol_a_dataset_configs() -> Dict[str, Dict[str, object]]:
    return {
        "NSL-KDD": {
            "root": resolve_repo_path(r"Datasets\NSL-KDD"),
            "has_header": False,
            "sep": ",",
            "column_names": [
                "duration",
                "protocol_type",
                "service",
                "flag",
                "src_bytes",
                "dst_bytes",
                "land",
                "wrong_fragment",
                "urgent",
                "hot",
                "num_failed_logins",
                "logged_in",
                "num_compromised",
                "root_shell",
                "su_attempted",
                "num_root",
                "num_file_creations",
                "num_shells",
                "num_access_files",
                "num_outbound_cmds",
                "is_host_login",
                "is_guest_login",
                "count",
                "srv_count",
                "serror_rate",
                "srv_serror_rate",
                "rerror_rate",
                "srv_rerror_rate",
                "same_srv_rate",
                "diff_srv_rate",
                "srv_diff_host_rate",
                "dst_host_count",
                "dst_host_srv_count",
                "dst_host_same_srv_rate",
                "dst_host_diff_srv_rate",
                "dst_host_same_src_port_rate",
                "dst_host_srv_diff_host_rate",
                "dst_host_serror_rate",
                "dst_host_srv_serror_rate",
                "dst_host_rerror_rate",
                "dst_host_srv_rerror_rate",
                "label",
                "difficulty",
            ],
            "label_col": "label",
            "benign_value": "normal",
            "strip_label_period": True,
            "predefined_splits": {
                "train": ["KDDTrain+.txt"],
                "test": ["KDDTest+.txt"],
            },
            "train_val_fracs": {"train": 0.85, "val": 0.15},
            "split_stratify_on": "family",
            "drop_cols": ["difficulty"],
        },
        "UNSW-NB15": {
            "root": resolve_repo_path(r"Datasets\UNSW-NB15"),
            "file_glob": "*.csv",
            "exclude_if_contains": ["features"],
            "label_col": "attack_cat",
            "benign_value": "Normal",
            "predefined_splits": {
                "train": ["UNSW_NB15_training-set.csv"],
                "test": ["UNSW_NB15_testing-set.csv"],
            },
            "train_val_fracs": {"train": 0.85, "val": 0.15},
            "split_stratify_on": "family",
            "drop_cols": ["label", "id"],
        },
    }


def external_protocol_b_dataset_configs() -> Dict[str, Dict[str, object]]:
    return {
        "UNSW-NB15": {
            "root": resolve_repo_path(r"Datasets\UNSW-NB15\Full + Details"),
            "file_glob": "UNSW-NB15_*.csv",
            "exclude_if_contains": ["features"],
            "label_col": "attack_cat",
            "benign_value": "Normal",
            "drop_cols": ["label", "id"],
        }
    }


@lru_cache(maxsize=32)
def count_csv_rows(path: str, has_header: bool = True) -> int:
    n = 0
    with open(path, "rb") as f:
        for _ in f:
            n += 1
    if has_header and n > 0:
        n -= 1
    return n


def split_variant_name_for_dataset(dataset: str) -> str:
    if dataset == "CICIDS2017":
        return "recovered contiguous-within-day Protocol B variant"
    if dataset == "CICIoT2023":
        return "day-file Protocol B baseline"
    return "external robustness validation"


def sort_cicids_unit_key(unit: str) -> Tuple[int, int]:
    day_order = {
        "monday": 1,
        "tuesday": 2,
        "wednesday": 3,
        "thursday": 4,
        "friday": 5,
    }
    base, _, suffix = unit.partition("::")
    day = os.path.splitext(os.path.basename(base))[0].strip().lower()
    m = re.search(r"part(\d+)of(\d+)", suffix.lower())
    part_idx = int(m.group(1)) if m else 1
    return (day_order.get(day, 999), part_idx)


def sort_ciciot_file_key(name: str) -> Tuple[int, str]:
    m = re.search(r"part-(\d+)-", os.path.basename(name))
    idx = int(m.group(1)) if m else 999999
    return (idx, name)


def load_manifest_units(manifest_path: str) -> List[str]:
    manifest = json.load(open(resolve_repo_path(manifest_path), "r", encoding="utf-8"))
    items: List[str] = []
    seen = set()
    for split in ("train", "val", "test"):
        for unit in manifest.get(split, []):
            s = str(unit)
            if s not in seen:
                seen.add(s)
                items.append(s)
    return items


def window_groups(items: Sequence[str], n_groups: int) -> List[List[str]]:
    n_groups = max(1, min(int(n_groups), len(items)))
    parts = np.array_split(np.array(items, dtype=object), n_groups)
    return [[str(x) for x in arr.tolist()] for arr in parts if len(arr) > 0]


def ensure_expected_feature_cols(df: pd.DataFrame, expected_feature_cols: Optional[Sequence[str]]) -> pd.DataFrame:
    if expected_feature_cols is None:
        return df
    out = df.copy()
    for col in expected_feature_cols:
        if col not in out.columns:
            out[col] = np.nan
    keep = list(expected_feature_cols) + [c for c in out.columns if c not in expected_feature_cols]
    return out[keep]


def clean_raw_dataframe(
    df: pd.DataFrame,
    ds_name: str,
    ds_cfg: Dict[str, object],
    prep_mod,
    expected_feature_cols: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    desired_label = str(ds_cfg["label_col"])
    mapper = prep_mod.get_mapper(ds_name)
    label_col = prep_mod.canonical_col(desired_label)
    if label_col not in [prep_mod.canonical_col(c) for c in df.columns]:
        raw_cols = {prep_mod.canonical_col(c): c for c in df.columns}
        if label_col not in raw_cols:
            raise RuntimeError(f"{ds_name}: label column '{desired_label}' not found in raw frame.")

    df = prep_mod.canonicalize_columns(df)
    label_col = prep_mod.canonical_col(desired_label)

    if prep_mod.CFG.get("drop_leakage_cols", True):
        leak = prep_mod.detect_leakage_cols(list(df.columns))
        if prep_mod.CFG.get("bucket_ports", True):
            for raw in ("src_port", "dst_port", "source_port", "destination_port", "sport", "dsport", "srcport", "dstport"):
                if raw in df.columns:
                    df[f"{raw}_bucket"] = prep_mod.port_bucket(df[raw])
            leak = [c for c in leak if not c.endswith("_bucket")]
        for c in leak:
            if c in df.columns:
                df = df.drop(columns=[c])

    df = df.replace([np.inf, -np.inf], np.nan)

    fine = df[label_col].astype(str).str.strip()
    if ds_cfg.get("strip_label_period", False):
        fine = fine.str.replace(".", "", regex=False)
    family = fine.map(mapper).astype(str)

    drop_families = set(ds_cfg.get("drop_families", []) or [])
    if drop_families:
        keep = ~family.isin(list(drop_families))
        df = df.loc[keep].copy()
        fine = fine.loc[keep]
        family = family.loc[keep]

    if len(df) == 0:
        return df

    stage1 = (family != "Benign").astype(np.int8)
    df["y_stage1_attack"] = stage1
    df["y_stage2_family"] = family
    df["y_stage2_fine"] = fine

    drop_cols = {label_col}
    for dc in ds_cfg.get("drop_cols", []):
        dcc = prep_mod.canonical_col(dc)
        if dcc in df.columns:
            drop_cols.add(dcc)
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    df = ensure_expected_feature_cols(df, expected_feature_cols)
    return df


def sample_frame(df: pd.DataFrame, max_rows: Optional[int], seed: int) -> pd.DataFrame:
    if max_rows is None or len(df) <= int(max_rows):
        return df.reset_index(drop=True)
    return df.sample(n=int(max_rows), random_state=int(seed)).reset_index(drop=True)


def load_clean_unit_frame(
    ds_name: str,
    ds_cfg: Dict[str, object],
    unit_spec: str,
    prep_mod,
    expected_feature_cols: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    desired_label = str(ds_cfg["label_col"])
    has_header = bool(ds_cfg.get("has_header", True))

    if "::" in unit_spec:
        base_name, _, suffix = unit_spec.partition("::")
        source_path = os.path.join(str(ds_cfg["root"]), base_name)
        m = re.search(r"part(\d+)of(\d+)", suffix.lower())
        if m is None:
            raise RuntimeError(f"Could not parse unit spec: {unit_spec}")
        part_idx = int(m.group(1))
        n_parts = int(m.group(2))
        total_rows = count_csv_rows(source_path, has_header=has_header)
        start_row = int(math.floor((part_idx - 1) * total_rows / n_parts))
        end_row = int(math.floor(part_idx * total_rows / n_parts))
        raw = prep_mod.read_csv_slice(
            source_path,
            ds_cfg,
            start_row=start_row,
            nrows=max(0, end_row - start_row),
            usecols=None,
        )
    else:
        source_path = os.path.join(str(ds_cfg["root"]), unit_spec)
        label_actual = prep_mod.resolve_label_col_for_file(source_path, desired_label, ds_cfg)
        if label_actual is None:
            raise RuntimeError(f"{ds_name}: could not resolve label column in {source_path}")
        raw = prep_mod.read_csv_slice(source_path, ds_cfg, start_row=0, nrows=None, usecols=None)

    return clean_raw_dataframe(raw, ds_name, ds_cfg, prep_mod, expected_feature_cols=expected_feature_cols)


def load_window_frame(
    ds_name: str,
    ds_cfg: Dict[str, object],
    units: Sequence[str],
    prep_mod,
    expected_feature_cols: Optional[Sequence[str]] = None,
    per_unit_max_rows: Optional[int] = None,
    total_max_rows: Optional[int] = None,
    seed: int = 123,
) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for idx, unit in enumerate(units):
        df_unit = load_clean_unit_frame(
            ds_name=ds_name,
            ds_cfg=ds_cfg,
            unit_spec=str(unit),
            prep_mod=prep_mod,
            expected_feature_cols=expected_feature_cols,
        )
        if len(df_unit) == 0:
            continue
        df_unit = sample_frame(df_unit, per_unit_max_rows, seed=seed + idx)
        frames.append(df_unit)
    if not frames:
        return pd.DataFrame(columns=list(expected_feature_cols or []))
    out = pd.concat(frames, ignore_index=True)
    return sample_frame(out, total_max_rows, seed=seed + 999)


def transformed_feature_names(prep) -> List[str]:
    names = list(prep.num_cols)
    for col in prep.cat_cols:
        inv = sorted(prep.cat_maps[col].items(), key=lambda kv: kv[1])
        names.extend([f"{col}={cat}" for cat, _ in inv])
    return names


def base_feature_name(transformed_name: str) -> str:
    if "=" in transformed_name:
        return transformed_name.split("=", 1)[0]
    return transformed_name


def protocol_a_rf_winners(protocol_a_summary_csv: str) -> pd.DataFrame:
    df = pd.read_csv(resolve_repo_path(protocol_a_summary_csv))
    df = df.loc[(df["model_family"] == "rf") & (df["policy_variant"] == "strict_tau")].copy()
    df = df.sort_values(
        ["dataset", "system_macro_f1_supported_labels", "run_name"],
        ascending=[True, False, False],
    )
    return df.drop_duplicates(subset=["dataset"], keep="first").reset_index(drop=True)


def load_json(path: str) -> Dict[str, object]:
    return json.load(open(resolve_repo_path(path), "r", encoding="utf-8"))
