# 02_prepare_datasets_v3.py
# Protocol A (stratified, row-wise) + Protocol B (file/day split)
#
# Fixes vs V2:
# - Better handling of rare classes: merges tiny fine labels into "__RARE__<family>" to ensure they appear in val/test if their family is big enough.
# - More robust leakage detection (including common variants like *_ip_dec, source/dest_ip, *_port, timestamp, flow_id, etc.)
#
# Requirements: pandas, numpy, (optional) pyarrow for parquet

import os, re, json, glob, time, hashlib, itertools, sys
from collections import Counter, defaultdict
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd

from ids_eval_framework.src.paths import repo_path

# =========================
# Config
# =========================
CFG = {
    # Choose: "A_stratified" (recommended baseline) or "B_day_file" (distribution shift / zero-day-ish split)
    "protocol": "B_day_file",

    # Output layout: processed/<PROTOCOL>/<DATASET>/{train,val,test}/part_*.parquet
    "out_root": repo_path("processed_V5_cicids17_recovery"),

    # Keep the sprint-focused recovery lane narrow so we do not rewrite unrelated outputs.
    "active_datasets": ["CICIDS2017"],

    # Split fractions for Protocol A
    "split_fracs": {"train": 0.70, "val": 0.15, "test": 0.15},

    # Protocol B (support-aware file/day split) controls.
    # These only matter when protocol == "B_day_file".
    #
    # split_mode:
    #   - "support_aware": search for a file-level split that preserves Benign in train/val/test
    #                      and keeps enough known-family support for later LOAO/open-set runs.
    #   - "sorted_slice": legacy behavior (kept only as a fallback/reference).
    "protocol_b_split_mode": "support_aware",

    # Minimum benign rows required in each split for Protocol B to be considered usable.
    # You may pass either:
    #   - a single int (same minimum for train/val/test), or
    #   - a dict like {"train": 1000, "val": 1000, "test": 1000}
    "protocol_b_min_benign_rows": {"train": 1000, "val": 1000, "test": 1000},

    # Minimum known-family support required for a family to count as a "known family"
    # in each split when auditing a Protocol B candidate split.
    # Values may be tuned; these are sane defaults for your LOAO/open-set setup.
    "protocol_b_min_family_support": {"train": 100, "val": 50, "test": 50},

    # A Protocol B split is considered useful only if at least this many known attack
    # families survive in BOTH val and test after applying the support thresholds above.
    "protocol_b_min_known_families_val_test": 2,

    # Minimum support required for a family to be considered a candidate LOAO holdout.
    # This matters because later tau/threshold tuning uses validation unknowns.
    "protocol_b_unknown_support_for_holdout": {"val": 30, "test": 50},

    # Search settings for support-aware Protocol B.
    # For small numbers of files we enumerate exactly; for larger ones we use randomized search.
    "protocol_b_exhaustive_max_files": 12,
    "protocol_b_search_restarts": 128,
    "protocol_b_search_steps": 400,
    "protocol_b_count_search_delta": 1,

    # Minimum number of raw files per split. These are file counts, not row counts.
    "protocol_b_min_files_per_split": {"train": 1, "val": 1, "test": 1},

    # When target_per_fine_label is enabled, Protocol B now uses split-aware fine-label
    # caps instead of one global cap_state that can starve val/test.
    "protocol_b_split_aware_cap": True,

    # Default stratification level for Protocol A if dataset doesn't override:
    #   "family" = y_stage2_family, "fine" = y_stage2_fine (recommended for CICIoT2023)
    "split_stratify_on_default": "family",

    # Chunk size while reading CSVs (keep modest on Windows)
    "chunksize": 200_000,

    # Leakage handling
    "drop_leakage_cols": True,
    "bucket_ports": True,  # keeps coarse port info without leaking exact port numbers

    # Rare-class handling:
    # If a stratum has too few total samples to appear in BOTH val and test reliably, we merge it.
    # This is applied for fine-stratification only (fine labels merged into "__RARE__<family>").
    #
    # Hard minimum per stratum per split (val AND test). E.g., 50 => need at least 150 total for that stratum.
    "min_per_split_per_stratum": 50,

    # Default rare-fine threshold (dataset can override). The effective threshold is:
    #   max(rare_fine_min_total, 3*min_per_split_per_stratum)
    "rare_fine_min_total_default": 500,

    # Columns that should NEVER be treated as features (we keep them as targets/metadata, then drop label)
    "never_feature_cols": {
        "y_stage1_attack", "y_stage2_family", "y_stage2_fine",
        "label", "attack", "class", "category", "target"
    },

    "seed": 123,

    "datasets": {
        "CICIoT2023": {
            "root": repo_path("Datasets", "CIC IoT Dataset 2023"),
            "file_glob": "part-*.csv",
            # exclude accidental non-data files if present
            "exclude_if_contains": ["features"],

            "label_col": "label",
            "benign_value": "BenignTraffic",

            # keep dataset trainable; applied BEFORE quotas (cap-aware quotas)
            "target_per_fine_label": 300_000,

            # IMPORTANT: for IoT we stratify by fine label so val/test contain the big DDoS/DoS subtypes too
            "split_stratify_on": "fine",

            # merge tiny fine labels into "__RARE__<family>" so val/test don't miss classes
            "rare_fine_min_total": 500,

            "drop_cols": ["attempted_category"],  # dropped if exists
        },

        "CICIDS2017": {
            "root": repo_path("Datasets", "CICIDS 2017"),
            "file_glob": "*.csv",
            "exclude_if_contains": ["features", "_plus"],

            "label_col": "Label",
            "benign_value": "BENIGN",

            # CICIDS is smaller; stratifying on family is usually enough
            "split_stratify_on": "family",
            # Option A: drop ultra-rare tail family entirely (removes missing-class issues)
            "drop_families": ["Other"],

            # Recovery lane:
            # - whole-file day splits are too sparse for defensible Protocol B holdouts
            # - keep day provenance, but split each raw day file into contiguous subfiles
            "protocol_b_partition_mode": "contiguous_subfiles",
            "protocol_b_subfiles_per_file": 4,
            "protocol_b_exhaustive_max_files": 20,
            "protocol_b_export_candidate_matrix": False,
            "protocol_b_assess_wholefile_baseline": True,
            "protocol_b_plus_policy": "assess_only",
        },

        # "NSL-KDD": {
        #     # NSL-KDD website lists train/test files (KDDTrain+.txt / KDDTest+.txt).
        #     "root": os.path.join("Datasets", "NSL-KDD"),

        #     # NSL-KDD files are typically headerless, comma-separated.
        #     "has_header": False,
        #     "sep": ",",
        #     "column_names": [
        #         "duration","protocol_type","service","flag","src_bytes","dst_bytes","land","wrong_fragment","urgent",
        #         "hot","num_failed_logins","logged_in","num_compromised","root_shell","su_attempted","num_root",
        #         "num_file_creations","num_shells","num_access_files","num_outbound_cmds","is_host_login","is_guest_login",
        #         "count","srv_count","serror_rate","srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
        #         "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count","dst_host_same_srv_rate",
        #         "dst_host_diff_srv_rate","dst_host_same_src_port_rate","dst_host_srv_diff_host_rate","dst_host_serror_rate",
        #         "dst_host_srv_serror_rate","dst_host_rerror_rate","dst_host_srv_rerror_rate","label","difficulty"
        #     ],

        #     # Use the attack label (fine) column; mapping collapses into {DoS, Probe, R2L, U2R, Benign, Other}.
        #     "label_col": "label",
        #     "benign_value": "normal",
        #     "strip_label_period": True,  # some variants use "normal."

        #     # Prefer the dataset's predefined train/test files, then we split TRAIN into train/val.
        #     "predefined_splits": {
        #         "train": ["KDDTrain+.txt"],
        #         "test":  ["KDDTest+.txt"],
        #     },
        #     "train_val_fracs": {"train": 0.85, "val": 0.15},

        #     "split_stratify_on": "family",
        #     "drop_cols": ["difficulty"],  # not a feature
        # },

        # "UNSW-NB15": {
        #     "root": os.path.join("Datasets", "UNSW-NB15"),
        #     "file_glob": "*.csv",
        #     "exclude_if_contains": ["features"],

        #     # UNSW typically has both `attack_cat` (multiclass) and `label` (binary).
        #     # We use attack_cat as the fine label and DROP the binary `label` column to avoid leakage.
        #     "label_col": "attack_cat",
        #     "benign_value": "Normal",

        #     "predefined_splits": {
        #         "train": ["*training*set*.csv", "*training-set*.csv", "UNSW_NB15_training-set.csv"],
        #         "test":  ["*testing*set*.csv",  "*testing-set*.csv",  "UNSW_NB15_testing-set.csv"],
        #     },
        #     "train_val_fracs": {"train": 0.85, "val": 0.15},

        #     "split_stratify_on": "family",
        #     "drop_cols": ["label", "id"],  # drop binary label + row id
        # },

        },
    }

# =========================
# Helpers
# =========================
def safe_mkdir(p: str):
    os.makedirs(p, exist_ok=True)

def canonical_col(name: str) -> str:
    s = str(name).strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s

def canonicalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [canonical_col(c) for c in df.columns]
    return df

def try_parquet() -> bool:
    try:
        import pyarrow  # noqa
        return True
    except Exception:
        return False

PARQUET_OK = try_parquet()

def write_part(df: pd.DataFrame, out_dir: str, part_idx: int):
    safe_mkdir(out_dir)
    if PARQUET_OK:
        out_path = os.path.join(out_dir, f"part_{part_idx:05d}.parquet")
        # Ensure pyarrow doesn't try to coerce mixed-type/object columns to numeric
        # by converting object-typed columns to pandas' string dtype first.
        df_out = df.copy()
        obj_cols = list(df_out.select_dtypes(include=["object"]).columns)
        for c in obj_cols:
            try:
                df_out[c] = df_out[c].astype("string")
            except Exception:
                df_out[c] = df_out[c].astype(str)
        df_out.to_parquet(out_path, index=False)
    else:
        out_path = os.path.join(out_dir, f"part_{part_idx:05d}.csv.gz")
        df.to_csv(out_path, index=False, compression="gzip")

def port_bucket(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce").fillna(-1).astype(int)
    out = np.where(x < 0, "NA",
          np.where(x <= 1023, "well_known",
          np.where(x <= 49151, "registered", "ephemeral")))
    return pd.Series(out, index=s.index, dtype="category")

def detect_leakage_cols(cols: list) -> list:
    """
    Identify leakage-like columns (IPs, ports, timestamps, flow identifiers, row IDs).

    Notes:
    - UNSW-NB15 often uses `srcip/dstip` and `sport/dsport` (no underscores).
    - We drop raw ports but (optionally) keep coarse *_bucket versions.
    """
    cols_c = [canonical_col(c) for c in cols]
    drop = set()

    def has_any(s: str, keys) -> bool:
        return any(k in s for k in keys)

    for c in cols_c:
        # ---- explicit IDs / time ----
        if c in {"id", "row_id", "record_id", "flow_id", "flowid", "flow_identifier", "packet_id", "session_id"}:
            drop.add(c)
            continue
        if "timestamp" in c or c in {"time", "datetime"} or c.endswith("_time") or c.endswith("_ts") or c.endswith("_date"):
            drop.add(c)
            continue

        # ---- IP columns (including *_ip_dec and UNSW-style srcip/dstip) ----
        if c in {"ip", "srcip", "dstip"}:
            drop.add(c); continue
        if c.endswith("ip_dec") or c.endswith("_ip_dec"):
            drop.add(c); continue
        if re.search(r"(^|_)ip($|_)", c):
            # catches source_ip, destination_ip, local_ip, etc.
            drop.add(c); continue
        if "ip" in c and has_any(c, ["src", "dst", "source", "dest", "destination", "local"]):
            drop.add(c); continue

        # ---- Port columns (including UNSW-style sport/dsport) ----
        if c in {"port", "sport", "dsport", "srcport", "dstport"}:
            drop.add(c); continue
        if c.endswith("_port") or re.search(r"(^|_)port($|_)", c):
            drop.add(c); continue
        if "port" in c and has_any(c, ["src", "dst", "source", "dest", "destination", "local"]):
            drop.add(c); continue

    return sorted(drop)


def list_files(ds_cfg: dict) -> List[str]:
    root = ds_cfg["root"]
    patt = ds_cfg.get("file_glob", "*.csv")
    files = sorted(glob.glob(os.path.join(root, patt)))
    ex = [x.lower() for x in ds_cfg.get("exclude_if_contains", [])]
    out = []
    for f in files:
        lf = os.path.basename(f).lower()
        if any(x in lf for x in ex):
            continue
        out.append(f)
    return out


def resolve_files_from_patterns(root: str, patterns: List[str], exclude_if_contains: Optional[List[str]] = None) -> List[str]:
    """
    Resolve a list of filename globs relative to root (unless pattern is absolute).
    Example:
      ["KDDTrain+.txt", "KDDTrain+_20Percent.txt"] or ["*training-set*.csv"]
    """
    exclude_if_contains = [x.lower() for x in (exclude_if_contains or [])]
    out = []
    for pat in (patterns or []):
        pat = str(pat)
        if os.path.isabs(pat) or re.match(r"^[A-Za-z]:\\", pat):
            full_pat = pat
        else:
            full_pat = os.path.join(root, pat)
        for f in sorted(glob.glob(full_pat)):
            lf = os.path.basename(f).lower()
            if any(x in lf for x in exclude_if_contains):
                continue
            out.append(f)
    # de-dup while keeping order
    seen = set()
    uniq = []
    for f in out:
        if f not in seen:
            seen.add(f); uniq.append(f)
    return uniq


def count_data_rows(path: str, ds_cfg: dict) -> int:
    """
    Count raw data rows (excluding the header when present).
    """
    total_lines = 0
    with open(path, "rb") as f:
        for total_lines, _ in enumerate(f, start=1):
            pass
    if ds_cfg.get("has_header", True) is False:
        return int(total_lines)
    return int(max(0, total_lines - 1))


def read_csv_slice(path: str, ds_cfg: dict, start_row: int = 0, nrows: Optional[int] = None, usecols=None) -> pd.DataFrame:
    """
    Read one contiguous row slice from a CSV.

    This is used only for Protocol B recovery variants where one raw file is treated as
    multiple contiguous pseudo-files.
    """
    kwargs = dict(low_memory=True)
    kwargs["sep"] = ds_cfg.get("sep", ",")

    if ds_cfg.get("has_header", True) is False:
        kwargs["header"] = None
        kwargs["names"] = ds_cfg.get("column_names")
        if start_row > 0:
            kwargs["skiprows"] = range(start_row)
    else:
        kwargs["header"] = 0
        if start_row > 0:
            kwargs["skiprows"] = range(1, start_row + 1)

    if usecols is not None:
        kwargs["usecols"] = usecols
    if nrows is not None:
        kwargs["nrows"] = int(nrows)

    return pd.read_csv(path, **kwargs)


def protocol_b_partition_specs(path: str, ds_cfg: dict, partition_mode: Optional[str] = None) -> List[dict]:
    """
    Describe how one raw file should be exposed to the Protocol B planner.

    Default behavior is one planning unit per file. Recovery variants may expose several
    contiguous subfiles per raw file while preserving source-file traceability.
    """
    mode = str(partition_mode or ds_cfg.get("protocol_b_partition_mode", "whole_file"))
    file_name = os.path.basename(path)
    total_rows = count_data_rows(path, ds_cfg)

    if total_rows <= 0:
        return []

    if mode == "whole_file":
        return [
            {
                "unit_id": file_name,
                "unit_name": file_name,
                "source_file_path": path,
                "source_file_name": file_name,
                "partition_mode": mode,
                "segment_index": 1,
                "segment_count": 1,
                "row_start": 0,
                "row_end": int(total_rows),
                "raw_row_count": int(total_rows),
            }
        ]

    if mode != "contiguous_subfiles":
        raise ValueError(f"Unsupported protocol_b_partition_mode: {mode}")

    n_parts = max(1, int(ds_cfg.get("protocol_b_subfiles_per_file", 3)))
    part_size = int(np.ceil(total_rows / max(1, n_parts)))
    specs = []
    for part_idx in range(n_parts):
        row_start = int(part_idx * part_size)
        row_end = int(min(total_rows, (part_idx + 1) * part_size))
        if row_start >= row_end:
            continue
        specs.append(
            {
                "unit_id": f"{file_name}::part{part_idx + 1}of{n_parts}",
                "unit_name": f"{file_name}::part{part_idx + 1}of{n_parts}",
                "source_file_path": path,
                "source_file_name": file_name,
                "partition_mode": mode,
                "segment_index": int(part_idx + 1),
                "segment_count": int(n_parts),
                "row_start": row_start,
                "row_end": row_end,
                "raw_row_count": int(row_end - row_start),
            }
        )
    return specs



def resolve_label_col_for_file(csv_path: str, desired_label_raw: str, ds_cfg: dict) -> Optional[str]:
    """
    Returns the actual header name in the CSV that matches desired_label_raw (case/space tolerant).

    For headerless files (e.g., NSL-KDD *.txt), we assume `ds_cfg["column_names"]` is provided
    and return the desired label if it's present in that list.
    """
    desired = canonical_col(desired_label_raw)

    # Headerless (NSL-KDD etc.)
    if ds_cfg.get("has_header", True) is False:
        names = [canonical_col(x) for x in (ds_cfg.get("column_names") or [])]
        return desired_label_raw if desired in names else None

    try:
        hdr = pd.read_csv(csv_path, nrows=0).columns.tolist()
    except Exception:
        return None
    if not hdr:
        return None
    hdr_c = [canonical_col(h) for h in hdr]
    # If it looks like a "features list" file: first columns often No/Name/Type/Description
    if len(hdr_c) >= 4 and hdr_c[:4] == ["no", "name", "type", "description"]:
        return None
    # Find match
    for raw, c in zip(hdr, hdr_c):
        if c == desired:
            return raw
    # not found: treat as non-data
    return None


def iter_read_csv(path: str, ds_cfg: dict, usecols=None):
    """
    Unified CSV reader:
      - Supports headerless comma-separated files (NSL-KDD *.txt)
      - Supports normal CSV files with headers
    """
    kwargs = dict(chunksize=CFG["chunksize"], low_memory=True)

    sep = ds_cfg.get("sep", ",")
    kwargs["sep"] = sep

    if ds_cfg.get("has_header", True) is False:
        kwargs["header"] = None
        kwargs["names"] = ds_cfg.get("column_names")
    else:
        kwargs["header"] = 0

    if usecols is not None:
        kwargs["usecols"] = usecols

    # keep strings as-is; we'll coerce later
    return pd.read_csv(path, **kwargs)

    if not hdr:
        return None
    hdr_c = [canonical_col(h) for h in hdr]
    # If it looks like a "features list" file: first columns often No/Name/Type/Description
    if len(hdr_c) >= 4 and hdr_c[:4] == ["no", "name", "type", "description"]:
        return None
    # Find match
    for raw, c in zip(hdr, hdr_c):
        if c == desired:
            return raw
    # not found: treat as non-data
    return None


# =========================
# Label mapping
# =========================
def mapper_ciciot2023(fine_label: str) -> str:
    s = str(fine_label).strip()
    if s == "BenignTraffic":
        return "Benign"
    # simple family mapping by prefix / keywords (adjust if you have a better mapping file)
    # examples in CICIoT2023: DDoS-..., DoS-..., Mirai-..., MITM-..., Recon-...
    if s.startswith("DDoS-") or "DDoS" in s:
        return "DDoS"
    if s.startswith("DoS-") or "DoS" in s:
        return "DoS"
    if s.startswith("Mirai") or "Botnet" in s or "botnet" in s:
        return "Botnet"
    if "Recon" in s or "Scan" in s or "Port" in s:
        return "Scan/Recon"
    if "Brute" in s or "Patator" in s:
        return "BruteForce"
    if "Web" in s or "HTTP" in s:
        return "Web/App"
    if "MITM" in s or "Spoof" in s or "ARP" in s:
        return "Other"
    return "Other"

def mapper_cicids2017(fine_label: str) -> str:
    s = str(fine_label).strip()
    if s.upper() == "BENIGN":
        return "Benign"
    # very coarse family mapping (good enough for stage-2 families)
    # refine if you prefer different taxonomy
    u = s.lower()
    if "ddos" in u:
        return "DDoS"
    if "dos" in u:
        return "DoS"
    if "portscan" in u or "scan" in u:
        return "Scan/Recon"
    if "patator" in u or "brute" in u:
        return "BruteForce"
    if "web attack" in u or "xss" in u or "sql" in u:
        return "Web/App"
    if "botnet" in u:
        return "Botnet"
    return "Other"

# --- NSL-KDD attack taxonomy (standard 4-category mapping) ---
_NSL_DOS = {
    "back","land","neptune","pod","smurf","teardrop","mailbomb","apache2","processtable","udpstorm"
}
_NSL_PROBE = {"ipsweep","nmap","portsweep","satan","mscan","saint"}
_NSL_R2L = {
    "ftp_write","guess_passwd","imap","multihop","phf","spy","warezclient","warezmaster",
    "sendmail","named","snmpgetattack","snmpguess","xlock","xsnoop","worm"
}
_NSL_U2R = {"buffer_overflow","loadmodule","perl","rootkit","ps","sqlattack","xterm","httptunnel"}

def mapper_nsl_kdd(fine_label: str) -> str:
    s = str(fine_label).strip()
    s = s.replace(".", "")  # some variants have trailing dot
    u = s.lower()
    if u in {"normal", "benign", "normaltraffic"}:
        return "Benign"
    if u in _NSL_DOS:
        return "DoS"
    if u in _NSL_PROBE:
        return "Probe"
    if u in _NSL_R2L:
        return "R2L"
    if u in _NSL_U2R:
        return "U2R"
    return "Other"

def mapper_unsw_nb15(fine_label: str) -> str:
    # UNSW attack_cat is already a family-like label; keep it.
    s = str(fine_label).strip()
    if not s or s.lower() in {"nan", "none"}:
        return "Other"
    if s.lower() in {"normal", "benign"}:
        return "Benign"
    # normalize common spacing/casing
    return s

def get_mapper(ds_name: str):
    if ds_name == "CICIoT2023":
        return mapper_ciciot2023
    if ds_name == "CICIDS2017":
        return mapper_cicids2017
    if ds_name == "NSL-KDD":
        return mapper_nsl_kdd
    if ds_name == "UNSW-NB15":
        return mapper_unsw_nb15
    return lambda x: "Other"


# =========================
# Protocol B: support-aware split by file/day
# =========================
def _cfg_get_dict_or_scalar(value, default_keys=("train", "val", "test"), default_scalar=0) -> Dict[str, int]:
    """
    Helper for config values that may be either:
      - a scalar int, or
      - a split-wise dict {"train": ..., "val": ..., "test": ...}
    """
    if isinstance(value, dict):
        return {k: int(value.get(k, default_scalar)) for k in default_keys}
    return {k: int(value if value is not None else default_scalar) for k in default_keys}


def _protocol_b_target_file_counts(n_files: int, ds_cfg: dict) -> List[Dict[str, int]]:
    """
    Generate candidate split-size triplets (train/val/test file counts).

    Why this exists:
    - The old code used one exact rounded fraction, e.g. 5/1/1 for 7 files.
    - For small datasets that can be too rigid, so we search nearby count triplets too.

    Override options:
    - ds_cfg["protocol_b_file_counts"] = {"train": x, "val": y, "test": z}
      forces a single exact triplet.
    """
    manual = ds_cfg.get("protocol_b_file_counts")
    if manual:
        out = {
            "train": int(manual["train"]),
            "val": int(manual["val"]),
            "test": int(manual["test"]),
        }
        if sum(out.values()) != n_files:
            raise ValueError(f"protocol_b_file_counts must sum to n_files={n_files}, got {out}")
        return [out]

    mins = _cfg_get_dict_or_scalar(CFG.get("protocol_b_min_files_per_split", {"train": 1, "val": 1, "test": 1}))
    fr = CFG["split_fracs"]
    target = {
        "test": max(mins["test"], int(round(fr["test"] * n_files))),
        "val": max(mins["val"], int(round(fr["val"] * n_files))),
    }
    target["train"] = n_files - target["val"] - target["test"]

    while target["train"] < mins["train"]:
        if target["test"] > mins["test"]:
            target["test"] -= 1
        elif target["val"] > mins["val"]:
            target["val"] -= 1
        else:
            break
        target["train"] = n_files - target["val"] - target["test"]

    if target["train"] < mins["train"]:
        raise RuntimeError(
            f"Protocol B file split impossible with n_files={n_files} and min files per split={mins}"
        )

    delta = int(CFG.get("protocol_b_count_search_delta", 1))
    options = []
    for n_val in range(max(mins["val"], target["val"] - delta), min(n_files, target["val"] + delta) + 1):
        for n_test in range(max(mins["test"], target["test"] - delta), min(n_files, target["test"] + delta) + 1):
            n_train = n_files - n_val - n_test
            if n_train < mins["train"]:
                continue
            opt = {"train": int(n_train), "val": int(n_val), "test": int(n_test)}
            options.append(opt)

    # de-dup + sort by closeness to the target ratio
    uniq = []
    seen = set()
    for opt in options:
        key = (opt["train"], opt["val"], opt["test"])
        if key not in seen:
            seen.add(key)
            uniq.append(opt)

    def ratio_penalty(opt: Dict[str, int]) -> Tuple[int, int, int]:
        return (
            abs(opt["train"] - target["train"]),
            abs(opt["val"] - target["val"]),
            abs(opt["test"] - target["test"]),
        )

    uniq.sort(key=ratio_penalty)
    return uniq


def _compute_protocol_b_file_stats(ds_name: str,
                                   files: List[str],
                                   ds_cfg: dict,
                                   partition_mode: Optional[str] = None) -> Tuple[List[dict], pd.DataFrame]:
    """
    Scan Protocol B planning units using label-only reads and compute:
    - per-unit family counts
    - per-unit fine-label counts
    - total rows after family dropping

    This is the key planning stage that the old sorted-slice split never had.
    """
    desired_label = ds_cfg["label_col"]
    mapper = get_mapper(ds_name)
    drop_families = set(ds_cfg.get("drop_families", []) or [])

    infos = []
    support_rows = []
    unit_order = 0

    for f in files:
        label_actual = resolve_label_col_for_file(f, desired_label, ds_cfg)
        if label_actual is None:
            continue

        label_col = canonical_col(label_actual)
        for spec in protocol_b_partition_specs(f, ds_cfg, partition_mode=partition_mode):
            family_counts = Counter()
            fine_counts = Counter()
            kept_rows = 0
            dropped_rows = 0

            if spec["partition_mode"] == "whole_file":
                readers = iter_read_csv(f, ds_cfg, usecols=[label_actual])
            else:
                readers = [
                    read_csv_slice(
                        f,
                        ds_cfg,
                        start_row=int(spec["row_start"]),
                        nrows=int(spec["row_end"] - spec["row_start"]),
                        usecols=[label_actual],
                    )
                ]

            for chunk in readers:
                chunk = canonicalize_columns(chunk)
                if label_col not in chunk.columns:
                    continue

                fine = chunk[label_col].astype(str).str.strip()
                if ds_cfg.get("strip_label_period", False):
                    fine = fine.str.replace(".", "", regex=False)
                family = fine.map(mapper).astype(str)

                if drop_families:
                    keep = ~family.isin(list(drop_families))
                    dropped_rows += int((~keep).sum())
                    fine = fine.loc[keep]
                    family = family.loc[keep]

                kept_rows += int(len(fine))
                fine_counts.update(fine.tolist())
                family_counts.update(family.tolist())

            attack_rows = int(sum(v for fam, v in family_counts.items() if fam != "Benign"))
            info = {
                "unit_id": spec["unit_id"],
                "unit_name": spec["unit_name"],
                "source_file_path": spec["source_file_path"],
                "source_file_name": spec["source_file_name"],
                "partition_mode": spec["partition_mode"],
                "segment_index": int(spec["segment_index"]),
                "segment_count": int(spec["segment_count"]),
                "row_start": int(spec["row_start"]),
                "row_end": int(spec["row_end"]),
                "raw_row_count": int(spec["raw_row_count"]),
                "unit_order": int(unit_order),
                "total_rows": int(kept_rows),
                "dropped_rows": int(dropped_rows),
                "family_counts": dict(family_counts),
                "fine_counts": dict(fine_counts),
                "attack_rows": attack_rows,
                "benign_rows": int(family_counts.get("Benign", 0)),
                "known_family_diversity": int(sum(1 for fam, cnt in family_counts.items() if fam != "Benign" and cnt > 0)),
            }
            infos.append(info)

            for fam, cnt in sorted(family_counts.items()):
                support_rows.append(
                    {
                        "dataset": ds_name,
                        "unit_order": int(unit_order),
                        "unit_id": spec["unit_id"],
                        "unit_name": spec["unit_name"],
                        "source_file_name": spec["source_file_name"],
                        "source_file_path": spec["source_file_path"],
                        "partition_mode": spec["partition_mode"],
                        "segment_index": int(spec["segment_index"]),
                        "segment_count": int(spec["segment_count"]),
                        "row_start": int(spec["row_start"]),
                        "row_end": int(spec["row_end"]),
                        "family": fam,
                        "count": int(cnt),
                    }
                )
            unit_order += 1

    df_support = pd.DataFrame(support_rows)
    return infos, df_support


def _post_leakage_feature_set(csv_path: str, ds_cfg: dict) -> List[str]:
    hdr = list(pd.read_csv(csv_path, nrows=0).columns)
    cols = [canonical_col(c) for c in hdr]
    drop = set(detect_leakage_cols(cols))
    drop.add(canonical_col(ds_cfg["label_col"]))
    for dc in ds_cfg.get("drop_cols", []):
        drop.add(canonical_col(dc))
    return sorted([c for c in cols if c not in drop])


def write_protocol_b_plus_assessment(ds_name: str, ds_cfg: dict, out_base: str) -> None:
    """
    Record whether *_plus.csv files are compatible and whether they actually help recovery.
    """
    policy = str(ds_cfg.get("protocol_b_plus_policy", "never"))
    if policy == "never":
        return

    root = str(ds_cfg["root"])
    plus_files = sorted(glob.glob(os.path.join(root, "*_plus.csv")))
    plus_files = [p for p in plus_files if "features" not in os.path.basename(p).lower()]
    if not plus_files:
        return

    base_files = list_files(ds_cfg)
    plus_cfg = dict(ds_cfg)
    plus_cfg["exclude_if_contains"] = ["features"]

    base_infos, _ = _compute_protocol_b_file_stats(ds_name, base_files, ds_cfg, partition_mode="whole_file")
    plus_infos, _ = _compute_protocol_b_file_stats(ds_name, plus_files, plus_cfg, partition_mode="whole_file")

    base_by_stem = {os.path.splitext(x["source_file_name"])[0].lower(): x for x in base_infos}
    plus_by_stem = {
        os.path.splitext(x["source_file_name"])[0].lower().replace("_plus", ""): x
        for x in plus_infos
    }

    pair_rows = []
    for stem, base_info in sorted(base_by_stem.items()):
        plus_info = plus_by_stem.get(stem)
        if plus_info is None:
            pair_rows.append(
                {
                    "stem": stem,
                    "pair_status": "missing_plus_pair",
                }
            )
            continue

        base_path = base_info["source_file_path"]
        plus_path = plus_info["source_file_path"]
        base_post = _post_leakage_feature_set(base_path, ds_cfg)
        plus_post = _post_leakage_feature_set(plus_path, plus_cfg)
        base_set = set(base_post)
        plus_set = set(plus_post)

        pair_rows.append(
            {
                "stem": stem,
                "base_file": os.path.basename(base_path),
                "plus_file": os.path.basename(plus_path),
                "row_count_match": bool(int(base_info["raw_row_count"]) == int(plus_info["raw_row_count"])),
                "family_counts_match": bool(dict(base_info["family_counts"]) == dict(plus_info["family_counts"])),
                "base_post_leakage_features": int(len(base_post)),
                "plus_post_leakage_features": int(len(plus_post)),
                "shared_post_leakage_features": int(len(base_set & plus_set)),
                "base_only_post_leakage_features": int(len(base_set - plus_set)),
                "plus_only_post_leakage_features": int(len(plus_set - base_set)),
                "plus_only_feature_examples": sorted(list(plus_set - base_set))[:15],
            }
        )

    row_count_match_all = all(bool(r.get("row_count_match")) for r in pair_rows if "row_count_match" in r)
    family_match_all = all(bool(r.get("family_counts_match")) for r in pair_rows if "family_counts_match" in r)
    shared_ok_all = all(int(r.get("shared_post_leakage_features", 0)) > 0 for r in pair_rows if "shared_post_leakage_features" in r)

    assessment = {
        "dataset": ds_name,
        "policy": policy,
        "n_base_files": int(len(base_infos)),
        "n_plus_files": int(len(plus_infos)),
        "paired_stems": sorted(list(base_by_stem.keys())),
        "row_count_match_all_pairs": bool(row_count_match_all),
        "family_counts_match_all_pairs": bool(family_match_all),
        "shared_post_leakage_features_exist_for_all_pairs": bool(shared_ok_all),
        "schema_harmonization_possible_via_intersection": bool(shared_ok_all),
        "use_plus_files_for_recovery": False,
        "decision_rationale": [
            "The *_plus.csv files mirror the same day-level row counts and family counts as the non-plus files.",
            "Using plus files as separate Protocol B units would duplicate examples across splits and create leakage rather than new unknown-support structure.",
            "Post-leakage feature harmonization is technically possible via feature intersection, but it does not solve the structural support problem.",
        ],
        "pairs": pair_rows,
    }
    with open(os.path.join(out_base, "protocol_b_plus_file_assessment.json"), "w", encoding="utf-8") as f:
        json.dump(assessment, f, indent=2)


def _aggregate_counts_for_files(file_infos_by_name: Dict[str, dict], file_names: List[str], key: str) -> Counter:
    out = Counter()
    for fn in file_names:
        info = file_infos_by_name[fn]
        out.update(info.get(key, {}))
    return out


def _evaluate_protocol_b_split(ds_name: str,
                               splits: Dict[str, List[str]],
                               file_infos_by_name: Dict[str, dict],
                               ds_cfg: dict) -> dict:
    """
    Score one candidate file assignment for Protocol B.

    What "good" means here:
    1) Benign survives in train/val/test.
    2) Enough known families survive in BOTH val and test.
    3) At least one family can be used later as a LOAO holdout with validation unknowns.
    """
    fam_min = _cfg_get_dict_or_scalar(
        ds_cfg.get("protocol_b_min_family_support", CFG.get("protocol_b_min_family_support", {"train": 100, "val": 50, "test": 50})),
        default_scalar=0
    )
    benign_min = _cfg_get_dict_or_scalar(
        ds_cfg.get("protocol_b_min_benign_rows", CFG.get("protocol_b_min_benign_rows", {"train": 1000, "val": 1000, "test": 1000})),
        default_scalar=0
    )
    holdout_min = _cfg_get_dict_or_scalar(
        ds_cfg.get("protocol_b_unknown_support_for_holdout", CFG.get("protocol_b_unknown_support_for_holdout", {"val": 30, "test": 50})),
        default_scalar=0
    )
    min_known = int(ds_cfg.get("protocol_b_min_known_families_val_test",
                               CFG.get("protocol_b_min_known_families_val_test", 2)))
    min_holdouts = int(ds_cfg.get("protocol_b_min_candidate_holdouts",
                                  CFG.get("protocol_b_min_candidate_holdouts", 1)))

    family_counts = {
        sp: _aggregate_counts_for_files(file_infos_by_name, file_list, "family_counts")
        for sp, file_list in splits.items()
    }
    fine_counts = {
        sp: _aggregate_counts_for_files(file_infos_by_name, file_list, "fine_counts")
        for sp, file_list in splits.items()
    }

    all_known = sorted(
        {
            fam
            for sp in ("train", "val", "test")
            for fam in family_counts[sp].keys()
            if fam != "Benign"
        }
    )
    supported = {
        sp: {fam for fam in all_known if int(family_counts[sp].get(fam, 0)) >= int(fam_min[sp])}
        for sp in ("train", "val", "test")
    }
    overlap_known = sorted(list(supported["train"] & supported["val"] & supported["test"]))
    train_attack_families = sorted(
        [fam for fam, cnt in family_counts["train"].items() if fam != "Benign" and int(cnt) > 0]
    )

    benign_counts = {sp: int(family_counts[sp].get("Benign", 0)) for sp in ("train", "val", "test")}
    benign_shortfall = {
        sp: max(0, int(benign_min[sp]) - int(benign_counts[sp]))
        for sp in ("train", "val", "test")
    }
    benign_ok = all(v == 0 for v in benign_shortfall.values())

    eligible_holdouts = []
    for fam in train_attack_families:
        val_unknown = int(family_counts["val"].get(fam, 0))
        test_unknown = int(family_counts["test"].get(fam, 0))
        if val_unknown < int(holdout_min["val"]) or test_unknown < int(holdout_min["test"]):
            continue
        remaining = [g for g in train_attack_families if g != fam]
        valid_known = []
        invalid_known = []
        for g in remaining:
            row = {
                "train": int(family_counts["train"].get(g, 0)),
                "val": int(family_counts["val"].get(g, 0)),
                "test": int(family_counts["test"].get(g, 0)),
            }
            ok = (
                row["train"] >= int(fam_min["train"])
                and row["val"] >= int(fam_min["val"])
                and row["test"] >= int(fam_min["test"])
            )
            if ok:
                valid_known.append(g)
            else:
                invalid_known.append(g)

        if len(valid_known) >= min_known and len(invalid_known) == 0:
            eligible_holdouts.append(
                {
                    "holdout_family": fam,
                    "unknown_val": int(val_unknown),
                    "unknown_test": int(test_unknown),
                    "remaining_known_families": int(len(remaining)),
                    "n_valid_known_families": int(len(valid_known)),
                }
            )

    support_shortfall = max(0, min_known - len(eligible_holdouts))
    valid = bool(benign_ok and len(eligible_holdouts) >= min_holdouts)

    # Score design:
    # - validity dominates
    # - then number of eligible holdouts
    # - then number of audit-valid shared known families
    # - then benign support mass
    # - then lower penalty for benign/support shortfalls
    benign_mass_eval = benign_counts["val"] + benign_counts["test"]
    score = (
        (10**12 if valid else 0)
        + int(len(eligible_holdouts)) * 10**8
        + int(max([x.get("n_valid_known_families", 0) for x in eligible_holdouts] or [0])) * 10**6
        + int(benign_mass_eval)
        - int(sum(benign_shortfall.values())) * 10**4
        - int(support_shortfall) * 10**6
    )

    return {
        "dataset": ds_name,
        "valid": bool(valid),
        "score": int(score),
        "family_counts": {sp: {k: int(v) for k, v in family_counts[sp].items()} for sp in ("train", "val", "test")},
        "fine_counts": {sp: {k: int(v) for k, v in fine_counts[sp].items()} for sp in ("train", "val", "test")},
        "supported_known_families": {sp: sorted(list(v)) for sp, v in supported.items()},
        "overlap_known_families": overlap_known,
        "eligible_holdouts": eligible_holdouts,
        "benign_counts": benign_counts,
        "benign_shortfall": benign_shortfall,
        "n_overlap_known": int(len(overlap_known)),
        "n_eligible_holdouts": int(len(eligible_holdouts)),
    }


def _assignment_to_splits(file_infos: List[dict], train_idx: List[int], val_idx: List[int], test_idx: List[int]) -> Dict[str, List[str]]:
    return {
        "train": [file_infos[i]["unit_id"] for i in train_idx],
        "val": [file_infos[i]["unit_id"] for i in val_idx],
        "test": [file_infos[i]["unit_id"] for i in test_idx],
    }


def _seeded_protocol_b_assignment(file_infos: List[dict], counts: Dict[str, int], rng: np.random.RandomState) -> Dict[str, List[str]]:
    """
    Build one initial randomized assignment while biasing val/test toward diverse files.
    """
    idxs = list(range(len(file_infos)))

    # Diversity-biased ordering helps small evaluation splits avoid all-benign collapse.
    diverse = sorted(
        idxs,
        key=lambda i: (
            file_infos[i]["known_family_diversity"],
            file_infos[i]["benign_rows"] > 0,
            file_infos[i]["attack_rows"],
            file_infos[i]["total_rows"],
        ),
        reverse=True,
    )
    chosen = []
    chosen_set = set()

    # Reserve one strong file for val and one for test if possible.
    if counts["val"] > 0 and diverse:
        chosen.append(diverse[0]); chosen_set.add(diverse[0])
    if counts["test"] > 0:
        for i in diverse:
            if i not in chosen_set:
                chosen.append(i); chosen_set.add(i)
                break

    remaining = [i for i in idxs if i not in chosen_set]
    rng.shuffle(remaining)

    ordered = chosen + remaining
    val_idx = ordered[:counts["val"]]
    test_idx = ordered[counts["val"]:counts["val"] + counts["test"]]
    train_idx = ordered[counts["val"] + counts["test"]:]
    rng.shuffle(train_idx)
    return {"train": train_idx, "val": val_idx, "test": test_idx}


def _swap_indices(assign_idx: Dict[str, List[int]], a: str, b: str, ia: int, ib: int) -> Dict[str, List[int]]:
    out = {k: list(v) for k, v in assign_idx.items()}
    out[a][ia], out[b][ib] = out[b][ib], out[a][ia]
    return out


def _search_protocol_b_randomized(ds_name: str,
                                  file_infos: List[dict],
                                  file_infos_by_name: Dict[str, dict],
                                  ds_cfg: dict) -> Tuple[Dict[str, List[str]], dict, List[dict]]:
    """
    Randomized hill-climb search for larger file sets (e.g., CICIoT2023).
    """
    count_options = _protocol_b_target_file_counts(len(file_infos), ds_cfg)
    restarts = int(ds_cfg.get("protocol_b_search_restarts", CFG.get("protocol_b_search_restarts", 128)))
    steps = int(ds_cfg.get("protocol_b_search_steps", CFG.get("protocol_b_search_steps", 400)))
    rng = np.random.RandomState(CFG["seed"])

    best_eval = None
    best_splits = None
    best_counts = None

    for counts in count_options[:3]:
        for _ in range(restarts):
            assign_idx = _seeded_protocol_b_assignment(file_infos, counts, rng)
            splits = _assignment_to_splits(file_infos, assign_idx["train"], assign_idx["val"], assign_idx["test"])
            cur_eval = _evaluate_protocol_b_split(ds_name, splits, file_infos_by_name, ds_cfg)

            for _step in range(steps):
                a, b = rng.choice(["train", "val", "test"], size=2, replace=False)
                if not assign_idx[a] or not assign_idx[b]:
                    continue
                ia = int(rng.randint(0, len(assign_idx[a])))
                ib = int(rng.randint(0, len(assign_idx[b])))
                prop_idx = _swap_indices(assign_idx, a, b, ia, ib)
                prop_splits = _assignment_to_splits(file_infos, prop_idx["train"], prop_idx["val"], prop_idx["test"])
                prop_eval = _evaluate_protocol_b_split(ds_name, prop_splits, file_infos_by_name, ds_cfg)
                if prop_eval["score"] > cur_eval["score"]:
                    assign_idx = prop_idx
                    cur_eval = prop_eval

            if best_eval is None or cur_eval["score"] > best_eval["score"]:
                best_eval = cur_eval
                best_splits = _assignment_to_splits(file_infos, assign_idx["train"], assign_idx["val"], assign_idx["test"])
                best_counts = counts
                if best_eval["valid"] and best_eval["n_eligible_holdouts"] >= 2:
                    # good enough early stop
                    pass

    report = {
        "search_mode": "randomized",
        "candidate_file_count_options": count_options[:3],
        "selected_file_counts": best_counts,
        "best_eval": best_eval,
        "candidate_rows_exported": 0,
    }
    return best_splits, report, []


def _search_protocol_b_exhaustive(ds_name: str,
                                  file_infos: List[dict],
                                  file_infos_by_name: Dict[str, dict],
                                  ds_cfg: dict) -> Tuple[Dict[str, List[str]], dict, List[dict]]:
    """
    Exact search for small numbers of files (e.g., CICIDS2017).
    """
    n = len(file_infos)
    all_idx = list(range(n))
    count_options = _protocol_b_target_file_counts(n, ds_cfg)

    best_eval = None
    best_splits = None
    best_counts = None
    candidate_rows = [] if bool(ds_cfg.get("protocol_b_export_candidate_matrix", False)) else None

    for counts in count_options:
        n_val = counts["val"]
        n_test = counts["test"]
        for val_idx in itertools.combinations(all_idx, n_val):
            rem_after_val = [i for i in all_idx if i not in set(val_idx)]
            for test_idx in itertools.combinations(rem_after_val, n_test):
                test_set = set(test_idx)
                train_idx = [i for i in rem_after_val if i not in test_set]
                splits = _assignment_to_splits(file_infos, train_idx, list(val_idx), list(test_idx))
                ev = _evaluate_protocol_b_split(ds_name, splits, file_infos_by_name, ds_cfg)
                if candidate_rows is not None:
                    candidate_rows.append(
                        {
                            "valid": bool(ev["valid"]),
                            "score": int(ev["score"]),
                            "n_overlap_known": int(ev["n_overlap_known"]),
                            "n_eligible_holdouts": int(ev["n_eligible_holdouts"]),
                            "overlap_known_families": "|".join(ev.get("overlap_known_families", []) or []),
                            "eligible_holdouts": "|".join(
                                [x.get("holdout_family", "") for x in (ev.get("eligible_holdouts", []) or [])]
                            ),
                            "train_units": "|".join(splits["train"]),
                            "val_units": "|".join(splits["val"]),
                            "test_units": "|".join(splits["test"]),
                            "benign_train": int(ev["benign_counts"].get("train", 0)),
                            "benign_val": int(ev["benign_counts"].get("val", 0)),
                            "benign_test": int(ev["benign_counts"].get("test", 0)),
                        }
                    )
                if best_eval is None or ev["score"] > best_eval["score"]:
                    best_eval = ev
                    best_splits = splits
                    best_counts = counts
                    if ev["valid"] and ev["n_eligible_holdouts"] >= 2:
                        # exact search still continues, but this indicates the split is already usable
                        pass

    report = {
        "search_mode": "exhaustive",
        "candidate_file_count_options": count_options,
        "selected_file_counts": best_counts,
        "best_eval": best_eval,
        "candidate_rows_exported": int(len(candidate_rows or [])),
    }
    if candidate_rows:
        top_preview = sorted(candidate_rows, key=lambda r: int(r["score"]), reverse=True)[:10]
        report["top_candidate_preview"] = top_preview
    return best_splits, report, (candidate_rows or [])


def _build_protocol_b_fine_cap_quotas(file_infos_by_name: Dict[str, dict],
                                      splits: Dict[str, List[str]],
                                      cap: Optional[int]) -> Tuple[Optional[Dict[str, Dict[str, int]]], Dict[str, Dict[str, int]]]:
    """
    Build split-aware fine-label cap quotas.

    Why this exists:
    - The old code used one global cap_state while processing train first, then val, then test.
    - That can starve val/test completely (especially BenignTraffic in CICIoT2023).
    - Here we allocate each fine label's cap across splits proportionally to the raw split support.
    """
    raw_counts = {
        sp: _aggregate_counts_for_files(file_infos_by_name, file_list, "fine_counts")
        for sp, file_list in splits.items()
    }

    if cap is None:
        return None, {sp: {k: int(v) for k, v in raw_counts[sp].items()} for sp in ("train", "val", "test")}

    quotas = {"train": {}, "val": {}, "test": {}}
    all_fine = sorted(set(raw_counts["train"]) | set(raw_counts["val"]) | set(raw_counts["test"]))

    for fine in all_fine:
        per_sp = {sp: int(raw_counts[sp].get(fine, 0)) for sp in ("train", "val", "test")}
        total = int(sum(per_sp.values()))
        if total <= int(cap):
            for sp in ("train", "val", "test"):
                quotas[sp][fine] = int(per_sp[sp])
            continue

        alloc = {sp: int(np.floor(int(cap) * (per_sp[sp] / max(1, total)))) for sp in ("train", "val", "test")}
        remainder = int(cap) - sum(alloc.values())

        # distribute remaining quota to splits with the biggest fractional leftovers and available raw support
        leftovers = sorted(
            ("train", "val", "test"),
            key=lambda sp: ((int(cap) * (per_sp[sp] / max(1, total))) - alloc[sp], per_sp[sp]),
            reverse=True,
        )
        safety = 0
        while remainder > 0 and safety < 20:
            progressed = False
            for sp in leftovers:
                if remainder <= 0:
                    break
                if alloc[sp] < per_sp[sp]:
                    alloc[sp] += 1
                    remainder -= 1
                    progressed = True
            if not progressed:
                break
            safety += 1

        for sp in ("train", "val", "test"):
            quotas[sp][fine] = int(min(alloc[sp], per_sp[sp]))

    return quotas, {sp: {k: int(v) for k, v in raw_counts[sp].items()} for sp in ("train", "val", "test")}


def cap_by_fine_label_with_quotas(df: pd.DataFrame,
                                  fine_col: str,
                                  split_quota: Optional[Dict[str, int]],
                                  cap_state: Dict[str, int],
                                  rng: np.random.RandomState) -> pd.DataFrame:
    """
    Split-aware version of fine-label capping.
    """
    if split_quota is None:
        return df

    fine = df[fine_col].astype(str).to_numpy()
    keep = np.zeros(len(df), dtype=bool)
    idxs = np.arange(len(df))
    rng.shuffle(idxs)

    for i in idxs:
        lbl = fine[i]
        quota = int(split_quota.get(lbl, 0))
        if quota <= 0:
            continue
        if cap_state[lbl] < quota:
            keep[i] = True
            cap_state[lbl] += 1

    return df.loc[keep].copy()


def write_protocol_b_planning_reports(out_base: str,
                                      file_support_df: pd.DataFrame,
                                      splits: Dict[str, List[str]],
                                      search_report: dict,
                                      fine_cap_quotas: Optional[Dict[str, Dict[str, int]]],
                                      fine_raw_counts: Dict[str, Dict[str, int]],
                                      unit_infos: Optional[List[dict]] = None,
                                      candidate_rows: Optional[List[dict]] = None,
                                      name_prefix: str = "protocol_b_"):
    """
    Save planning artifacts so you can inspect whether the split is actually usable
    before training anything.
    """
    if file_support_df is not None and len(file_support_df) > 0:
        file_support_df.to_csv(os.path.join(out_base, f"{name_prefix}file_family_support.csv"), index=False)

    if unit_infos:
        unit_rows = []
        for info in unit_infos:
            unit_rows.append(
                {
                    "unit_order": int(info["unit_order"]),
                    "unit_id": info["unit_id"],
                    "unit_name": info["unit_name"],
                    "source_file_name": info["source_file_name"],
                    "source_file_path": info["source_file_path"],
                    "partition_mode": info["partition_mode"],
                    "segment_index": int(info["segment_index"]),
                    "segment_count": int(info["segment_count"]),
                    "row_start": int(info["row_start"]),
                    "row_end": int(info["row_end"]),
                    "raw_row_count": int(info["raw_row_count"]),
                    "kept_rows_after_family_drop": int(info["total_rows"]),
                    "dropped_rows": int(info["dropped_rows"]),
                }
            )
        pd.DataFrame(unit_rows).to_csv(os.path.join(out_base, f"{name_prefix}unit_manifest.csv"), index=False)

    file_manifest = {
        sp: list(paths)
        for sp, paths in splits.items()
    }
    with open(os.path.join(out_base, f"{name_prefix}file_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(file_manifest, f, indent=2)

    with open(os.path.join(out_base, f"{name_prefix}search_report.json"), "w", encoding="utf-8") as f:
        json.dump(search_report, f, indent=2)

    if candidate_rows:
        pd.DataFrame(candidate_rows).sort_values(
            ["valid", "score", "n_eligible_holdouts", "n_overlap_known"],
            ascending=[False, False, False, False],
        ).to_csv(os.path.join(out_base, f"{name_prefix}candidate_split_scores.csv"), index=False)

    # Split-level family counts and candidate holdouts from the chosen split
    best_eval = (search_report or {}).get("best_eval", {}) or {}
    fam_rows = []
    for sp, fam_counts in (best_eval.get("family_counts", {}) or {}).items():
        for fam, cnt in sorted(fam_counts.items()):
            fam_rows.append({"split": sp, "family": fam, "count": int(cnt)})
    if fam_rows:
        pd.DataFrame(fam_rows).to_csv(os.path.join(out_base, f"{name_prefix}selected_split_family_counts.csv"), index=False)

    holdouts = best_eval.get("eligible_holdouts", []) or []
    if holdouts:
        pd.DataFrame(holdouts).to_csv(os.path.join(out_base, f"{name_prefix}eligible_holdouts.csv"), index=False)
    else:
        pd.DataFrame(columns=["holdout_family", "unknown_val", "unknown_test", "remaining_known_families"]).to_csv(
            os.path.join(out_base, f"{name_prefix}eligible_holdouts.csv"), index=False
        )

    if fine_raw_counts:
        rows = []
        for sp, d in fine_raw_counts.items():
            for fine, cnt in sorted(d.items()):
                rows.append({"split": sp, "fine_label": fine, "raw_count": int(cnt)})
        pd.DataFrame(rows).to_csv(os.path.join(out_base, f"{name_prefix}fine_raw_counts.csv"), index=False)

    if fine_cap_quotas:
        rows = []
        for sp, d in fine_cap_quotas.items():
            for fine, quota in sorted(d.items()):
                rows.append({"split": sp, "fine_label": fine, "cap_quota": int(quota)})
        pd.DataFrame(rows).to_csv(os.path.join(out_base, f"{name_prefix}fine_cap_quotas.csv"), index=False)


def split_files_protocol_b(ds_name: str,
                           files: list,
                           ds_cfg: dict,
                           out_base: Optional[str] = None) -> Tuple[Dict[str, List[str]], pd.DataFrame, dict, Optional[Dict[str, Dict[str, int]]], Dict[str, Dict[str, int]], Dict[str, dict]]:
    """
    New Protocol B planner.

    Returns:
      - splits: planning-unit ids assigned to train/val/test
      - file_support_df: per-unit family counts (for diagnostics)
      - search_report: why this split was chosen + eligible holdouts
      - fine_cap_quotas: split-aware fine-label caps if target_per_fine_label is enabled
      - fine_raw_counts: raw fine counts per chosen split
      - unit_infos_by_id: source-file metadata for each planning unit
    """
    files = sorted(files)
    n = len(files)
    if n < 3:
        raise RuntimeError(f"{ds_name}: Protocol B needs >=3 files for train/val/test; use A_stratified.")

    split_mode = str(ds_cfg.get("protocol_b_split_mode", CFG.get("protocol_b_split_mode", "support_aware")))
    partition_mode = str(ds_cfg.get("protocol_b_partition_mode", "whole_file"))

    if out_base:
        write_protocol_b_plus_assessment(ds_name, ds_cfg, out_base)

    if out_base and partition_mode != "whole_file" and bool(ds_cfg.get("protocol_b_assess_wholefile_baseline", False)):
        baseline_cfg = dict(ds_cfg)
        baseline_cfg["protocol_b_export_candidate_matrix"] = True
        baseline_infos, baseline_support_df = _compute_protocol_b_file_stats(
            ds_name, files, ds_cfg, partition_mode="whole_file"
        )
        baseline_by_id = {x["unit_id"]: x for x in baseline_infos}
        if len(baseline_infos) <= int(ds_cfg.get("protocol_b_exhaustive_max_files", CFG.get("protocol_b_exhaustive_max_files", 12))):
            baseline_splits, baseline_report, baseline_candidate_rows = _search_protocol_b_exhaustive(
                ds_name, baseline_infos, baseline_by_id, baseline_cfg
            )
        else:
            baseline_splits, baseline_report, baseline_candidate_rows = _search_protocol_b_randomized(
                ds_name, baseline_infos, baseline_by_id, baseline_cfg
            )
        write_protocol_b_planning_reports(
            out_base,
            baseline_support_df,
            baseline_splits,
            baseline_report,
            fine_cap_quotas=None,
            fine_raw_counts={},
            unit_infos=baseline_infos,
            candidate_rows=baseline_candidate_rows,
            name_prefix="protocol_b_wholefile_",
        )

    # Legacy mode kept only as a fallback/reference.
    if split_mode == "sorted_slice":
        n_test = max(1, int(round(CFG["split_fracs"]["test"] * n)))
        n_val = max(1, int(round(CFG["split_fracs"]["val"] * n)))
        n_train = n - n_val - n_test
        if n_train < 1:
            n_train = 1
            while n_train + n_val + n_test > n:
                if n_test > 1:
                    n_test -= 1
                elif n_val > 1:
                    n_val -= 1
                else:
                    break
        splits = {
            "train": files[:n_train],
            "val": files[n_train:n_train + n_val],
            "test": files[n_train + n_val:n_train + n_val + n_test],
        }
        file_infos, file_support_df = _compute_protocol_b_file_stats(ds_name, files, ds_cfg, partition_mode=partition_mode)
        file_infos_by_name = {x["unit_id"]: x for x in file_infos}
        search_report = {
            "search_mode": "sorted_slice",
            "best_eval": _evaluate_protocol_b_split(ds_name, splits, file_infos_by_name, ds_cfg),
        }
        cap = ds_cfg.get("target_per_fine_label")
        fine_cap_quotas, fine_raw_counts = _build_protocol_b_fine_cap_quotas(file_infos_by_name, splits, int(cap) if cap is not None else None)
        if out_base:
            write_protocol_b_planning_reports(
                out_base,
                file_support_df,
                splits,
                search_report,
                fine_cap_quotas,
                fine_raw_counts,
                unit_infos=file_infos,
                candidate_rows=[],
            )
        return splits, file_support_df, search_report, fine_cap_quotas, fine_raw_counts, file_infos_by_name

    file_infos, file_support_df = _compute_protocol_b_file_stats(ds_name, files, ds_cfg, partition_mode=partition_mode)
    if not file_infos:
        raise RuntimeError(f"{ds_name}: no readable data files found for Protocol B planning.")
    file_infos_by_name = {x["unit_id"]: x for x in file_infos}

    exhaustive_max = int(ds_cfg.get("protocol_b_exhaustive_max_files", CFG.get("protocol_b_exhaustive_max_files", 12)))
    if len(file_infos) <= exhaustive_max:
        splits, search_report, candidate_rows = _search_protocol_b_exhaustive(ds_name, file_infos, file_infos_by_name, ds_cfg)
    else:
        splits, search_report, candidate_rows = _search_protocol_b_randomized(ds_name, file_infos, file_infos_by_name, ds_cfg)

    cap = ds_cfg.get("target_per_fine_label")
    split_aware_cap = bool(ds_cfg.get("protocol_b_split_aware_cap", CFG.get("protocol_b_split_aware_cap", True)))
    fine_cap_quotas, fine_raw_counts = (None, {})
    if split_aware_cap:
        fine_cap_quotas, fine_raw_counts = _build_protocol_b_fine_cap_quotas(
            file_infos_by_name, splits, int(cap) if cap is not None else None
        )

    if out_base:
        write_protocol_b_planning_reports(
            out_base,
            file_support_df,
            splits,
            search_report,
            fine_cap_quotas,
            fine_raw_counts,
            unit_infos=file_infos,
            candidate_rows=candidate_rows,
        )

    return splits, file_support_df, search_report, fine_cap_quotas, fine_raw_counts, file_infos_by_name


# =========================
# Protocol A: stratified quotas
# =========================
def compute_stratum_counts(ds_name: str, files: list, ds_cfg: dict, stratify_on: str, n_splits_required: int = 3) -> Tuple[Counter, set]:
    """
    Returns:
      - stratum_counts: Counter of the chosen stratification label (cap-aware if needed)
      - rare_fine: set of fine labels to merge into "__RARE__<family>" (only meaningful when stratify_on="fine")
    """
    desired_label = ds_cfg["label_col"]
    mapper = get_mapper(ds_name)
    drop_families = set(ds_cfg.get("drop_families", []) or [])

    cap = ds_cfg.get("target_per_fine_label")
    cap = int(cap) if cap is not None else None

    # Effective rare threshold: ensure each stratum can populate BOTH val and test
    min_each = int(CFG.get("min_per_split_per_stratum", 1))
    rare_min = int(ds_cfg.get("rare_fine_min_total", CFG.get("rare_fine_min_total_default", 0)) or 0)
    rare_min = max(rare_min, int(n_splits_required) * min_each)  # ensures >=min_each in train/val/test after split

    fine_counts = Counter()
    cap_state = defaultdict(int)

    for f in files:
        label_actual = resolve_label_col_for_file(f, desired_label, ds_cfg)
        if label_actual is None:
            continue
        label_col = canonical_col(label_actual)
        for chunk in iter_read_csv(f, ds_cfg, usecols=[label_actual]):
            chunk = canonicalize_columns(chunk)
            fine = chunk[label_col].astype(str).str.strip()

            if cap is not None:
                # cap-aware counting: only count up to cap per fine label
                # (we don't need perfect randomness here; it just prevents quotas from exceeding cap)
                for lbl in fine.tolist():
                    if cap_state[lbl] < cap:
                        fine_counts[lbl] += 1
                        cap_state[lbl] += 1
            else:
                fine_counts.update(fine.tolist())

    rare_fine = set()
    if stratify_on == "fine":
        for lbl, cnt in fine_counts.items():
            if cnt < rare_min:
                rare_fine.add(lbl)

        stratum_counts = Counter()
        for lbl, cnt in fine_counts.items():
            fam = mapper(lbl)
            if fam in drop_families:
                continue
            if lbl in rare_fine:
                stratum_counts["__RARE__" + fam] += cnt
            else:
                stratum_counts[lbl] += cnt
        return stratum_counts, rare_fine

    # stratify_on == "family"
    family_counts = Counter()
    for lbl, cnt in fine_counts.items():
        fam = mapper(lbl)
        if fam in drop_families:
            continue
        family_counts[fam] += cnt
    return family_counts, set()


def make_quotas_from_counts(stratum_counts: Counter) -> Dict[str, Dict[str, int]]:
    """
    Compute integer quotas per stratum per split.

    Rule:
      - If a stratum has enough samples to appear in val and test, enforce min_per_split_per_stratum.
      - Otherwise, allocate everything to train (and you'll see it in missing_strata reports).
    """
    fr = CFG["split_fracs"]
    min_each = int(CFG.get("min_per_split_per_stratum", 1))
    quotas = {"train": {}, "val": {}, "test": {}}

    for s, n in stratum_counts.items():
        n = int(n)
        if n <= 0:
            quotas["train"][s] = 0
            quotas["val"][s] = 0
            quotas["test"][s] = 0
            continue

        # If too small to guarantee coverage, send to train only
        if n < 3 * min_each:
            quotas["train"][s] = n
            quotas["val"][s] = 0
            quotas["test"][s] = 0
            continue

        # Start with minimums
        rem = n - 3 * min_each
        n_train = min_each + int(round(fr["train"] * rem))
        n_val = min_each + int(round(fr["val"] * rem))
        # Make remainder consistent
        n_test = n - n_train - n_val
        if n_test < min_each:
            # steal from train first, then val
            need = min_each - n_test
            take = min(need, n_train - min_each)
            n_train -= take
            need -= take
            if need > 0:
                take2 = min(need, n_val - min_each)
                n_val -= take2
                need -= take2
            n_test = n - n_train - n_val

        quotas["train"][s] = int(n_train)
        quotas["val"][s] = int(n_val)
        quotas["test"][s] = int(n_test)

    return quotas
def make_quotas_from_counts_train_val(stratum_counts: Counter,
                                     train_frac: float,
                                     val_frac: float) -> Dict[str, Dict[str, int]]:
    """
    Compute integer quotas per stratum for datasets with predefined test split.

    We split ONLY the provided TRAIN files into train/val (test quotas are 0 here),
    then we write the dataset's provided TEST files fully into the test split.

    Guarantees:
      - if a stratum has >= 2*min_each, it appears in BOTH train and val (at least min_each each).
      - otherwise everything goes to train (val gets 0).
    """
    min_each = int(CFG.get("min_per_split_per_stratum", 1))
    train_frac = float(train_frac)
    val_frac = float(val_frac)
    ssum = max(1e-9, train_frac + val_frac)
    train_frac = train_frac / ssum
    val_frac = val_frac / ssum

    quotas = {"train": {}, "val": {}, "test": {}}

    for s, n in stratum_counts.items():
        n = int(n)
        if n <= 0:
            quotas["train"][s] = 0
            quotas["val"][s] = 0
            quotas["test"][s] = 0
            continue

        if n < 2 * min_each:
            quotas["train"][s] = n
            quotas["val"][s] = 0
            quotas["test"][s] = 0
            continue

        rem = n - 2 * min_each
        n_train = min_each + int(round(train_frac * rem))
        n_val = n - n_train
        if n_val < min_each:
            # steal from train if needed
            need = min_each - n_val
            take = min(need, n_train - min_each)
            n_train -= take
            n_val = n - n_train

        quotas["train"][s] = int(n_train)
        quotas["val"][s] = int(n_val)
        quotas["test"][s] = 0

    return quotas



def split_chunk_by_quotas(df: pd.DataFrame, strat_col: str, quotas, filled, rng: np.random.RandomState):
    out = {"train": [], "val": [], "test": []}
    if len(df) == 0:
        return {k: df for k in out.keys()}

    strat_vals = df[strat_col].astype(str).to_numpy()
    by_stratum = defaultdict(list)
    for i, s in enumerate(strat_vals):
        by_stratum[s].append(i)

    for s, idxs in by_stratum.items():
        idxs = np.array(idxs, dtype=int)
        rng.shuffle(idxs)

        for split in ["train", "val", "test"]:
            need = quotas.get(split, {}).get(s, 0) - filled.get(split, {}).get(s, 0)
            if need <= 0 or len(idxs) == 0:
                continue
            take = min(int(need), len(idxs))
            take_idxs = idxs[:take]
            idxs = idxs[take:]
            out[split].append(df.iloc[take_idxs])
            filled[split][s] += take

        if len(idxs) > 0:
            # leftover goes to train
            out["train"].append(df.iloc[idxs])
            filled["train"][s] += len(idxs)

    return {k: (pd.concat(v, ignore_index=True) if v else df.iloc[:0].copy()) for k, v in out.items()}


def cap_by_fine_label(df: pd.DataFrame, fine_col: str, cap: int, cap_state: Dict[str, int], rng: np.random.RandomState) -> pd.DataFrame:
    if cap is None:
        return df
    fine = df[fine_col].astype(str).to_numpy()
    keep = np.zeros(len(df), dtype=bool)
    idxs = np.arange(len(df))
    rng.shuffle(idxs)
    for i in idxs:
        lbl = fine[i]
        if cap_state[lbl] < cap:
            keep[i] = True
            cap_state[lbl] += 1
    return df.loc[keep].copy()


# =========================
# Main processing
# =========================
def process_dataset(ds_name: str, ds_cfg: dict):
    protocol = CFG["protocol"]
    out_base = os.path.join(CFG["out_root"], protocol, ds_name)
    safe_mkdir(out_base)

    predef = ds_cfg.get("predefined_splits", None)
    ex = ds_cfg.get("exclude_if_contains", []) or []

    train_files = None
    test_files = None

    if predef:
        train_files = resolve_files_from_patterns(ds_cfg["root"], predef.get("train", []), exclude_if_contains=ex)
        test_files  = resolve_files_from_patterns(ds_cfg["root"], predef.get("test", []),  exclude_if_contains=ex)

        # Fallback: if patterns didn't match, try normal listing
        if not train_files:
            train_files = list_files(ds_cfg)
        files = (train_files or []) + (test_files or [])
    else:
        files = list_files(ds_cfg)

    if not files:
        raise RuntimeError(f"No files found for {ds_name} in {ds_cfg['root']}")

    desired_label = ds_cfg["label_col"]
    mapper = get_mapper(ds_name)
    rng = np.random.RandomState(CFG["seed"])

    # planning
    stratify_on_used = None
    quotas = None
    filled = None
    rare_fine = set()
    splits = None
    protocol_b_file_support_df = None
    protocol_b_search_report = None
    protocol_b_fine_cap_quotas = None
    protocol_b_unit_infos_by_id = {}
    protocol_b_split_cap_states = {"train": defaultdict(int), "val": defaultdict(int), "test": defaultdict(int)}

    if protocol == "B_day_file":
        # Support-aware split-by-file/day. This preserves file-level shift while
        # avoiding the pathological "sorted contiguous slice" behavior that was
        # collapsing val/test support.
        if predef and train_files is not None:
            splits = {"train": train_files, "val": [], "test": (test_files or [])}
        else:
            splits, protocol_b_file_support_df, protocol_b_search_report, protocol_b_fine_cap_quotas, _, protocol_b_unit_infos_by_id = split_files_protocol_b(
                ds_name, files, ds_cfg, out_base=out_base
            )

    elif protocol == "A_stratified":
        stratify_on_used = ds_cfg.get("split_stratify_on", CFG["split_stratify_on_default"])
        if stratify_on_used not in ("family", "fine"):
            raise ValueError(f"{ds_name}: split_stratify_on must be family|fine, got {stratify_on_used}")

        if predef and train_files is not None:
            tv = ds_cfg.get("train_val_fracs", {"train": 0.85, "val": 0.15})
            tv_train = float(tv.get("train", 0.85))
            tv_val = float(tv.get("val", 0.15))

            # Only compute quotas from TRAIN files; TEST files are written fully to test split.
            stratum_counts, rare_fine = compute_stratum_counts(
                ds_name, train_files, ds_cfg,
                stratify_on=stratify_on_used,
                n_splits_required=2
            )
            quotas = make_quotas_from_counts_train_val(stratum_counts, train_frac=tv_train, val_frac=tv_val)
            filled = {"train": defaultdict(int), "val": defaultdict(int), "test": defaultdict(int)}
            splits = {"train": train_files, "val": [], "test": (test_files or [])}
        else:
            stratum_counts, rare_fine = compute_stratum_counts(ds_name, files, ds_cfg, stratify_on=stratify_on_used, n_splits_required=3)
            quotas = make_quotas_from_counts(stratum_counts)
            filled = {"train": defaultdict(int), "val": defaultdict(int), "test": defaultdict(int)}

    else:
        raise ValueError(f"Unknown protocol: {protocol}")

    used_columns = None
    dropped_leakage_cols = set()

    global_fine_counts = Counter()
    global_family_counts = Counter()

    per_split_counts = {
        "train": {"stage1": Counter(), "family": Counter(), "fine": Counter(), "rows": 0},
        "val":   {"stage1": Counter(), "family": Counter(), "fine": Counter(), "rows": 0},
        "test":  {"stage1": Counter(), "family": Counter(), "fine": Counter(), "rows": 0},
    }

    drop_families = set(ds_cfg.get("drop_families", []) or [])
    dropped_family_rows_total = 0

    cap = ds_cfg.get("target_per_fine_label")
    cap_state = defaultdict(int)

    out_dirs = {sp: os.path.join(out_base, sp) for sp in ["train", "val", "test"]}
    for d in out_dirs.values():
        safe_mkdir(d)
    part_idx = {"train": 0, "val": 0, "test": 0}

    def handle_and_write(df_in: pd.DataFrame, sp: str):
        nonlocal used_columns
        if df_in is None or len(df_in) == 0:
            return

        per_split_counts[sp]["rows"] += len(df_in)
        per_split_counts[sp]["fine"].update(df_in["y_stage2_fine"].astype(str).tolist())
        per_split_counts[sp]["family"].update(df_in["y_stage2_family"].astype(str).tolist())
        per_split_counts[sp]["stage1"].update(df_in["y_stage1_attack"].astype(int).tolist())

        global_fine_counts.update(df_in["y_stage2_fine"].astype(str).tolist())
        global_family_counts.update(df_in["y_stage2_family"].astype(str).tolist())

        if used_columns is None:
            used_columns = [c for c in df_in.columns]

        write_part(df_in, out_dirs[sp], part_idx[sp])
        part_idx[sp] += 1

    def finalize_reports():
        # per-split distributions
        for sp in ["train", "val", "test"]:
            base = os.path.join(out_base, f"class_distributions_{sp}")
            safe_mkdir(base)
            pd.DataFrame(per_split_counts[sp]["family"].most_common(), columns=["family_label", "count"]).to_csv(
                os.path.join(base, "family.csv"), index=False
            )
            pd.DataFrame(per_split_counts[sp]["fine"].most_common(), columns=["fine_label", "count"]).to_csv(
                os.path.join(base, "fine.csv"), index=False
            )
            pd.DataFrame([{"y_stage1_attack": int(k), "count": int(v)} for k, v in sorted(per_split_counts[sp]["stage1"].items())]).to_csv(
                os.path.join(base, "stage1.csv"), index=False
            )

        # explicit missing-strata report (relative to train)
        if protocol == "A_stratified":
            if stratify_on_used == "fine":
                # Compare fine labels (not merged stratum names) is less meaningful; compare family + fine counts.
                train_fine = set(per_split_counts["train"]["fine"].keys())
                val_fine = set(per_split_counts["val"]["fine"].keys())
                test_fine = set(per_split_counts["test"]["fine"].keys())
                missing_val = sorted(list(train_fine - val_fine))
                missing_test = sorted(list(train_fine - test_fine))
                pd.DataFrame({"missing_fine_label": missing_val}).to_csv(os.path.join(out_base, "MISSING_FINE_in_val.csv"), index=False)
                pd.DataFrame({"missing_fine_label": missing_test}).to_csv(os.path.join(out_base, "MISSING_FINE_in_test.csv"), index=False)

            # Always report missing families too
            train_fam = set(per_split_counts["train"]["family"].keys())
            val_fam = set(per_split_counts["val"]["family"].keys())
            test_fam = set(per_split_counts["test"]["family"].keys())
            pd.DataFrame({"missing_family": sorted(list(train_fam - val_fam))}).to_csv(os.path.join(out_base, "MISSING_FAMILY_in_val.csv"), index=False)
            pd.DataFrame({"missing_family": sorted(list(train_fam - test_fam))}).to_csv(os.path.join(out_base, "MISSING_FAMILY_in_test.csv"), index=False)

        # taxonomy mapping
        mapping_rows = []
        for lbl, cnt in global_fine_counts.items():
            mapping_rows.append({"fine_label": lbl, "family_label": mapper(lbl), "count": int(cnt)})
        pd.DataFrame(mapping_rows).sort_values("count", ascending=False).to_csv(
            os.path.join(out_base, "taxonomy_mapping.csv"), index=False
        )

        with open(os.path.join(out_base, "USED_COLUMNS.json"), "w", encoding="utf-8") as f:
            json.dump({"columns": used_columns or []}, f, indent=2)

        with open(os.path.join(out_base, "SPLIT_PROTOCOL.json"), "w", encoding="utf-8") as f:
            json.dump({
                "protocol": protocol,
                "split_fracs": CFG["split_fracs"],
                "stratify_on": stratify_on_used,
                "min_per_split_per_stratum": CFG.get("min_per_split_per_stratum"),
                "rare_fine_min_total_effective": max(int(ds_cfg.get("rare_fine_min_total", CFG.get("rare_fine_min_total_default", 0)) or 0),
                                                     3 * int(CFG.get("min_per_split_per_stratum", 1))),
                "rare_fine_labels_merged": len(rare_fine) if rare_fine else 0,
                "protocol_b_split_mode": ds_cfg.get("protocol_b_split_mode", CFG.get("protocol_b_split_mode", "support_aware")) if protocol == "B_day_file" else None,
                "protocol_b_partition_mode": ds_cfg.get("protocol_b_partition_mode", "whole_file") if protocol == "B_day_file" else None,
                "protocol_b_subfiles_per_file": ds_cfg.get("protocol_b_subfiles_per_file") if protocol == "B_day_file" else None,
                "protocol_b_selected_holdouts": [x.get("holdout_family") for x in ((protocol_b_search_report or {}).get("best_eval", {}) or {}).get("eligible_holdouts", [])] if protocol == "B_day_file" else [],
            }, f, indent=2)

        with open(os.path.join(out_base, "DROPPED_LEAKAGE_COLS.json"), "w", encoding="utf-8") as f:
            json.dump({"dropped_cols": sorted(list(dropped_leakage_cols))}, f, indent=2)

        rep = {"dataset": ds_name, "protocol": protocol, "drop_families": sorted(list(drop_families)), "dropped_family_rows_total": int(dropped_family_rows_total), "splits": {}}
        for sp in ["train", "val", "test"]:
            rep["splits"][sp] = {
                "rows": int(per_split_counts[sp]["rows"]),
                "stage1_counts": {str(k): int(v) for k, v in per_split_counts[sp]["stage1"].items()},
                "family_counts": dict(per_split_counts[sp]["family"]),
                "fine_top": per_split_counts[sp]["fine"].most_common(25),
            }

        # dataset signature (helps TrainEval validate it's reading the intended processed folder)
        try:
            cols_for_hash = json.dumps({"columns": used_columns or []}, sort_keys=True).encode("utf-8")
            cols_sha1 = hashlib.sha1(cols_for_hash).hexdigest()
        except Exception:
            cols_sha1 = None

        signature = {
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "dataset": ds_name,
            "protocol": protocol,
            "out_base": out_base,
            "seed": CFG.get("seed"),
            "stratify_on": stratify_on_used,
            "columns_sha1": cols_sha1,
            "rows": {sp: int(per_split_counts[sp]["rows"]) for sp in ["train","val","test"]},
            "num_parts": {sp: int(part_idx[sp]) for sp in ["train","val","test"]},
            "drop_families": sorted(list(drop_families)),
            "dropped_family_rows_total": int(dropped_family_rows_total),
        }
        with open(os.path.join(out_base, "DATASET_SIGNATURE.json"), "w", encoding="utf-8") as f:
            json.dump(signature, f, indent=2)

        with open(os.path.join(out_base, "SPLIT_REPORT.json"), "w", encoding="utf-8") as f:
            json.dump(rep, f, indent=2)

    try:
        if protocol == "B_day_file":
            # process split-by-file or split-by-contiguous-protocol-B-units
            for sp in ["train", "val", "test"]:
                for unit_id in splits[sp]:
                    info = protocol_b_unit_infos_by_id.get(unit_id, {})
                    source_path = str(info.get("source_file_path", unit_id))
                    row_start = info.get("row_start")
                    row_end = info.get("row_end")

                    label_actual = resolve_label_col_for_file(source_path, desired_label, ds_cfg)
                    if label_actual is None:
                        continue
                    label_col = canonical_col(label_actual)

                    if row_start is None or row_end is None or (int(row_start) == 0 and int(row_end) == int(info.get("raw_row_count", row_end)) and str(info.get("partition_mode", "whole_file")) == "whole_file"):
                        readers = iter_read_csv(source_path, ds_cfg, usecols=None)
                    else:
                        readers = [
                            read_csv_slice(
                                source_path,
                                ds_cfg,
                                start_row=int(row_start),
                                nrows=int(row_end) - int(row_start),
                                usecols=None,
                            )
                        ]

                    for chunk in readers:
                        chunk = canonicalize_columns(chunk)
                        if label_col not in chunk.columns:
                            raise RuntimeError(f"{ds_name}: label column '{desired_label}' not found in {source_path}")

                        if CFG["drop_leakage_cols"]:
                            leak = detect_leakage_cols(list(chunk.columns))
                            if CFG["bucket_ports"]:
                                for raw in ("src_port", "dst_port", "source_port", "destination_port", "sport", "dsport", "srcport", "dstport"):
                                    if raw in chunk.columns:
                                        chunk[f"{raw}_bucket"] = port_bucket(chunk[raw])
                                leak = [c for c in leak if not c.endswith("_bucket")]
                            for c in leak:
                                if c in chunk.columns:
                                    dropped_leakage_cols.add(c)
                                    chunk = chunk.drop(columns=[c])

                        chunk = chunk.replace([np.inf, -np.inf], np.nan)

                        fine = chunk[label_col].astype(str).str.strip()
                        if ds_cfg.get("strip_label_period", False):
                            fine = fine.str.replace(".", "", regex=False)
                        family = fine.map(mapper).astype(str)
                        stage1 = (family != "Benign").astype(np.int8)

                        # Option A: drop specified families BEFORE split outputs
                        nonlocal_dropped = 0
                        if drop_families:
                            keep = ~family.isin(list(drop_families))
                            nonlocal_dropped = int((~keep).sum())
                            if nonlocal_dropped > 0:
                                chunk = chunk.loc[keep].copy()
                                fine = fine.loc[keep]
                                family = family.loc[keep]
                        if len(chunk) == 0:
                            dropped_family_rows_total += nonlocal_dropped
                            continue
                        dropped_family_rows_total += nonlocal_dropped
                        stage1 = (family != "Benign").astype(np.int8)

                        chunk["y_stage1_attack"] = stage1
                        chunk["y_stage2_family"] = family
                        chunk["y_stage2_fine"] = fine

                        # drop raw label + drop_cols
                        drop_cols = set([label_col])
                        for dc in ds_cfg.get("drop_cols", []):
                            dcc = canonical_col(dc)
                            if dcc in chunk.columns:
                                drop_cols.add(dcc)
                        chunk = chunk.drop(columns=[c for c in drop_cols if c in chunk.columns])

                        # optional cap
                        if cap is not None:
                            if protocol == "B_day_file" and protocol_b_fine_cap_quotas is not None:
                                chunk = cap_by_fine_label_with_quotas(
                                    chunk,
                                    "y_stage2_fine",
                                    split_quota=protocol_b_fine_cap_quotas.get(sp, {}),
                                    cap_state=protocol_b_split_cap_states[sp],
                                    rng=rng,
                                )
                            else:
                                chunk = cap_by_fine_label(chunk, "y_stage2_fine", cap=int(cap), cap_state=cap_state, rng=rng)
                            if len(chunk) == 0:
                                continue

                        handle_and_write(chunk, sp)

        else:
            # protocol A: row-wise stratified split
            for f in (train_files if (predef and train_files is not None) else files):
                label_actual = resolve_label_col_for_file(f, desired_label, ds_cfg)
                if label_actual is None:
                    continue
                label_col = canonical_col(label_actual)

                for chunk in iter_read_csv(f, ds_cfg, usecols=None):
                    chunk = canonicalize_columns(chunk)
                    if label_col not in chunk.columns:
                        raise RuntimeError(f"{ds_name}: label column '{desired_label}' not found in {f}")

                    if CFG["drop_leakage_cols"]:
                        leak = detect_leakage_cols(list(chunk.columns))
                        if CFG["bucket_ports"]:
                            for raw in ("src_port", "dst_port", "source_port", "destination_port", "sport", "dsport", "srcport", "dstport"):
                                if raw in chunk.columns:
                                    chunk[f"{raw}_bucket"] = port_bucket(chunk[raw])
                            leak = [c for c in leak if not c.endswith("_bucket")]
                        for c in leak:
                            if c in chunk.columns:
                                dropped_leakage_cols.add(c)
                                chunk = chunk.drop(columns=[c])

                    chunk = chunk.replace([np.inf, -np.inf], np.nan)

                    fine = chunk[label_col].astype(str).str.strip()
                    if ds_cfg.get("strip_label_period", False):
                        fine = fine.str.replace(".", "", regex=False)
                    family = fine.map(mapper).astype(str)

                    # Option A: drop specified families BEFORE quotas/splitting
                    n_drop = 0
                    if drop_families:
                        keep = ~family.isin(list(drop_families))
                        n_drop = int((~keep).sum())
                        if n_drop > 0:
                            chunk = chunk.loc[keep].copy()
                            fine = fine.loc[keep]
                            family = family.loc[keep]
                    if len(chunk) == 0:
                        dropped_family_rows_total += n_drop
                        continue
                    dropped_family_rows_total += n_drop

                    stage1 = (family != "Benign").astype(np.int8)

                    chunk["y_stage1_attack"] = stage1
                    chunk["y_stage2_family"] = family
                    chunk["y_stage2_fine"] = fine

                    # drop raw label + drop_cols
                    drop_cols = set([label_col])
                    for dc in ds_cfg.get("drop_cols", []):
                        dcc = canonical_col(dc)
                        if dcc in chunk.columns:
                            drop_cols.add(dcc)
                    chunk = chunk.drop(columns=[c for c in drop_cols if c in chunk.columns])

                    # optional cap
                    if cap is not None:
                        chunk = cap_by_fine_label(chunk, "y_stage2_fine", cap=int(cap), cap_state=cap_state, rng=rng)
                        if len(chunk) == 0:
                            continue

                    # Choose stratification key
                    if stratify_on_used == "fine":
                        if rare_fine:
                            is_rare = chunk["y_stage2_fine"].astype(str).isin(rare_fine)
                            chunk["_split_stratum"] = np.where(
                                is_rare,
                                "__RARE__" + chunk["y_stage2_family"].astype(str),
                                chunk["y_stage2_fine"].astype(str)
                            )
                        else:
                            chunk["_split_stratum"] = chunk["y_stage2_fine"].astype(str)
                        strat_col = "_split_stratum"
                    else:
                        strat_col = "y_stage2_family"

                    sub = split_chunk_by_quotas(chunk, strat_col, quotas, filled, rng)

                    for sp, df_sp in sub.items():
                        if "_split_stratum" in df_sp.columns:
                            df_sp = df_sp.drop(columns=["_split_stratum"])
                        handle_and_write(df_sp, sp)

            # If dataset provides predefined test files, write them fully into the test split (no stratified sampling).
            if predef and test_files:
                for f in test_files:
                    label_actual = resolve_label_col_for_file(f, desired_label, ds_cfg)
                    if label_actual is None:
                        continue
                    label_col = canonical_col(label_actual)

                    for chunk in iter_read_csv(f, ds_cfg, usecols=None):
                        chunk = canonicalize_columns(chunk)
                        if label_col not in chunk.columns:
                            raise RuntimeError(f"{ds_name}: label column '{desired_label}' not found in {f}")

                        if CFG["drop_leakage_cols"]:
                            leak = detect_leakage_cols(list(chunk.columns))
                            if CFG["bucket_ports"]:
                                for raw in ("src_port", "dst_port", "source_port", "destination_port", "sport", "dsport", "srcport", "dstport"):
                                    if raw in chunk.columns:
                                        chunk[f"{raw}_bucket"] = port_bucket(chunk[raw])
                                leak = [c for c in leak if not c.endswith("_bucket")]
                            for c in leak:
                                if c in chunk.columns:
                                    dropped_leakage_cols.add(c)
                                    chunk = chunk.drop(columns=[c])

                        chunk = chunk.replace([np.inf, -np.inf], np.nan)

                        fine = chunk[label_col].astype(str).str.strip()
                        if ds_cfg.get("strip_label_period", False):
                            fine = fine.str.replace(".", "", regex=False)
                        family = fine.map(mapper).astype(str)

                        nonlocal_dropped = 0
                        if drop_families:
                            keep = ~family.isin(list(drop_families))
                            nonlocal_dropped = int((~keep).sum())
                            if nonlocal_dropped > 0:
                                chunk = chunk.loc[keep].copy()
                                fine = fine.loc[keep]
                                family = family.loc[keep]
                        if len(chunk) == 0:
                            dropped_family_rows_total += nonlocal_dropped
                            continue
                        dropped_family_rows_total += nonlocal_dropped
                        stage1 = (family != "Benign").astype(np.int8)

                        chunk["y_stage1_attack"] = stage1
                        chunk["y_stage2_family"] = family
                        chunk["y_stage2_fine"] = fine

                        drop_cols = set([label_col])
                        for dc in ds_cfg.get("drop_cols", []):
                            dcc = canonical_col(dc)
                            if dcc in chunk.columns:
                                drop_cols.add(dcc)
                        chunk = chunk.drop(columns=[c for c in drop_cols if c in chunk.columns])

                        if cap is not None:
                            chunk = cap_by_fine_label(chunk, "y_stage2_fine", cap=int(cap), cap_state=cap_state, rng=rng)
                            if len(chunk) == 0:
                                continue

                        handle_and_write(chunk, "test")
    finally:
        # Always write diagnostics, even if a crash happens mid-run
        finalize_reports()

    # Hard sanity checks for Protocol A
    if protocol == "A_stratified":
        for sp in ["val", "test"]:
            s1 = per_split_counts[sp]["stage1"]
            if len(s1) < 2:
                print(f"[WARN] {ds_name} {sp}: Stage1 has only one class: {dict(s1)}")
        # If val/test have 0 rows, that's a hard failure
        for sp in ["train", "val", "test"]:
            if per_split_counts[sp]["rows"] == 0:
                raise RuntimeError(f"{ds_name}: split '{sp}' produced 0 rows. Check file_glob/excludes and label_col.")

    print(f"[OK] {ds_name} -> {out_base}")


def main():
    safe_mkdir(CFG["out_root"])
    active = set(CFG.get("active_datasets") or CFG["datasets"].keys())
    for ds_name, ds_cfg in CFG["datasets"].items():
        if ds_name not in active:
            continue
        process_dataset(ds_name, ds_cfg)

if __name__ == "__main__":
    main()
