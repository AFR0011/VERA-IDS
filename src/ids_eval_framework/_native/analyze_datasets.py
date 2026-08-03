# analyze_ids_datasets.py
# Requirements: pandas, numpy, matplotlib
# Install: pip install pandas numpy matplotlib

import os
import re
import json
import glob
import hashlib
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ids_eval_framework.src.paths import repo_path


# =========================
# CONFIG (EDIT THIS)
# =========================
CONFIG = {
    "out_dir": "IDS_dataset_analysis_out",

    # Set these to your local folders (as in your screenshots)
    "datasets": {
        "CICIoT2023": {
            "root": repo_path("Datasets", "CIC IoT Dataset 2023"),
            "features_csv": repo_path("Datasets", "CIC IoT Dataset 2023", "CIC IoT Dataset 2023_features.csv"),
        },
        "CICIDS2017": {
            "root": repo_path("Datasets", "CICIDS 2017"),
            "features_csv": repo_path("Datasets", "CICIDS 2017", "CICIDS 2017_features.csv"),
        },
        # Optional stress tests (set if you have them locally)
        # "UNSW-NB15": {"root": r"...", "features_csv": None},
        # "NSL-KDD": {"root": r"...", "features_csv": None},
    },

    # Analysis modes:
    # - quick_scan: samples rows for numeric stats/missingness (fast)
    # - full_scan: chunk-streams all files for numeric stats/missingness (slower but complete)
    "quick_scan": False,

    # Chunk size for streaming passes
    "chunksize": 200_000,

    # Sampling for quick scan: max rows total across all files (approx)
    "quick_max_rows": 300_000,

    # Files to include: default includes all *.csv except anything containing "features"
    "include_pattern": "*.csv",
    "exclude_if_name_contains": ["features"],

    # Plot limits
    "top_k_labels_plot": 30,
    "top_k_missing_plot": 30,
    "top_k_leakage_plot": 20,
}


# =========================
# Utility: string/column helpers
# =========================
def safe_mkdir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def canonical_col(name: str) -> str:
    # Normalize: lower, strip, replace non-alphanum with underscore, collapse underscores
    s = str(name).strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s

def canonicalize_columns(cols: List[str]) -> List[str]:
    return [canonical_col(c) for c in cols]

def schema_hash(cols: List[str]) -> str:
    joined = "|".join(cols).encode("utf-8")
    return hashlib.md5(joined).hexdigest()[:12]

def is_probably_time_col(c: str) -> bool:
    c0 = canonical_col(c)
    return any(k in c0 for k in ["time", "timestamp", "date"])

def leakage_name_score(c: str) -> int:
    """
    Higher score => more likely leakage/identifier.
    """
    c0 = canonical_col(c)
    score = 0
    keywords = [
        ("flow_id", 5), ("id", 3),
        ("src_ip", 5), ("dst_ip", 5), ("ip", 4), ("address", 3), ("mac", 4),
        ("src_port", 4), ("dst_port", 4), ("port", 3),
        ("timestamp", 5), ("time", 3), ("date", 3),
        ("local", 2),  # CICIDS has Local_*, often derived from addresses/identifiers
    ]
    for k, w in keywords:
        if k in c0:
            score += w
    return score


# =========================
# Reading feature list (optional)
# =========================
def load_features_csv(path: Optional[str]) -> Optional[pd.DataFrame]:
    if not path:
        return None
    if not os.path.exists(path):
        print(f"[WARN] features_csv not found: {path}")
        return None
    df = pd.read_csv(path)
    # Expect columns: No., Name, Type, Description
    # Keep a canonical column name for join/matching
    df["Name_canon"] = df["Name"].astype(str).map(canonical_col)
    df["Type"] = df["Type"].astype(str).str.lower()
    return df


# =========================
# Core analysis
# =========================
@dataclass
class DatasetProfile:
    name: str
    root: str
    files: List[str]
    features_df: Optional[pd.DataFrame]

def list_csv_files(root: str) -> List[str]:
    files = sorted(glob.glob(os.path.join(root, CONFIG["include_pattern"])))
    out = []
    for f in files:
        base = os.path.basename(f).lower()
        if any(x in base for x in CONFIG["exclude_if_name_contains"]):
            continue
        out.append(f)
    return out

def read_header_cols(csv_path: str) -> List[str]:
    # Only read header
    df0 = pd.read_csv(csv_path, nrows=0)
    return list(df0.columns)

def infer_label_candidates(all_cols: List[str], features_df: Optional[pd.DataFrame]) -> List[str]:
    """
    Find candidate label columns by name patterns and features list hints.
    Returns canonical names.
    """
    canon_cols = canonicalize_columns(all_cols)
    candidates = set()

    # Name-based patterns
    for c in canon_cols:
        if c in ["label", "class", "target", "y"]:
            candidates.add(c)
        if "label" in c or "attack" in c or "category" in c or "class" in c:
            candidates.add(c)

    # Features list hints (if present)
    if features_df is not None:
        for _, r in features_df.iterrows():
            nm = str(r.get("Name_canon", ""))
            if nm in ["label", "attempted_category"] or "label" in nm or "category" in nm:
                candidates.add(nm)

    # Keep only those that actually exist in columns
    existing = set(canon_cols)
    return sorted([c for c in candidates if c in existing])

def infer_numeric_cols_sample(sample_df: pd.DataFrame, features_df: Optional[pd.DataFrame]) -> List[str]:
    """
    Decide numeric columns.
    If features list exists, trust it (Type float/integer).
    Otherwise, infer by coercion success rate on a sample.
    """
    cols = list(sample_df.columns)
    if features_df is not None:
        type_map = dict(zip(features_df["Name_canon"], features_df["Type"]))
        numeric = []
        for c in cols:
            t = type_map.get(canonical_col(c), "")
            if t in ["float", "integer", "int"]:
                numeric.append(c)
        return numeric

    numeric = []
    for c in cols:
        s = pd.to_numeric(sample_df[c], errors="coerce")
        ok = s.notna().mean()
        if ok >= 0.90:
            numeric.append(c)
    return numeric

def sample_rows(files: List[str], max_rows: int) -> pd.DataFrame:
    """
    Sample up to max_rows total by reading small chunks from each file.
    """
    pieces = []
    remaining = max_rows
    if not files:
        return pd.DataFrame()

    per_file = max(1_000, max_rows // min(len(files), 50))
    for f in files[:50]:
        if remaining <= 0:
            break
        n = min(per_file, remaining)
        try:
            df = pd.read_csv(f, nrows=n)
            df.columns = canonicalize_columns(df.columns)
            pieces.append(df)
            remaining -= len(df)
        except Exception as e:
            print(f"[WARN] Failed sampling {f}: {e}")
            continue

    if not pieces:
        return pd.DataFrame()

    return pd.concat(pieces, ignore_index=True)

def streaming_label_counts(files: List[str], label_cols: List[str], chunksize: int) -> Tuple[int, Dict[str, Counter]]:
    """
    Stream through all files reading ONLY label columns (cheap) and count distributions.
    Returns total_rows and dict[label_col] -> Counter(value).
    """
    total = 0
    counters = {lc: Counter() for lc in label_cols}

    for f in files:
        # discover original names that match canonical label cols
        try:
            hdr = read_header_cols(f)
        except Exception as e:
            print(f"[WARN] Failed reading header for {f}: {e}")
            continue

        hdr_canon = canonicalize_columns(hdr)
        canon_to_orig = {c: o for c, o in zip(hdr_canon, hdr)}

        usecols = []
        for lc in label_cols:
            if lc in canon_to_orig:
                usecols.append(canon_to_orig[lc])

        if not usecols:
            continue

        try:
            for chunk in pd.read_csv(f, usecols=usecols, chunksize=chunksize, dtype=str, low_memory=True):
                total += len(chunk)
                chunk.columns = canonicalize_columns(chunk.columns)
                for lc in label_cols:
                    if lc in chunk.columns:
                        vals = chunk[lc].fillna("<<NA>>").astype(str).str.strip()
                        counters[lc].update(vals.tolist())
        except Exception as e:
            print(f"[WARN] Failed label scan for {f}: {e}")
            continue

    return total, counters

def streaming_missing_and_numeric_stats(
    files: List[str],
    numeric_cols: List[str],
    chunksize: int,
    full_scan: bool,
    quick_max_rows: int
) -> Tuple[int, Dict[str, int], Dict[str, Dict[str, float]]]:
    """
    Compute:
      - total rows
      - missing counts for all columns
      - numeric stats: count, mean, std, min, max (streaming)
    If full_scan=False, only analyze up to quick_max_rows rows total (sampled).
    """
    total = 0
    miss = defaultdict(int)

    # numeric accumulators
    # sum, sumsq, count, min, max
    acc = {c: {"count": 0, "sum": 0.0, "sumsq": 0.0, "min": np.inf, "max": -np.inf} for c in numeric_cols}

    rows_budget = quick_max_rows if not full_scan else None

    for f in files:
        if rows_budget is not None and rows_budget <= 0:
            break

        try:
            for chunk in pd.read_csv(f, chunksize=chunksize, low_memory=True):
                chunk.columns = canonicalize_columns(chunk.columns)

                # quick mode budget handling
                if rows_budget is not None and len(chunk) > rows_budget:
                    chunk = chunk.iloc[:rows_budget].copy()

                n = len(chunk)
                total += n
                if rows_budget is not None:
                    rows_budget -= n

                # missingness for all columns
                for c in chunk.columns:
                    miss[c] += int(chunk[c].isna().sum())

                # numeric stats
                for c in numeric_cols:
                    if c not in chunk.columns:
                        continue
                    s = pd.to_numeric(chunk[c], errors="coerce")
                    # treat inf as missing
                    s = s.replace([np.inf, -np.inf], np.nan)
                    nn = s.notna().sum()
                    if nn == 0:
                        continue
                    vals = s.dropna().to_numpy(dtype=float)
                    acc[c]["count"] += int(nn)
                    acc[c]["sum"] += float(vals.sum())
                    acc[c]["sumsq"] += float((vals * vals).sum())
                    acc[c]["min"] = float(min(acc[c]["min"], vals.min()))
                    acc[c]["max"] = float(max(acc[c]["max"], vals.max()))

                if rows_budget is not None and rows_budget <= 0:
                    break

        except Exception as e:
            print(f"[WARN] Failed stats scan for {f}: {e}")
            continue

    # finalize numeric stats
    stats = {}
    for c, a in acc.items():
        cnt = a["count"]
        if cnt <= 1:
            continue
        mean = a["sum"] / cnt
        var = max(0.0, (a["sumsq"] / cnt) - mean * mean)
        std = float(np.sqrt(var))
        stats[c] = {
            "count": int(cnt),
            "mean": float(mean),
            "std": float(std),
            "min": float(a["min"]) if np.isfinite(a["min"]) else None,
            "max": float(a["max"]) if np.isfinite(a["max"]) else None,
        }

    return total, dict(miss), stats

def analyze_schema_groups(files: List[str]) -> Dict[str, dict]:
    groups = {}
    for f in files:
        try:
            cols = read_header_cols(f)
            cols_c = canonicalize_columns(cols)
            h = schema_hash(cols_c)
            if h not in groups:
                groups[h] = {
                    "n_files": 0,
                    "example_file": f,
                    "columns": cols_c,
                }
            groups[h]["n_files"] += 1
        except Exception as e:
            print(f"[WARN] Failed schema read for {f}: {e}")
            continue
    return groups

def plot_bar(counter: Counter, title: str, out_path: str, top_k: int = 30) -> None:
    items = counter.most_common(top_k)
    if not items:
        return
    labels, values = zip(*items)
    plt.figure(figsize=(12, 5))
    plt.bar(range(len(values)), values)
    plt.xticks(range(len(values)), labels, rotation=60, ha="right")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def plot_missingness(miss_counts: Dict[str, int], total_rows: int, title: str, out_path: str, top_k: int = 30) -> None:
    if total_rows <= 0 or not miss_counts:
        return
    miss_rate = [(c, v / total_rows) for c, v in miss_counts.items()]
    miss_rate.sort(key=lambda x: x[1], reverse=True)
    miss_rate = miss_rate[:top_k]
    cols = [x[0] for x in miss_rate]
    vals = [x[1] for x in miss_rate]
    plt.figure(figsize=(12, 5))
    plt.bar(range(len(vals)), vals)
    plt.xticks(range(len(vals)), cols, rotation=60, ha="right")
    plt.title(title)
    plt.ylabel("Missing rate")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def write_markdown_summary(
    out_path: str,
    ds_name: str,
    root: str,
    n_files: int,
    total_rows: int,
    schema_groups: Dict[str, dict],
    label_cols: List[str],
    label_counters: Dict[str, Counter],
    leakage_cols: List[Tuple[str, int]],
    time_cols: List[str],
    numeric_stats_count: int,
    quick_scan: bool
) -> None:
    lines = []
    lines.append(f"# Dataset Analysis Summary: {ds_name}\n")
    lines.append(f"- Root: `{root}`")
    lines.append(f"- Files analyzed: **{n_files}**")
    lines.append(f"- Total rows counted (label pass): **{total_rows}**")
    lines.append(f"- Scan mode for numeric/missingness: **{'QUICK (sampled)' if quick_scan else 'FULL (all rows)'}**\n")

    lines.append("## Schema groups (important if you have `*_plus.csv` variants)\n")
    lines.append(f"- Unique schemas found: **{len(schema_groups)}**")
    for h, g in sorted(schema_groups.items(), key=lambda x: x[1]["n_files"], reverse=True)[:5]:
        lines.append(f"  - `{h}`: {g['n_files']} files (example: `{os.path.basename(g['example_file'])}`)")

    lines.append("\n## Candidate label columns\n")
    if not label_cols:
        lines.append("- None detected. (Check config / column names.)")
    else:
        for lc in label_cols:
            cnt = label_counters.get(lc, Counter())
            lines.append(f"- `{lc}`: **{len(cnt)}** unique values")
            top5 = cnt.most_common(5)
            if top5:
                top_str = ", ".join([f"{k} ({v})" for k, v in top5])
                lines.append(f"  - Top values: {top_str}")

    lines.append("\n## Time columns (for drift / pseudo-online splits)\n")
    if not time_cols:
        lines.append("- None detected by name. If CICIDS is split by day files, you can still do day-ordered evaluation.")
    else:
        lines.append("- Detected: " + ", ".join([f"`{c}`" for c in time_cols]))

    lines.append("\n## Leakage / identifier candidates (drop or bucket these)\n")
    if not leakage_cols:
        lines.append("- None detected by name patterns (still manually verify).")
    else:
        for c, s in leakage_cols[:20]:
            lines.append(f"- `{c}` (risk score={s})")

    lines.append("\n## Numeric stats\n")
    lines.append(f"- Numeric columns with stats computed: **{numeric_stats_count}**")

    lines.append("\n## Decision helper (Stage-2 label choice)\n")
    lines.append("- If you have both `label` and something like `attempted_category`:")
    lines.append("  - Prefer the one with **fewer unique values** (coarser taxonomy) *if* it’s meaningful (not mostly `-1`).")
    lines.append("  - Otherwise use fine-grained label and build your own mapping to 5–8 families.\n")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

def analyze_one_dataset(profile: DatasetProfile, out_root: str) -> None:
    ds_out = os.path.join(out_root, profile.name)
    safe_mkdir(ds_out)
    safe_mkdir(os.path.join(ds_out, "figures"))

    print(f"\n=== Analyzing {profile.name} ===")
    if not profile.files:
        print(f"[WARN] No CSV files found in: {profile.root}")
        return

    # Schema groups
    schema_groups = analyze_schema_groups(profile.files)
    with open(os.path.join(ds_out, "schema_groups.json"), "w", encoding="utf-8") as f:
        json.dump(schema_groups, f, indent=2)

    # Use the most common schema to define columns for analysis
    dominant_schema = max(schema_groups.items(), key=lambda x: x[1]["n_files"])[1]["columns"]

    # Label candidates (canonical names)
    label_cols = infer_label_candidates(dominant_schema, profile.features_df)

    # Always do exact label counting (streamed)
    total_rows, label_counters = streaming_label_counts(profile.files, label_cols, CONFIG["chunksize"])

    # Save label distributions
    label_dir = os.path.join(ds_out, "labels")
    safe_mkdir(label_dir)
    for lc, ctr in label_counters.items():
        df = pd.DataFrame(ctr.most_common(), columns=[lc, "count"])
        df.to_csv(os.path.join(label_dir, f"{lc}_distribution.csv"), index=False)

        plot_bar(
            ctr,
            title=f"{profile.name} - {lc} distribution (top {CONFIG['top_k_labels_plot']})",
            out_path=os.path.join(ds_out, "figures", f"{lc}_top.png"),
            top_k=CONFIG["top_k_labels_plot"]
        )

    # Time columns (for drift)
    time_cols = [c for c in dominant_schema if is_probably_time_col(c)]

    # Leakage candidates by name score
    leak_scored = [(c, leakage_name_score(c)) for c in dominant_schema]
    leak_scored = sorted([x for x in leak_scored if x[1] >= 3], key=lambda x: x[1], reverse=True)

    pd.DataFrame(leak_scored, columns=["column", "risk_score"]).to_csv(
        os.path.join(ds_out, "leakage_candidates.csv"), index=False
    )

    # Sample for dtype inference / numeric list
    sample_df = sample_rows(profile.files, CONFIG["quick_max_rows"])
    if sample_df.empty:
        numeric_cols = []
    else:
        numeric_cols = infer_numeric_cols_sample(sample_df, profile.features_df)
        # canonicalize, because sample_rows canonicalized them
        numeric_cols = [canonical_col(c) for c in numeric_cols]

    # Missingness + numeric stats (quick or full)
    full_scan = not CONFIG["quick_scan"]
    total_stats_rows, miss_counts, numeric_stats = streaming_missing_and_numeric_stats(
        files=profile.files,
        numeric_cols=numeric_cols,
        chunksize=CONFIG["chunksize"],
        full_scan=full_scan,
        quick_max_rows=CONFIG["quick_max_rows"]
    )

    # Save missingness + numeric stats
    pd.DataFrame(
        [{"column": c, "missing_count": v, "missing_rate": (v / max(1, total_stats_rows))} for c, v in miss_counts.items()]
    ).sort_values("missing_rate", ascending=False).to_csv(os.path.join(ds_out, "missingness.csv"), index=False)

    pd.DataFrame(
        [{"column": c, **st} for c, st in numeric_stats.items()]
    ).to_csv(os.path.join(ds_out, "numeric_stats.csv"), index=False)

    plot_missingness(
        miss_counts,
        total_stats_rows,
        title=f"{profile.name} - missingness (top {CONFIG['top_k_missing_plot']})",
        out_path=os.path.join(ds_out, "figures", "missingness_top.png"),
        top_k=CONFIG["top_k_missing_plot"]
    )

    # Markdown summary
    write_markdown_summary(
        out_path=os.path.join(ds_out, "SUMMARY.md"),
        ds_name=profile.name,
        root=profile.root,
        n_files=len(profile.files),
        total_rows=total_rows,
        schema_groups=schema_groups,
        label_cols=label_cols,
        label_counters=label_counters,
        leakage_cols=leak_scored,
        time_cols=time_cols,
        numeric_stats_count=len(numeric_stats),
        quick_scan=CONFIG["quick_scan"],
    )

    # Extra: “Stage-2 label recommendation” heuristics
    rec = {"recommended_stage2_col": None, "reason": None}
    if len(label_cols) >= 2:
        # choose the label-like column with fewer unique values, unless it's mostly -1 / NA
        scored = []
        for lc in label_cols:
            ctr = label_counters.get(lc, Counter())
            uniq = len(ctr)
            total_l = sum(ctr.values())
            frac_bad = 0.0
            if total_l > 0:
                bad = ctr.get("-1", 0) + ctr.get("<<NA>>", 0) + ctr.get("nan", 0) + ctr.get("None", 0)
                frac_bad = bad / total_l
            scored.append((lc, uniq, frac_bad))
        scored.sort(key=lambda x: (x[1], x[2]))  # fewer unique, less garbage
        best = scored[0]
        if best[2] < 0.80:
            rec["recommended_stage2_col"] = best[0]
            rec["reason"] = f"Fewer unique values ({best[1]}) with manageable garbage fraction ({best[2]:.2f})."
        else:
            rec["recommended_stage2_col"] = scored[-1][0]
            rec["reason"] = "The coarse column looks mostly garbage (-1/NA). Use the other and build your own mapping."
    elif len(label_cols) == 1:
        rec["recommended_stage2_col"] = label_cols[0]
        rec["reason"] = "Only one label-like column detected."

    with open(os.path.join(ds_out, "stage2_label_recommendation.json"), "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=2)

    print(f"[OK] Wrote report to: {ds_out}")


def main():
    out_root = CONFIG["out_dir"]
    safe_mkdir(out_root)

    profiles = []
    for ds_name, ds_cfg in CONFIG["datasets"].items():
        root = ds_cfg["root"]
        files = list_csv_files(root)
        features_df = load_features_csv(ds_cfg.get("features_csv"))
        profiles.append(DatasetProfile(name=ds_name, root=root, files=files, features_df=features_df))

    for p in profiles:
        analyze_one_dataset(p, out_root)

    # Cross-dataset overlap (canonical feature names based on dominant schema)
    # Helpful to see whether you can cross-train (usually: no).
    overlap_out = os.path.join(out_root, "cross_dataset")
    safe_mkdir(overlap_out)

    dom_cols = {}
    for p in profiles:
        if not p.files:
            continue
        groups = analyze_schema_groups(p.files)
        dominant_schema = max(groups.items(), key=lambda x: x[1]["n_files"])[1]["columns"]
        dom_cols[p.name] = set(dominant_schema)

    names = list(dom_cols.keys())
    rows = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            inter = dom_cols[a] & dom_cols[b]
            rows.append({
                "A": a,
                "B": b,
                "A_cols": len(dom_cols[a]),
                "B_cols": len(dom_cols[b]),
                "overlap_cols": len(inter),
                "overlap_examples": ", ".join(sorted(list(inter))[:20])
            })
    pd.DataFrame(rows).to_csv(os.path.join(overlap_out, "feature_overlap.csv"), index=False)

    print(f"\n[OK] All done. See: {os.path.abspath(out_root)}")


if __name__ == "__main__":
    main()
