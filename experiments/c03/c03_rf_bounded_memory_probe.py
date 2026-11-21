#!/usr/bin/env python3
"""C03 engineering-only RF RAM probe. Never uses validation/test or creates models for scientific use.

Usage: python -B c03_rf_bounded_memory_probe.py --inspect
       python -B c03_rf_bounded_memory_probe.py --probe

The default probe runs a child process using training rows ONLY, deliberately
limited to 8 trees on 30k and 80k sampled known-attack rows. It is NOT a C03 fit.
"""
from __future__ import annotations

import argparse
import ctypes
import gc
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

GiB = 1024**3
MiB = 1024**2
EXPECTED_HEAD = "ca13f1097f851ce6693658ccc0c6dd35480fabd6"
DATA_DIR = Path("processed_V5/B_day_file/CICIoT2023/train")
META_FILE = Path("outputs/13_jisa_q1_revision/phase3_validation_blind/score_generation/seed_123/Botnet/preprocessor_meta.json")
KNOWN = ("BruteForce", "DDoS", "DoS", "Other", "Scan/Recon")


def os_memory():
    try:
        import psutil
        v = psutil.virtual_memory()
        s = psutil.swap_memory()
        return {"physical_total": int(v.total), "physical_available": int(v.available),
                "swap_total": int(s.total), "swap_used": int(s.used)}
    except ImportError:
        if sys.platform != "win32":
            raise RuntimeError("Memory monitor needs psutil outside Windows")
        class MemoryStatus(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        st = MemoryStatus()
        st.dwLength = ctypes.sizeof(st)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
            raise RuntimeError("GlobalMemoryStatusEx failed")
        return {"physical_total": st.ullTotalPhys, "physical_available": st.ullAvailPhys,
                "commit_limit": st.ullTotalPageFile, "commit_available": st.ullAvailPageFile}


def process_rss(pid):
    try:
        import psutil
        return psutil.Process(pid).memory_info().rss
    except ImportError:
        if sys.platform != "win32":
            return 0
        from ctypes import wintypes
        # Native Windows WinAPI fallback, so psutil is not mandatory.
        class Ctrs(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
        k = ctypes.windll.kernel32
        k.OpenProcess.restype = wintypes.HANDLE
        handle = k.OpenProcess(0x1000 | 0x0010, False, int(pid))
        if not handle:
            return 0
        try:
            p = Ctrs()
            p.cb = ctypes.sizeof(p)
            ok = ctypes.windll.psapi.GetProcessMemoryInfo(handle, ctypes.byref(p), p.cb)
            return int(p.WorkingSetSize) if ok else 0
        finally:
            k.CloseHandle(handle)
    except Exception:
        return 0


def precheck_environment():
    if not DATA_DIR.is_dir():
        raise RuntimeError(f"Missing training directory: {DATA_DIR}")
    if not META_FILE.is_file():
        raise RuntimeError(f"Missing archived feature metadata: {META_FILE}")
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    if head != EXPECTED_HEAD:
        raise RuntimeError(f"Git HEAD differs from pinned C03 reference: {head}")
    meta = json.loads(META_FILE.read_text(encoding="utf-8"))
    cols = meta.get("num_cols")
    if not isinstance(cols, list) or len(cols) != 46 or meta.get("cat_cols") != []:
        raise RuntimeError("Archived feature names are not the expected 46 numeric, zero categorical")
    if len(set(cols)) != len(cols):
        raise RuntimeError("Duplicate feature names")
    return cols


def run_worker(rows_max, trees):
    # Only entered from --probe child. No validation/test references anywhere.
    import numpy as np
    import pyarrow.parquet as pq
    from scipy.sparse import csr_matrix
    # Import the audited local helper, never an unrelated installed package.
    src = str(Path.cwd().resolve() / "src")
    if not (Path(src) / "ids_eval_framework" / "_native" / "protocol_b_grid.py").is_file():
        raise RuntimeError("Audited Protocol B grid helper not found in local src/")
    if src not in sys.path:
        sys.path.insert(0, src)
    from ids_eval_framework._native import protocol_b_grid as grid
    if not Path(grid.__file__).resolve().is_relative_to(Path(src)):
        raise RuntimeError("Imported Protocol B helper outside pinned repository")

    cols = precheck_environment()
    paths = sorted(DATA_DIR.glob("*.parquet"))
    if len(paths) != 239:
        raise RuntimeError(f"Expected 239 training files; found {len(paths)}")
    # Bernoulli sampling across ALL source TRAIN parts avoids first-shard bias.
    # Approximate engineering sample, deliberately NOT the historical 1.5m cap sample.
    rng = np.random.RandomState(123456)
    raw_total = sum(pq.ParquetFile(str(p)).metadata.num_rows for p in paths)
    if raw_total != 3676338:
        raise RuntimeError(f"Training source footer rows changed: {raw_total}")
    p_take = min(1.0, 1.3 * rows_max / raw_total)
    chunks_x, chunks_y = [], []
    count = 0
    ts = time.monotonic()
    for p in paths:
        pf = pq.ParquetFile(str(p))
        for batch in pf.iter_batches(batch_size=8192, columns=cols+["y_stage1_attack", "y_stage2_family"]):
            df = batch.to_pandas()
            y1 = df["y_stage1_attack"].to_numpy()
            y2 = df["y_stage2_family"].astype(str).to_numpy()
            # Stage2 trained only on represented attack families. Botnet and benign excluded.
            eligible = (y1 == 1) & np.isin(y2, KNOWN)
            pick = np.flatnonzero(eligible & (rng.random(len(df)) < p_take))
            if len(pick):
                x = df.iloc[pick][cols].to_numpy(dtype=np.float64, copy=True)
                chunks_x.append(x)
                chunks_y.extend(y2[pick])
                count += len(pick)
            del df
    X_raw = np.concatenate(chunks_x, axis=0)
    labels = np.asarray(chunks_y, dtype=object)
    del chunks_x, chunks_y
    if len(labels) < rows_max:
        raise RuntimeError(f"Only {len(labels)} sampled known attack rows; expected at least {rows_max}")
    # Sample to exact cap after uniform source scan.
    selected = np.random.RandomState(123457).choice(len(labels), size=rows_max, replace=False)
    X_raw = X_raw[selected]
    labels = labels[selected]
    counts = {f: int(np.sum(labels == f)) for f in KNOWN}
    if min(counts.values()) < 20:
        raise RuntimeError(f"Sample missing adequate known-family support: {counts}")
    print(f"Sampled source TRAIN ONLY; rows={rows_max}; known-family counts={counts}; data scan={time.monotonic()-ts:.1f}s", flush=True)
    # Approximate numeric scaler from sample for memory sizing, not historic saved preprocess.
    means = np.nanmean(X_raw, axis=0)
    stds = np.nanstd(X_raw, axis=0)
    stds = np.where(np.isfinite(stds) & (stds >= 1e-8), stds, 1.0)
    X_scaled = (X_raw - means) / stds
    np.nan_to_num(X_scaled, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    X_full = csr_matrix(X_scaled)
    del X_raw, X_scaled, means, stds
    gc.collect()
    y_idx_full = np.array([KNOWN.index(str(v)) for v in labels], dtype=np.int32)
    del labels
    gc.collect()

    for nrows in [30000, rows_max]:
        if nrows > rows_max: continue
        idx = np.random.RandomState(123459).choice(rows_max, size=nrows, replace=False)
        X = X_full[idx].tocsr()
        y = y_idx_full[idx]
        weights = grid.stage2_sample_weights(y, len(KNOWN), "balanced")
        model = grid.build_stage2_model(
            "rf", {"n_estimators": trees, "max_depth": None,
                   "min_samples_leaf": 1, "max_features": "sqrt"},
            n_classes=len(KNOWN), seed=133,
        )
        model.set_params(n_jobs=1)
        t0 = time.monotonic()
        model.fit(X, y, sample_weight=weights)
        sec = time.monotonic() - t0
        node_counts = [e.tree_.node_count for e in model.estimators_]
        tree_bytes = sum(e.tree_.__getstate__()["nodes"].nbytes + e.tree_.value.nbytes for e in model.estimators_)
        print(f"FIT_PROBE ONLY sample_rows={nrows} trees={trees} elapsed_s={sec:.2f} "
              f"total_tree_nodes={sum(node_counts)} tree_storage_MiB={tree_bytes/MiB:.2f} "
              f"mean_nodes_per_tree={sum(node_counts)/trees:.1f} "
              f"hypothetical_300tree_storage_at_THIS_sample_MiB={(tree_bytes/trees*300)/MiB:.1f}", flush=True)
        del X, y, weights, model
        gc.collect()
    del X_full, y_idx_full
    gc.collect()
    print("WORKER COMPLETE: No model checkpoints, test reads, or scientific metrics produced.", flush=True)


def parent_probe(rows_max, trees, min_start_gib, stop_free_gib, max_child_gib):
    precheck_environment()
    m = os_memory()
    print(f"SYSTEM baseline physical: total={m['physical_total']/GiB:.2f} GiB; available={m['physical_available']/GiB:.2f} GiB", flush=True)
    if m['physical_available'] < min_start_gib*GiB:
        raise RuntimeError(f"ABORT BEFORE FIT: available RAM < {min_start_gib:.1f} GiB. Close apps or restart first.")
    with tempfile.TemporaryFile(mode="w+t", encoding="utf-8") as f:
        cmd = [sys.executable, "-B", "-u", str(Path(__file__).resolve()), "--_worker",
               "--rows", str(rows_max), "--trees", str(trees)]
        proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=Path.cwd())
        peak_rss = 0
        min_free = m["physical_available"]
        halted = None
        start = time.monotonic()
        try:
            while proc.poll() is None:
                time.sleep(0.3)
                vm = os_memory()
                min_free = min(min_free, vm["physical_available"])
                peak_rss = max(peak_rss, process_rss(proc.pid))
                if vm["physical_available"] < stop_free_gib*GiB:
                    halted = f"System available RAM fell below {stop_free_gib:.1f} GiB"
                    proc.kill()
                    break
                if peak_rss > max_child_gib*GiB:
                    halted = f"Probe process RSS exceeded {max_child_gib:.1f} GiB"
                    proc.kill()
                    break
            proc.wait()
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
        f.seek(0)
        print(f.read(), end="", flush=True)
        print(f"MONITOR peak_child_working_set={peak_rss/GiB:.2f} GiB; "
              f"lowest_system_available={min_free/GiB:.2f} GiB; wall_s={time.monotonic()-start:.1f}")
        if halted:
            print(f"SAFETY HALT: {halted}. This is NOT an experimental failure.")
            return 3
        if proc.returncode != 0:
            print(f"PROBE DID NOT COMPLETE: worker exit code={proc.returncode}")
            return 2
        print("RESULT: BOUNDED MEMORY PROBE COMPLETE (engineering data only).")
        print("CAUTION: Full 1.2M-row, 300-tree RF memory cannot be linearly certified from this sample.")
        return 0


def main():
    ap = argparse.ArgumentParser(description="C03 engineering-only, RAM-guarded Random Forest Stage2 sizing probe")
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--inspect", action="store_true", help="Print physical / pagefile status; NO fit")
    group.add_argument("--probe", action="store_true", help="Explicitly authorize BOUNDED train-only RF fits")
    group.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--rows", type=int, default=80000, help="Engineering sampling cap, max 100000")
    ap.add_argument("--trees", type=int, default=8, help="Engineering tree count, max 12")
    ap.add_argument("--min-start-gib", type=float, default=3.0)
    ap.add_argument("--stop-free-gib", type=float, default=1.5)
    ap.add_argument("--max-child-gib", type=float, default=3.0)
    args = ap.parse_args()
    if args.inspect:
        m = os_memory()
        print("=== WINDOWS SYSTEM MEMORY / PAGE FILE ===")
        print(f"Physical installed/usable: {m['physical_total']/GiB:.2f} GiB; available: {m['physical_available']/GiB:.2f} GiB")
        if "commit_limit" in m:
            print(f"Approx system commit limit: {m['commit_limit']/GiB:.2f} GiB; available commit: {m['commit_available']/GiB:.2f} GiB")
        else:
            print(f"Swap/pagefile reported: total={m['swap_total']/GiB:.2f} GiB; used={m['swap_used']/GiB:.2f} GiB")
        print("No training, file modification or test-data access.")
        return 0
    if not (30000 <= args.rows <= 100000 and 1 <= args.trees <= 12):
        raise RuntimeError("Unsafe or invalid sample/tree cap (rows 30000..100000; trees 1..12)")
    if args._worker:
        run_worker(args.rows, args.trees)
        return 0
    print("=== C03 ENGINEERING RF MEMORY PROBE ===")
    print("No validation/test rows; 30k and 80k (or specified <=100k) TRAIN samples; 8 (or <=12) trees only")
    print("Historical RF Stage2 constructor and balanced weights; model metrics NOT reported or retained")
    return parent_probe(args.rows, args.trees, args.min_start_gib, args.stop_free_gib, args.max_child_gib)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"BLOCKED: {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(2)
