"""Repeated robustness evaluation for literature-derived classifier configurations.

This lane transfers externally specified classifier settings into common VERA
Protocol-A and Protocol-B conditions. It does not reproduce source-paper scores.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from ids_eval_framework.src.paths import load_config, resolve_repo_path
from ids_eval_framework.src.reference_framework_eval import (
    configured_profiles,
    run_protocol_a_reference_profiles,
    run_protocol_b_reference_profiles,
)


def robustness_cfg(config: Mapping[str, Any] | None) -> dict[str, Any]:
    return dict((config or {}).get("external_profile_robustness", {}) or {})


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return Path(resolve_repo_path(str(path)))


def _prepare_seed_config(
    base_config: Mapping[str, Any],
    *,
    out_root: Path,
    seed: int,
    profiles: Sequence[str],
    smoke: bool,
    protocol_a_processed_root: str | Path | None = None,
    protocol_b_audit_roots: Sequence[str | Path] | None = None,
    protocol_b_processed_overrides: Mapping[str, str | Path] | None = None,
) -> dict[str, Any]:
    cfg = deepcopy(dict(base_config))
    ref = cfg.setdefault("reference_framework_eval", {})
    ref["out_root"] = str(out_root)
    ref["enabled_profiles"] = list(profiles)
    proto_a = ref.setdefault("protocol_a", {})
    proto_a["seed"] = int(seed)
    if protocol_a_processed_root is not None:
        proto_a["processed_root"] = str(protocol_a_processed_root)

    proto_b = ref.setdefault("protocol_b", {})
    legacy = proto_b.setdefault("legacy_overrides", {})
    legacy["random_seed"] = int(seed)
    if protocol_b_audit_roots:
        roots = [str(x) for x in protocol_b_audit_roots]
        legacy["audit_roots"] = roots
        legacy["audit_root"] = roots[0]
        legacy["manifest_glob"] = [
            str(Path(root) / "*" / "manifests" / "*.json")
            for root in roots
        ]
    if protocol_b_processed_overrides:
        legacy["processed_dir_overrides"] = {
            str(k): str(v) for k, v in protocol_b_processed_overrides.items()
        }

    if smoke:
        smoke_cfg = ref.setdefault("smoke", {})
        smoke_cfg["out_root"] = str(out_root)
    return cfg


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def aggregate_protocol_a(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    metrics = [
        "stage1_auc",
        "stage1_fpr",
        "stage2_macro_f1_fixedK",
        "stage2_macro_f1_present",
        "system_macro_f1_supported_labels",
        "system_accuracy",
        "system_benign_family_fp_rate",
        "system_overall_reject_rate",
    ]
    rows: list[dict[str, Any]] = []
    groups = frame.groupby(["model_profile", "paper", "dataset"], dropna=False)
    for keys, group in groups:
        profile, paper, dataset = keys
        for metric in metrics:
            if metric not in group.columns:
                continue
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            if values.empty:
                continue
            rows.append({
                "model_profile": profile,
                "paper": paper,
                "dataset": dataset,
                "metric": metric,
                "n_seeds": int(len(values)),
                "mean": float(values.mean()),
                "sd": float(values.std(ddof=1)) if len(values) > 1 else float("nan"),
                "min": float(values.min()),
                "max": float(values.max()),
            })
    return pd.DataFrame(rows)


def aggregate_protocol_b_holdouts(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    metrics = [
        "macro_f1",
        "accuracy",
        "unknown_detection_rate",
        "false_unknown_rate_all_known",
        "false_unknown_rate_known_attacks",
        "benign_family_fp_rate",
        "overall_reject_rate",
    ]
    rows: list[dict[str, Any]] = []
    keys = ["model_profile", "dataset", "holdout_family"]
    for group_keys, group in frame.groupby(keys, dropna=False):
        profile, dataset, holdout = group_keys
        paper = group["paper"].iloc[0] if "paper" in group.columns and len(group) else ""
        for metric in metrics:
            if metric not in group.columns:
                continue
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            if values.empty:
                continue
            rows.append({
                "model_profile": profile,
                "paper": paper,
                "dataset": dataset,
                "holdout_family": holdout,
                "metric": metric,
                "n_seeds": int(len(values)),
                "mean": float(values.mean()),
                "sd": float(values.std(ddof=1)) if len(values) > 1 else float("nan"),
                "min": float(values.min()),
                "max": float(values.max()),
            })
    return pd.DataFrame(rows)


def protocol_b_family_heterogeneity(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    metrics = ["macro_f1", "unknown_detection_rate"]
    rows: list[dict[str, Any]] = []
    for (seed, profile, dataset), group in frame.groupby(
        ["seed", "model_profile", "dataset"], dropna=False
    ):
        paper = group["paper"].iloc[0] if "paper" in group.columns and len(group) else ""
        for metric in metrics:
            if metric not in group.columns:
                continue
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            if values.empty:
                continue
            rows.append({
                "seed": int(seed),
                "model_profile": profile,
                "paper": paper,
                "dataset": dataset,
                "metric": metric,
                "n_holdouts": int(len(values)),
                "holdout_mean": float(values.mean()),
                "holdout_min": float(values.min()),
                "holdout_max": float(values.max()),
                "holdout_range": float(values.max() - values.min()),
            })
    return pd.DataFrame(rows)


def summarize_family_heterogeneity(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    rows: list[dict[str, Any]] = []
    for (profile, paper, dataset, metric), group in frame.groupby(
        ["model_profile", "paper", "dataset", "metric"], dropna=False
    ):
        for field in ["holdout_mean", "holdout_range"]:
            values = pd.to_numeric(group[field], errors="coerce").dropna()
            if values.empty:
                continue
            rows.append({
                "model_profile": profile,
                "paper": paper,
                "dataset": dataset,
                "metric": metric,
                "summary": field,
                "n_seeds": int(len(values)),
                "mean": float(values.mean()),
                "sd": float(values.std(ddof=1)) if len(values) > 1 else float("nan"),
                "min": float(values.min()),
                "max": float(values.max()),
            })
    return pd.DataFrame(rows)


def collect_seed_outputs(out_root: Path, seeds: Sequence[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    pa_frames: list[pd.DataFrame] = []
    pb_frames: list[pd.DataFrame] = []
    for seed in seeds:
        seed_root = out_root / f"seed_{int(seed)}"
        pa = _read_csv(seed_root / "protocol_a" / "summary" / "protocol_a_reference_profile_summary.csv")
        if not pa.empty:
            pa.insert(0, "seed", int(seed))
            pa_frames.append(pa)
        pb = _read_csv(seed_root / "protocol_b" / "summary" / "best_per_holdout.csv")
        if not pb.empty:
            pb.insert(0, "seed", int(seed))
            pb_frames.append(pb)
    return (
        pd.concat(pa_frames, ignore_index=True, sort=False) if pa_frames else pd.DataFrame(),
        pd.concat(pb_frames, ignore_index=True, sort=False) if pb_frames else pd.DataFrame(),
    )


def write_aggregate_outputs(out_root: Path, seeds: Sequence[int]) -> None:
    summary_dir = out_root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    pa, pb = collect_seed_outputs(out_root, seeds)
    pa.to_csv(summary_dir / "protocol_a_external_profile_runs.csv", index=False)
    pb.to_csv(summary_dir / "protocol_b_external_profile_runs.csv", index=False)
    aggregate_protocol_a(pa).to_csv(
        summary_dir / "protocol_a_external_profile_summary.csv", index=False
    )
    aggregate_protocol_b_holdouts(pb).to_csv(
        summary_dir / "protocol_b_holdout_seed_summary.csv", index=False
    )
    heterogeneity = protocol_b_family_heterogeneity(pb)
    heterogeneity.to_csv(
        summary_dir / "protocol_b_family_heterogeneity_by_seed.csv", index=False
    )
    summarize_family_heterogeneity(heterogeneity).to_csv(
        summary_dir / "protocol_b_family_heterogeneity_summary.csv", index=False
    )


def run_external_profile_robustness(
    config: Mapping[str, Any] | None,
    *,
    dry_run: bool = False,
    smoke: bool = False,
    profiles: Sequence[str] | None = None,
    seeds: Sequence[int] | None = None,
    skip_protocol_a: bool = False,
    skip_protocol_b: bool = False,
    protocol_a_processed_root: str | Path | None = None,
    protocol_b_audit_roots: Sequence[str | Path] | None = None,
    protocol_b_processed_overrides: Mapping[str, str | Path] | None = None,
) -> Path:
    cfg = robustness_cfg(config)
    if not cfg:
        raise ValueError("Missing external_profile_robustness configuration.")

    base_config_path = str(cfg.get("base_reference_config", "config/reference_framework_eval.yml"))
    base = load_config(base_config_path)
    available = configured_profiles(base)

    selected_profiles = list(profiles or cfg.get("profiles", []))
    selected_seeds = [int(x) for x in (seeds or cfg.get("seeds", []))]
    if smoke:
        smoke_cfg = dict(cfg.get("smoke", {}) or {})
        selected_profiles = list(smoke_cfg.get("profiles", selected_profiles))
        selected_seeds = [int(x) for x in smoke_cfg.get("seeds", selected_seeds)]

    unknown = sorted(set(selected_profiles) - set(available))
    if unknown:
        raise ValueError(f"Unknown external profiles: {unknown}")

    smoke_cfg = dict(cfg.get("smoke", {}) or {})
    out_root_value = smoke_cfg.get("out_root") if smoke else None
    out_root = resolve_path(
        out_root_value or cfg.get("out_root", "outputs/13_external_profile_robustness")
    )
    run_a = bool(cfg.get("run_protocol_a", True)) and not skip_protocol_a
    run_b = bool(cfg.get("run_protocol_b", True)) and not skip_protocol_b

    if dry_run:
        print(f"[dry-run] base_reference_config={base_config_path}")
        print(f"[dry-run] out_root={out_root}")
        print(f"[dry-run] profiles={selected_profiles}")
        print(f"[dry-run] seeds={selected_seeds}")
        print(f"[dry-run] protocol_a={run_a} protocol_b={run_b}")
        if protocol_a_processed_root is not None:
            pa_root = resolve_path(protocol_a_processed_root)
            print(f"[dry-run] protocol_a_processed_root={pa_root} exists={pa_root.exists()}")
        if protocol_b_audit_roots:
            for root in protocol_b_audit_roots:
                p = resolve_path(root)
                print(f"[dry-run] protocol_b_audit_root={p} exists={p.exists()}")
        if protocol_b_processed_overrides:
            for dataset, root in protocol_b_processed_overrides.items():
                p = resolve_path(root)
                print(f"[dry-run] protocol_b_processed[{dataset}]={p} exists={p.exists()}")
        print("[dry-run] open_set_replay=False sink_aware=False")
        return out_root

    for seed in selected_seeds:
        seed_root = out_root / f"seed_{seed}"
        seed_cfg = _prepare_seed_config(
            base,
            out_root=seed_root,
            seed=seed,
            profiles=selected_profiles,
            smoke=smoke,
            protocol_a_processed_root=protocol_a_processed_root,
            protocol_b_audit_roots=protocol_b_audit_roots,
            protocol_b_processed_overrides=protocol_b_processed_overrides,
        )
        if run_a:
            run_protocol_a_reference_profiles(
                seed_cfg,
                smoke=smoke,
                profiles=selected_profiles,
            )
        if run_b:
            run_protocol_b_reference_profiles(
                seed_cfg,
                smoke=smoke,
                profiles=selected_profiles,
            )

    write_aggregate_outputs(out_root, selected_seeds)
    return out_root
