#!/usr/bin/env python3
"""Regenerate release summaries that consume Protocol A system macro-F1."""

from __future__ import annotations

import csv
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ids_eval_framework.src.protocol_a_correction import load_evidence, metric_index  # noqa: E402


EVIDENCE = ROOT / "outputs" / "evidence" / "protocol_a_confusion_matrices.jsonl"
PRIMARY = "system_macro_f1_supported_labels"
HISTORICAL = "system_macro_f1_declared_output_labels_historical"


def read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def write_rows(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def replace_field(fields: list[str], old: str, new: list[str]) -> list[str]:
    idx = fields.index(old)
    return fields[:idx] + new + fields[idx + 1 :]


def update_run_summary(path: Path, index: dict[tuple[str, str], dict[str, object]]) -> None:
    fields, rows = read_rows(path)
    if "system_macro_f1" not in fields:
        raise ValueError(f"Expected historical system_macro_f1 in {path}")
    for row in rows:
        record = index[(row["run_name"], row["policy_variant"])]
        historical = float(record[HISTORICAL])
        if abs(float(row["system_macro_f1"]) - historical) > 5e-15:
            raise ValueError(f"Historical metric mismatch in {path}: {row['run_name']} {row['policy_variant']}")
        row[PRIMARY] = format(float(record[PRIMARY]), ".15f")
        row[HISTORICAL] = format(historical, ".15f")
        del row["system_macro_f1"]
    write_rows(path, replace_field(fields, "system_macro_f1", [PRIMARY, HISTORICAL]), rows)


def update_seed_summary(path: Path, records: list[dict[str, object]]) -> None:
    fields, rows = read_rows(path)
    out: list[dict[str, object]] = []
    by_key: dict[str, list[float]] = {}
    for dataset in sorted({str(r["dataset"]) for r in records if r["surface"] == "seed_reliability"}):
        for model in ("rf", "xgb"):
            for policy in ("strict", "strict_tau"):
                key = f"{dataset}::{model}::{policy}"
                values = [
                    float(r[PRIMARY])
                    for r in records
                    if r["surface"] == "seed_reliability"
                    and r["dataset"] == dataset
                    and r["model_family"] == model
                    and r["policy_variant"] == policy
                ]
                if len(values) != 5:
                    raise ValueError(f"Expected five seed values for {key}, found {len(values)}")
                by_key[key] = values
    for row in rows:
        if row["lane"] == "protocol_a_two_stage" and row["metric"] == "system_macro_f1":
            historical = dict(row)
            historical["metric"] = HISTORICAL
            out.append(historical)
            values = by_key[row["comparison_key"]]
            mean = statistics.mean(values)
            sd = statistics.stdev(values)
            half = 2.7764451051977987 * sd / (len(values) ** 0.5)
            corrected = dict(row)
            corrected.update(
                {
                    "metric": PRIMARY,
                    "n": str(len(values)),
                    "mean": repr(mean),
                    "sd": repr(sd),
                    "min": repr(min(values)),
                    "max": repr(max(values)),
                    "ci95_low": repr(mean - half),
                    "ci95_high": repr(mean + half),
                }
            )
            out.append(corrected)
        else:
            out.append(row)
    write_rows(path, fields, out)


def update_reference(path: Path, records: list[dict[str, object]]) -> None:
    fields, rows = read_rows(path)
    selected: dict[tuple[str, str], dict[str, object]] = {}
    for record in records:
        if record["surface"] != "reference_profile" or record["policy_variant"] != "strict_tau":
            continue
        key = (str(record["profile"]), str(record["dataset"]))
        if key not in selected or str(record["run_id"]) > str(selected[key]["run_id"]):
            selected[key] = record
    for row in rows:
        old = float(row["full_framework_macro_f1"])
        if row["full_framework_surface"] == "protocol_a_reference_profile":
            record = selected[(row["model_profile"], row["dataset"])]
            historical = float(record[HISTORICAL])
            if abs(old - historical) > 5e-15:
                raise ValueError(f"Reference historical mismatch: {row['model_profile']} {row['dataset']}")
            row["full_framework_macro_f1_supported_labels"] = format(float(record[PRIMARY]), ".15f")
            row["full_framework_macro_f1_declared_output_labels_historical"] = format(historical, ".15f")
        else:
            row["full_framework_macro_f1_supported_labels"] = row["full_framework_macro_f1"]
            row["full_framework_macro_f1_declared_output_labels_historical"] = ""
        del row["full_framework_macro_f1"]
    write_rows(
        path,
        replace_field(
            fields,
            "full_framework_macro_f1",
            ["full_framework_macro_f1_supported_labels", "full_framework_macro_f1_declared_output_labels_historical"],
        ),
        rows,
    )


def update_flat_comparison(path: Path, records: list[dict[str, object]]) -> None:
    fields, rows = read_rows(path)
    core = {
        (str(r["dataset"]), str(r["policy_variant"])): r
        for r in records
        if r["surface"] == "core" and r["model_family"] == "rf"
    }
    for row in rows:
        strict = core[(row["dataset"], "strict")]
        operational = core[(row["dataset"], "strict_tau")]
        competitive = float(row["competitive_macro_f1"])
        row.update(
            {
                "strict_baseline_macro_f1_supported_labels": format(float(strict[PRIMARY]), ".15f"),
                "strict_baseline_macro_f1_declared_output_labels_historical": format(float(strict[HISTORICAL]), ".15f"),
                "operational_macro_f1_supported_labels": format(float(operational[PRIMARY]), ".15f"),
                "operational_macro_f1_declared_output_labels_historical": format(float(operational[HISTORICAL]), ".15f"),
                "delta_macro_f1_supported_labels_vs_strict": repr(competitive - float(strict[PRIMARY])),
                "delta_macro_f1_supported_labels_vs_operational": repr(competitive - float(operational[PRIMARY])),
            }
        )
        for old in ("strict_baseline_macro_f1", "operational_macro_f1", "delta_macro_f1_vs_strict", "delta_macro_f1_vs_operational"):
            del row[old]
    replacements = {
        "strict_baseline_macro_f1": ["strict_baseline_macro_f1_supported_labels", "strict_baseline_macro_f1_declared_output_labels_historical"],
        "operational_macro_f1": ["operational_macro_f1_supported_labels", "operational_macro_f1_declared_output_labels_historical"],
        "delta_macro_f1_vs_strict": ["delta_macro_f1_supported_labels_vs_strict"],
        "delta_macro_f1_vs_operational": ["delta_macro_f1_supported_labels_vs_operational"],
    }
    new_fields: list[str] = []
    for field in fields:
        new_fields.extend(replacements.get(field, [field]))
    write_rows(path, new_fields, rows)


def main() -> int:
    raw = load_evidence(EVIDENCE)
    index = metric_index(raw)
    records = list(index.values())
    update_run_summary(ROOT / "outputs" / "summaries" / "protocol_a_core_summary.csv", index)
    update_run_summary(ROOT / "outputs" / "summaries" / "external_protocol_a_summary.csv", index)
    update_seed_summary(ROOT / "outputs" / "summaries" / "seed_reliability_summary.csv", records)
    update_reference(ROOT / "outputs" / "summaries" / "reference_profile_metric_drop.csv", records)
    update_flat_comparison(ROOT / "outputs" / "summaries" / "protocol_a_flat_vs_two_stage.csv", records)
    print("Corrected Protocol A release summaries regenerated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
