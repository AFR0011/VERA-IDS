# Edge-IIoTset original timestamps — Phase C4 first-row probe (2026-10-18)

**Status: sampled read successful, timestamp representation UNRESOLVED. Do not infer absent, corrupt or unusable timestamps. No independent attack-run boundaries have been recovered.**

Evidence: user-uploaded `outputs/17_revision_campaign/edgeiiot/original_timestamp_probe.json`, produced by `scripts/17_probe_edgeiiot_timestamps.py` on the actual original Edge-IIoTset files.

## Complete bounded-probe observations

- **24 of 24 expected original source CSVs** probed: 14 attack and 10 normal-device CSVs.
- First **256** rows inspected from each source, **6,144** total sampled rows. No full-source rows/counts, source-file timestamp spans, capture gaps or duplicate record checks were produced.
- `frame.time` parser categories across all samples: **5,873 `other_unparsed` (95.5892%)** and **271 `unresolved_numeric` (4.4108%)**. **Zero** sampled values parsed as calendar dates by the current fixed month-name / ISO / plausible-Unix-seconds recognizer.
- 22 of the 24 files: all 256 initial timestamps classified `other_unparsed`. `MITM_attack.csv`: all 256 classified `unresolved_numeric`. `Modbus.csv`: 241 `other_unparsed`, 15 `unresolved_numeric`.
- All 14 attack files have first-256-row binary label `1` and one filename-consistent `Attack_type`; all 10 normal CSVs have `Attack_label=0`, `Attack_type=Normal` in the first 256. This is **bounded sample label consistency only**, not full file consistency.
- `sample_calendar_day_counts` and `sample_first_parsed_dates` are empty in all 24 files. **No absence of repeat executions, source corruption, date impossibility or attack timing flaw can be inferred.**

## Interpretation and controlled blocker

This is a **format-identification problem in our probe code**, until literal source values are inspected. The original paper explicitly describes repeated attacks across different days/hours, but the filenames do not identify independent repetitions. The returned JSON intentionally does not contain literal `frame.time` values, making it impossible to choose a reliable parser from this evidence alone.

**Next minimal action, read only:** inspect a few literal `frame.time` strings from four representative original files (`Backdoor_attack.csv`, `MITM_attack.csv`, `Distance.csv`, and `Modbus.csv`), including both numeric and nonnumeric examples in Modbus. Use Python's `csv.DictReader` under a PowerShell here-string, printing only file basename and `repr(frame.time)`, never raw flow rows, IPs or payloads. Do not write the timestamps to GitHub/Drive until their syntax and provenance value are understood.

After observing the actual source format, revise the original timestamp parser and synthetic tests **without guessing semantics or timezone**; only then run a complete, read-only per-original-file date/range/type/support audit. This may also require PCAP corroboration for independent run boundaries.

**Other gates unchanged:** The complete curated ML/DNN binary label shortcut via text `"0.0"` was independently verified in Phase C3, but original CSVs have not yet undergone full-row leakage/support review. Edge-IIoTset remains an unapproved candidate for Protocol B. No model, split, feature exclusions, campaign YAML, or frozen release artifacts changed.

## Literal-value inspection command (PowerShell repository root)

```powershell
@'
import csv
from pathlib import Path
root = Path("data/raw/Edge_IIoTset")
targets = ("Backdoor_attack.csv", "MITM_attack.csv", "Distance.csv", "Modbus.csv")
for name in targets:
    files = list(root.rglob(name))
    if len(files) != 1:
        print(f"{name}: expected one file; found {len(files)}")
        continue
    numeric, other = [], []
    with files[0].open(encoding="utf-8-sig", newline="") as stream:
        for i, row in enumerate(csv.DictReader(stream)):
            if i == 256:
                break
            s = str(row.get("frame.time") or "")
            try:
                float(s.strip())
                group = numeric
            except ValueError:
                group = other
            if s not in group and len(group) < 3:
                group.append(s)
    print(f"{name}:")
    for label, values in (("numeric", numeric), ("other", other)):
        for value in values:
            print(f"  {label} (len={len(value)}): {value!r}")
'@ | uv run python -
```

The command prints **only timestamp field values**, which are part of the public research dataset. Do not use PowerShell `Import-Csv` for whole huge CSV files merely to inspect three timestamps. No source or report writes are performed.
