#!/usr/bin/env python3
"""C03 pre-execution source/code byte snapshot. No model execution or repo writes.

Run at VERA-IDS root:
 python -B c03_input_content_snapshot.py --out "$HOME\\Downloads\\C03_input_content_manifest.json"
Later verify with:
 python -B c03_input_content_snapshot.py --check "$HOME\\Downloads\\C03_input_content_manifest.json"

The tool hashes the current files only. It CANNOT establish equality with the
historical experiment absent independently recorded historical byte hashes.
This is not an independence, proxy-leakage, or test-access audit.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
from c03_paths import RELOCATION

ORIGINAL_BASELINE_SHA = "13e667a86a6872ff6192d1662085383e2b526752ceb783f8fc372b51487de9f4"

DATASET = Path("processed_V5/B_day_file/CICIoT2023")
EXPECTED_COUNTS = {"train": 239, "val": 65, "test": 65}
CODE_FILES = [
    "src/ids_eval_framework/_native/protocol_b_grid.py",
    "src/ids_eval_framework/src/two_stage_engine.py",
]
EXTRA_IF_PRESENT = ["config/jisa_phase3_validation_blind.yml"]


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical(obj: object) -> bytes:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def verify_relocation(original, current, *, record_path=RELOCATION):
    """Verify the unchanged data against the old snapshot and code against its successor.

    The original snapshot is never rewritten or presented as matching edited code.
    The explicit relocation record pins a separate snapshot of the new layout.
    """
    for doc in (original, current):
        scope = {k: v for k, v in doc.items() if k != "snapshot_sha256"}
        if hashlib.sha256(canonical(scope)).hexdigest() != doc.get("snapshot_sha256"):
            raise RuntimeError("Snapshot content does not match its fingerprint")
    record = json.loads(Path(record_path).read_text(encoding="utf-8"))
    if (original["snapshot_sha256"] != ORIGINAL_BASELINE_SHA or
            record.get("original_snapshot_sha256") != ORIGINAL_BASELINE_SHA):
        raise RuntimeError("Unexpected original C03 snapshot; cannot apply relocation")
    if current["snapshot_sha256"] != record.get("relocated_snapshot_sha256"):
        raise RuntimeError("Relocated code/data differs from the recorded layout snapshot")
    if original["data_files"] != current["data_files"] or original["split_totals"] != current["split_totals"]:
        raise RuntimeError("Relocation must not change data identities or bytes")
    now = {f["relative_path"]: f for f in current["code_files"]}
    for f in original["code_files"]:
        if f["relative_path"].startswith(("src/", "config/")) and now.get(f["relative_path"]) != f:
            raise RuntimeError("Historical package source/config changed during relocation")
    old_policy = next(f for f in original["code_files"] if f["relative_path"] == "C03_APPROVED_SELECTOR_POLICY.json")
    policy = now["experiments/c03/C03_APPROVED_SELECTOR_POLICY.json"]
    if (policy["sha256"], policy["bytes"]) != (old_policy["sha256"], old_policy["bytes"]):
        raise RuntimeError("Scientific policy changed during relocation")
    print("Verified original data baseline plus separately recorded relocated code snapshot")
    return True


def below(candidate: Path, root: Path) -> bool:
    try:
        candidate.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def inventory(repo: Path, expected=True):
    dataset = repo / DATASET
    if not dataset.is_dir():
        raise FileNotFoundError(f"Missing {dataset}; run from repository root")
    files = []
    totals = {}
    seen = set()
    for split in ("train", "val", "test"):
        # Same supported filename families as the historical split collector.
        paths = []
        for glob in (f"{split}/*.parquet", f"{split}/*.csv",
                     f"{split}/*.csv.gz", f"{split}_*.parquet",
                     f"{split}_*.csv", f"{split}_*.csv.gz"):
            paths.extend(sorted(dataset.glob(glob)))
        paths = sorted(set(paths), key=lambda p: p.as_posix())
        if expected and len(paths) != EXPECTED_COUNTS[split]:
            raise RuntimeError(f"{split}: expected {EXPECTED_COUNTS[split]} files, got {len(paths)}")
        if not paths:
            raise RuntimeError(f"No {split} source files")
        byte_count = 0
        for p in paths:
            if not p.is_file() or p.is_symlink() or not below(p, dataset):
                raise RuntimeError(f"Unexpected/unsafe source path {p}")
            resolved = p.resolve()
            if resolved in seen:
                raise RuntimeError(f"File occurs in multiple split groups: {p}")
            seen.add(resolved)
            nbytes = p.stat().st_size
            byte_count += nbytes
            files.append({"role":split, "relative_path":p.relative_to(repo).as_posix(),
                          "bytes":nbytes, "sha256":sha(p)})
        totals[split] = {"files":len(paths),"bytes":byte_count}
    code = []
    tool_files = sorted((repo / "experiments/c03").glob("*.py"))
    tool_files += [repo / "experiments/c03/C03_APPROVED_SELECTOR_POLICY.json"]
    tool_files += sorted((repo / "experiments/c03/docs").glob("*.md"))
    relocated = [p.relative_to(repo).as_posix() for p in tool_files]
    for rel in CODE_FILES + [x for x in EXTRA_IF_PRESENT if (repo / x).is_file()] + relocated:
        p = repo / rel
        if not p.is_file() or p.is_symlink() or not below(p, repo):
            raise RuntimeError(f"Missing/unsafe source code file: {rel}")
        code.append({"relative_path":rel, "bytes":p.stat().st_size, "sha256":sha(p)})
    # This is a point-in-time prospective local snapshot; not a historical-source check.
    scope = {"schema_version":1,"claim_id":"C03","pilot":{
                "dataset":"CICIoT2023","holdout":"Botnet","model_seed":123},
             "snapshot_status":"CURRENT_LOCAL_BYTES_ONLY_NOT_HISTORICAL_PROOF",
             "data_files":files,"code_files":code,"split_totals":totals}
    return {**scope,"snapshot_sha256":hashlib.sha256(canonical(scope)).hexdigest()}


def summarize(doc):
    print("C03 CONTENT-BYTE PROVENANCE SNAPSHOT (NO MODEL FITTING)")
    for split, info in doc["split_totals"].items():
        print(f"  {split:5s}: files={info['files']:3d} bytes={info['bytes']:,}")
    print(f"  code/policy files hashed: {len(doc['code_files'])}")
    print(f"  manifest/content fingerprint: {doc['snapshot_sha256']}")
    print("LIMIT: present-day source bytes only, not a historical byte match,")
    print("      group-independence test, feature-proxy audit, or runner test isolation.")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    group=p.add_mutually_exclusive_group(required=True)
    group.add_argument("--out", type=Path, help="Save a new manifest OUTSIDE the repository")
    group.add_argument("--check", type=Path, help="Recalculate and compare to prior manifest")
    p.add_argument("--root", type=Path, default=Path.cwd())
    p.add_argument("--fixture",action="store_true",help=argparse.SUPPRESS)
    args=p.parse_args(argv)
    repo = args.root.resolve()
    if not (repo/"src/ids_eval_framework/_native/protocol_b_grid.py").is_file():
        raise RuntimeError(f"Not VERA-IDS repo root: {repo}")
    doc=inventory(repo,expected=not args.fixture)
    summarize(doc)
    if args.out:
        out=args.out.expanduser().resolve()
        if below(out,repo):
            raise RuntimeError("OUTPUT BLOCKED: manifest path must be OUTSIDE repository")
        if out.exists():
            raise FileExistsError("Refusing to overwrite prior provenance manifest")
        if not out.parent.is_dir():
            raise FileNotFoundError(f"Output directory absent: {out.parent}")
        with out.open("x",encoding="utf-8",newline="\n") as fh:
            json.dump(doc,fh,sort_keys=True,ensure_ascii=False,indent=2)
            fh.write("\n")
        print(f"SAVED outside repository: {out}")
        print("STATUS: BASELINE CREATED; runner not executed, no historical equivalence inferred")
    else:
        old=json.loads(args.check.expanduser().read_text(encoding="utf-8"))
        if old != doc:
            verify_relocation(old, doc)
        if old == doc or old.get("snapshot_sha256") == ORIGINAL_BASELINE_SHA:
            print("PASS: Snapshot verified (original baseline or documented relocation); data bytes unchanged")
        else:
            before={ (f["relative_path"],f.get("role","code")):(f["sha256"],f["bytes"]) for f in old.get("data_files",[])+old.get("code_files",[])}
            after={ (f["relative_path"],f.get("role","code")):(f["sha256"],f["bytes"]) for f in doc["data_files"]+doc["code_files"]}
            problems=[str(key) for key in sorted(set(before)|set(after)) if before.get(key)!=after.get(key)]
            print("FAIL: content snapshot changed, first discrepancies:")
            for x in problems[:15]: print("  "+x)
            if not problems: print("  Manifest metadata differs despite same per-file hashes")
            return 2
    return 0

if __name__=="__main__":
    try:
        sys.exit(main())
    except (OSError,ValueError,RuntimeError) as exc:
        print(f"BLOCKED: {exc}",file=sys.stderr)
        sys.exit(1)
