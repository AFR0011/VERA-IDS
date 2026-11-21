#!/usr/bin/env python3
"""C03 pilot: read-only *effective feature* and proxy-naming preflight.

Inspects the exact first TRAIN Parquet part used by the historical preprocessor
for numeric/categorical inference, compares its computed effective features to
prepared-data leakage and label/identifier naming rules, and inspects all split
Parquet schema footers. Never fits models or preprocessors; no writes.

This is NOT a substitute for source capture provenance, row-level grouping,
statistical shortcut ablations, label-rule audits or independent environment tests.

From VERA-IDS root: python -B c03_feature_proxy_preflight.py
"""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import re
import sys

DATASET = Path('processed_V5/B_day_file/CICIoT2023')
REQUIRED = {'y_stage1_attack', 'y_stage2_family'}
SUBJECT = {'y_stage1_attack', 'y_stage2_family', 'y_stage2_fine',
           'label', 'attempted_category'}
PROVENANCE_TERMS = ('source_file', 'filename', 'scenario', 'capture', 'session',
                    'experiment_id', 'device_id', 'flow_id', 'packet_id',
                    'record_id', 'row_id', 'run_id', 'split_id', 'label_rule')
LABEL_TERMS = ('label', 'attack', 'category', 'class', 'target', 'ground_truth',
               'y_stage', 'is_unknown', 'is_benign', 'fine_label')


def norm(name: object) -> str:
    return str(name).strip().lower().replace(' ', '_')


def suspicious_reasons(column: str) -> list[str]:
    c = norm(column)
    hits=[]
    if c in SUBJECT or any(t in c for t in LABEL_TERMS):
        hits.append('target/label-like name (needs source-level review)')
    if any(t in c for t in PROVENANCE_TERMS):
        hits.append('provenance/identity-like name (needs source-level review)')
    if c in ('time','datetime','date','timestamp') or c.endswith(('_timestamp','_date','_time','_ts')):
        hits.append('raw timing identifier candidate')
    if c in ('ip','srcip','dstip','port','srcport','dstport','sport','dsport') or re.search(r'(^|_)(ip|port)(_|$)',c):
        hits.append('raw address/port identifier candidate')
    # These can be legitimate measurements, not necessarily leaks.
    if 'bucket' in c and ('port' in c or 'time' in c):
        hits.append('coarsened port/time: review but not automatically leakage')
    return hits


def run(root: Path) -> int:
    print('=== C03 EFFECTIVE-FEATURE / PROXY-NAMING PREFLIGHT (READ ONLY) ===')
    print('Scope: CICIoT2023 B_day_file, Botnet seed123 pilot; current training helper')
    if not (root / 'src/ids_eval_framework/src/two_stage_engine.py').is_file():
        raise RuntimeError('Run from VERA-IDS root')
    source=root/DATASET
    if not source.is_dir():
        raise RuntimeError('Processed dataset missing: '+str(source))
    import pandas as pd
    import pyarrow.parquet as pq
    from ids_eval_framework.src import two_stage_engine as helper
    from ids_eval_framework._native import protocol_b_grid as grid
    from ids_eval_framework._native.prepare_datasets import detect_leakage_cols

    def canonical_columns(p: Path) -> tuple[str,...]:
        return tuple(norm(c) for c in pq.ParquetFile(p).schema_arrow.names)

    files={}
    baseline=None
    schema_problems=[]
    for split,expected in [('train',239),('val',65),('test',65)]:
        raw=[Path(p) for p in grid.list_parts(str(source),split)]
        files[split]=raw
        if len(raw)!=expected:
            schema_problems.append(f'{split}: expected {expected} parts, got {len(raw)}')
        sets=[]; rows=0
        for f in raw:
            if f.suffix != '.parquet':
                schema_problems.append(f'{split}: non-Parquet input {f.name}')
                continue
            cols=canonical_columns(f)
            if len(cols)!=len(set(cols)):
                schema_problems.append(f'{split}: duplicate normalized names in {f.name}')
            if not REQUIRED.issubset(cols):
                schema_problems.append(f'{split}: required labels missing from {f.name}')
            if baseline is None: baseline=cols
            if set(cols) != set(baseline):
                schema_problems.append(f'{split}: schema differs in {f.name}')
            rows+=pq.ParquetFile(f).metadata.num_rows
            sets.append(cols)
        print(f'{split:5s}: {len(raw)} parts, {rows:,} footer rows, schema_variants={len(set(sets))}')
    if not files['train']:
        raise RuntimeError('No train files to inspect')
    first=files['train'][0]
    # Historical helper.fit_preprocessor calls read_part(parts[0]), then infer_cols(df0.head(5000)).
    # Read only that first part; do not fit or serialize a preprocessing object.
    df0=grid.read_part(str(first),usecols=None)
    df0.columns=[grid.canonical_col(c) for c in df0.columns]
    head=df0.head(5000)
    numeric, categorical=helper.infer_cols(head)
    effective=list(numeric)+list(categorical)
    dropped=[c for c in df0.columns if c not in effective]
    # historical prep's leakage NAME detector (not empirical source validity)
    prepared_leakage=set(detect_leakage_cols(list(df0.columns)))
    retained_prepared=sorted(set(effective)&prepared_leakage)
    effective_labels=sorted(set(effective)&set(helper.CFG['never_feature_cols']))
    feature_candidates={c:suspicious_reasons(c) for c in effective if suspicious_reasons(c)}
    omitted_candidates={c:suspicious_reasons(c) for c in dropped if suspicious_reasons(c)}

    print('\n=== HISTORICAL FEATURE INFERENCE ===')
    print('First source part:',first.relative_to(root).as_posix())
    print('First part rows:',len(df0),'rows used to infer features:',len(head))
    print(f'Processed input columns={len(df0.columns)}; numeric={len(numeric)}; categorical={len(categorical)}; omitted={len(dropped)}')
    print('Historical never_feature_cols:',', '.join(sorted(map(str,helper.CFG['never_feature_cols']))))
    print('NUMERIC FEATURES:',', '.join(numeric) if numeric else '(none)')
    print('CATEGORICAL FEATURES:',', '.join(categorical) if categorical else '(none)')
    print('OMITTED INPUT COLUMNS:',', '.join(dropped) if dropped else '(none)')

    print('\n=== NAME-BASED LEAKAGE / PROXY SCREEN ===')
    print('Direct excluded-target names retained as features:',effective_labels)
    print('Prepared-data identifier/time/IP/port rules match retained features:',retained_prepared)
    print('Review candidates within model feature inputs:',len(feature_candidates))
    for c,why in sorted(feature_candidates.items()):
        print('  REVIEW',repr(c),'|','; '.join(why))
    print('Flagged names among omitted columns (not model features):',len(omitted_candidates))
    for c,why in sorted(omitted_candidates.items()):
        print('  OMITTED',repr(c),'|','; '.join(why))

    meta= root/'outputs/13_jisa_q1_revision/phase3_validation_blind/score_generation/seed_123/Botnet/preprocessor_meta.json'
    if meta.exists():
        data=json.loads(meta.read_text(encoding='utf-8'))
        num_saved=[norm(x) for x in data.get('num_cols',[])]
        cat_saved=[norm(x) for x in data.get('cat_cols',[])]
        print('\nArchived blind-preprocessor column metadata found:',meta.relative_to(root))
        print('Saved numeric names identical:',num_saved==numeric)
        print('Saved categorical names identical:',cat_saved==categorical)
        if num_saved!=numeric or cat_saved!=categorical:
            schema_problems.append('archived preprocessor feature list differs from current inferred features')
    else:
        print('\nArchived preprocessor_meta.json absent; no historic object loaded or inferred')

    print('\n=== DISPOSITION ===')
    if schema_problems:
        print('BLOCKED: structural issues=',len(schema_problems))
        for problem in schema_problems[:15]: print('  FLAG',problem)
    elif effective_labels:
        print('BLOCKED: explicit excluded target column entered model features')
    else:
        print('PASS: current schemas and explicit target-exclusion checks')
    if retained_prepared or feature_candidates:
        print('OPEN: flagged model-feature names need source/proxy review; this is NOT a leakage verdict')
    else:
        print('OPEN: no suspicious names found, but proxy leakage remains possible in numeric feature values')
    print('NOT ESTABLISHED: raw capture independence, label-rule validity, cross-split flow identity or absence of proxy leakage.')
    print('No trained models, no test labels/scores, no repository writes.')
    return 2 if schema_problems or effective_labels else 0


if __name__=='__main__':
    try:
        sys.exit(run(Path.cwd()))
    except Exception as exc:
        print('AUDIT BLOCKED:',type(exc).__name__,str(exc))
        sys.exit(2)
