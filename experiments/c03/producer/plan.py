"""Metadata-only planning. Never imports models or opens benchmark observations."""
from __future__ import annotations

import csv
import hashlib
from importlib.metadata import version
import json
from pathlib import Path, PurePosixPath
import platform
import re
import subprocess
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[3]
TOOLS = ROOT / 'experiments/c03'
OUTPUT_ROOT = ROOT.parent / 'VERA-IDS-local-runs/c03'


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=False, allow_nan=False).encode('utf-8')


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def valid_hash(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def safe_relative(value):
    require(isinstance(value, str) and value and '\\' not in value and ':' not in value,
            'Invalid relative artifact path')
    path = PurePosixPath(value)
    require(not path.is_absolute() and '..' not in path.parts, 'Path escapes namespace')
    return path


def output_location(path=OUTPUT_ROOT):
    resolved = Path(path).resolve()
    require(resolved != ROOT and not resolved.is_relative_to(ROOT),
            'Artifact root must be outside repository')
    return resolved


def component_graph(profiles):
    expected = {'xgb_inv_family_clipped', 'rf_inv_family_clipped', 'rf_class_weight_balanced'}
    require(set(profiles) == expected, 'Incomplete candidate universe')
    rf_a, rf_b = profiles['rf_inv_family_clipped'], profiles['rf_class_weight_balanced']
    require(rf_a['stage2_params'] == rf_b['stage2_params'] and
            rf_a['stage2_weight'] == rf_b['stage2_weight'], 'RF Stage-2 sharing mismatch')
    components, references = {}, {}
    for name in sorted(profiles):
        profile = profiles[name]
        require(profile['apply_loao_stage1'] is True, 'Strict LOAO required')
        refs = {}
        for stage, seed in [('stage1', 123), ('stage2', 133)]:
            key = f'{stage}_{name}' if stage == 'stage1' else f'{stage}_{profile["family"]}_shared'
            node = {'stage': stage, 'model_family': profile['family'],
                    'parameters': profile[f'{stage}_params'], 'seed': seed,
                    'weight_mode': profile[f'{stage}_weight'],
                    'input_binding': stage,
                    'constructor': f'protocol_b_grid.build_{stage}_model',
                    'defaults': 'bound_to_historical_source; unresolved until fit-stage review'}
            if key in components:
                require(components[key] == node, 'Unequal component identity cannot share')
            components[key] = node
            refs[stage] = key
        references[name] = refs
    require(len(components) == 5, 'Expected five unique fitted components')
    return {'components': components, 'candidate_references': references,
            'prerequisites': ['preprocessor', 'stage1_inputs', 'stage2_inputs'],
            'fit_order': sorted(components)}


def build_plan():
    # Import only the existing standard-library candidate contract.
    if str(TOOLS) not in sys.path:
        sys.path.insert(0, str(TOOLS))
    import c03_candidate_fit_contract as contract
    head = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT,
                          text=True, capture_output=True, check=True).stdout.strip()
    require(head == contract.PINNED_HEAD, 'Historical Git revision differs')
    policy = TOOLS / 'C03_APPROVED_SELECTOR_POLICY.json'
    require(file_hash(policy) == contract.POLICY_SHA, 'Approved policy differs')
    baseline_path = ROOT / '.release-audit/c03/C03_input_content_manifest.json'
    baseline = json.loads(baseline_path.read_text(encoding='utf-8'))
    scope = {k: v for k, v in baseline.items() if k != 'snapshot_sha256'}
    require(digest(scope) == baseline.get('snapshot_sha256') == contract.BASELINE_SHA,
            'Original snapshot fingerprint differs')
    counts = {'train': 239, 'val': 65, 'test': 65}
    require(len(baseline['data_files']) == 369, 'Incomplete dataset inventory')
    paths = set()
    for entry in baseline['data_files']:
        safe_relative(entry['relative_path'])
        require(entry['relative_path'] not in paths and valid_hash(entry['sha256']),
                'Invalid/duplicate dataset identity')
        require(entry['role'] in counts, 'Unknown dataset role')
        paths.add(entry['relative_path'])
    require({role: sum(f['role'] == role for f in baseline['data_files'])
             for role in counts} == counts, 'Split counts differ')
    # Validate code bytes only. Dataset identities are declarations from the manifest.
    for entry in baseline['code_files']:
        if entry['relative_path'].startswith(('src/', 'config/')):
            require(file_hash(ROOT / entry['relative_path']) == entry['sha256'],
                    'Historical source/config drift')
    relocation_path = TOOLS / 'RELOCATION.json'
    relocation = json.loads(relocation_path.read_text(encoding='utf-8'))
    require(relocation['original_snapshot_sha256'] == contract.BASELINE_SHA,
            'Relocation points to a different baseline')
    for entry in relocation['files']:
        path = ROOT / safe_relative(entry['relocated_path'])
        require(path.resolve().is_relative_to(ROOT), 'Relocation path escapes repository')
        require(file_hash(path) == entry['relocated_sha256'], 'Relocated source drift')
    aggregate = ROOT / contract.AGG
    with aggregate.open(encoding='utf-8-sig', newline='') as stream:
        profiles = contract.candidate_contract(list(csv.DictReader(stream)))['profiles']
    project = tomllib.loads((ROOT / 'pyproject.toml').read_text(encoding='utf-8'))
    require(sys.version_info[:2] == (3, 12), 'Python 3.12 required')
    environment = {'python': platform.python_version(), 'platform': platform.system(), 'packages': {}}
    for spec in project['project']['dependencies']:
        name, expected = spec.split('==')
        installed = version(name)
        require(installed == expected, f'Environment version drift: {name}')
        environment['packages'][name] = installed
    source_paths = [*Path(__file__).parent.glob('*.py'), TOOLS/'engineering/preparation_probe.py', TOOLS/'c03_paths.py',
                    TOOLS/'c03_paired_runner_core.py', TOOLS/'c03_selector_reference.py',
                    ROOT/'src/ids_eval_framework/_native/protocol_b_grid.py',
                    ROOT/'src/ids_eval_framework/src/two_stage_engine.py',
                    *ROOT.glob('tests/test_c03_producer*.py'), ROOT/'uv.lock',
                    ROOT/'pyproject.toml', aggregate]
    sources = {p.relative_to(ROOT).as_posix(): file_hash(p) for p in source_paths}
    return {'schema_version': 1, 'status': 'NO_FIT_PLAN', 'execution_authorized': False,
            'claim_id': 'C03', 'dataset': 'CICIoT2023', 'holdout': 'Botnet',
            'git_revision': head, 'policy_sha256': contract.POLICY_SHA,
            'original_snapshot_sha256': contract.BASELINE_SHA,
            'relocation_record_sha256': file_hash(relocation_path),
            'sources': sources, 'environment': environment,
            'sampling': {'train_cap': 1500000, 'train_seed': 123,
                         'val_cap': 700000, 'val_seed': 124,
                         'algorithm': 'historical sequential buffer mixing'},
            'expected_support': {'stage1_train': 1270025, 'stage2_train': 1201422,
                                 'validation_K': 599278, 'validation_U': 100722},
            'graph': component_graph(profiles),
            'resources': {'status': 'FULL_FIT_UNREVIEWED', 'device': None,
                          'workers': None, 'memory_limit_bytes': None, 'wall_seconds': None},
            'data_verification': 'manifest identity only; no current data-byte or observation reads'}
