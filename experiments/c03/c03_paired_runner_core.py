#!/usr/bin/env python3
"""C03 paired pilot: validation-selector and pretest sealing production CORE.

The intended full runner will supply validation scores from three common candidate
fits. This module consumes ONLY validation predictions and sealed provenance.
It neither trains, preprocesses real data, reads/accepts real test paths, nor
allows actual experimental evaluation. The only executable CLI command is
`--preflight`, which checks the approved local policy, historical profiles,
input manifest AND the pinned source module's Stage1 interface without fitting.

Run at VERA-IDS root: python -B c03_paired_runner_core.py --preflight

This code is new/prospective, not the legacy historical run_one_combo(), which
improperly opens test before selecting thresholds. Pinned to 2026-10-25
approved C03 selector. No repository modification or model fit is authorized.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass
import hashlib
import importlib
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Callable

import numpy as np
from c03_selector_reference import (ValidationScores, choose_profile,
    select_global, select_family)
from c03_candidate_fit_contract import (
    AGG, PINNED_HEAD, POLICY_SHA, BASELINE_SHA, EXPECTED,
    candidate_contract,
)

from c03_paths import ROOT, POLICY as POLICY_PATH, ORIGINAL_SNAPSHOT as TRAIN_SNAPSHOT
FAMILIES = ('BruteForce', 'DDoS', 'DoS', 'Other', 'Scan/Recon')
LABELS = ('Benign',) + FAMILIES + ('Unknown',)
STAGE1_LABELS = ('Benign',) + FAMILIES + ('Botnet',)
METHODS = ('max_confidence', 'margin', 'entropy', 'family_conditional')
PRIMARY_BUDGET = 0.03
SEED = 123


def require(condition: bool, message: str):
    if not condition:
        raise RuntimeError('C03 BLOCKED: ' + message)


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as fh:
        for block in iter(lambda: fh.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def strict_json(value) -> bytes:
    def norm(v):
        if isinstance(v, dict):
            return {str(k): norm(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [norm(x) for x in v]
        if isinstance(v, np.generic):
            return norm(v.item())
        if isinstance(v, float):
            if math.isnan(v):
                raise ValueError('NaN not permitted in sealed decisions')
            if math.isinf(v):
                return '+inf' if v > 0 else '-inf'
            return v
        if isinstance(v, (int, str, bool)) or v is None:
            return v
        raise TypeError(f'Unsupported decision type: {type(v)}')
    return json.dumps(norm(value), sort_keys=True, separators=(',', ':'),
                      ensure_ascii=False, allow_nan=False).encode('utf-8')


def sha_record(value) -> str:
    return sha_bytes(strict_json(value))


def array_fingerprint(array: np.ndarray) -> str:
    a = np.ascontiguousarray(array)
    h = hashlib.sha256()
    h.update(str(a.dtype).encode('ascii'))
    h.update(str(a.shape).encode('ascii'))
    h.update(a.tobytes(order='C'))
    return h.hexdigest()


def history_modules():
    """Import checked historical code only. Never call run_one_combo()."""
    source = ROOT / 'src' / 'ids_eval_framework'
    require((source / '_native' / 'protocol_b_grid.py').is_file(),
            'run from VERA-IDS root (historical grid source missing)')
    spath = str(ROOT / 'src')
    if spath not in sys.path:
        sys.path.insert(0, spath)
    grid = importlib.import_module('ids_eval_framework._native.protocol_b_grid')
    engine = importlib.import_module('ids_eval_framework.src.two_stage_engine')
    import inspect
    for module in (grid, engine):
        require(Path(inspect.getfile(module)).resolve().is_relative_to(source.resolve()),
                'historical module resolves outside pinned repo')
    return grid, engine


def check_policy_and_profiles(*, snapshot_path=TRAIN_SNAPSHOT,
                              check_git=True, require_snapshot=True):
    """Read only: policy/provenance/candidate definitions, never a test dataset."""
    if check_git:
        git = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT,
                             text=True, capture_output=True, check=True).stdout.strip()
        require(git == PINNED_HEAD, f'Git HEAD {git} differs from approved {PINNED_HEAD}')
    require(sha_file(POLICY_PATH) == POLICY_SHA, 'approved policy SHA-256 differs')
    policy = json.loads(POLICY_PATH.read_text(encoding='utf-8'))
    require(policy['claim_id'] == 'C03' and policy['policy_status'] ==
            'SCIENTIFIC_POLICY_APPROVED__INTEGRATION_AND_EXECUTION_NOT_FROZEN',
            'policy identity/status mismatch')
    require(tuple(policy['common_fit']['candidate_profiles']) == tuple(EXPECTED),
            'candidate profile order differs from approved policy')
    require(tuple(policy['labels_in_order']) == LABELS, 'label order mismatch')
    r = policy['rejector_selection']
    require(tuple(r['methods']) == METHODS and
            PRIMARY_BUDGET == r['primary_budget'] and
            tuple(r['budgets']) == (0.01, 0.03, 0.05, 0.10),
            'rejector methods/budget drift')
    require(r['budget_scope'] == 'K-known-only validation rows in BOTH selector arms',
            'known-only feasibility denominator drift')
    require(r['benign_to_family_fp_ceiling'] == .02 and
            r['benign_rejection_ceiling'] == .10 and
            r['false_unknown_K_ceiling'] == .05 and
            r['false_unknown_K_attacks_ceiling'] == .10,
            'constraints differ from approved reference selector')
    require(policy['first_pilot']['expected_known_validation_count'] == 599278 and
            policy['first_pilot']['expected_unknown_validation_count'] == 100722,
            'sample support expectation drift')
    import csv
    with (ROOT / AGG).open('r', encoding='utf-8-sig', newline='') as fh:
        profiles = candidate_contract(list(csv.DictReader(fh)))['profiles']
    if require_snapshot:
        manifest = json.loads(Path(snapshot_path).read_text(encoding='utf-8'))
        require(manifest.get('snapshot_sha256') == BASELINE_SHA, 'snapshot fingerprint mismatch')
        require(len(manifest.get('data_files', [])) == 369, 'data-file inventory incomplete')
        # Trust ONLY the user's previously verified manifest; byte re-hash is a
        # separate pre-execution gate done with c03_input_content_snapshot.py --check.
    return policy, profiles


@dataclass(frozen=True)
class CandidateScores:
    """Shared fitted-model scores on the SAME ordered validation observations."""
    attack_prob: np.ndarray    # N, probabilities from Stage 1
    family_prob: np.ndarray    # N x 5 ordered FAMILIES probabilities from Stage 2


def validate_prediction_bundle(truth, candidate_scores, *, expected_counts=None):
    """Reject missing, misordered, malformed or unsupported validation scores.

    These are genuinely precomputed validation-only arrays. This function cannot
    establish their origin; the future fit worker MUST attach a preprocessor,
    sampled-row-index and fitted-model digest, verified separately.
    """
    t = np.asarray(truth)
    require(t.ndim == 1 and t.size > 0 and np.issubdtype(t.dtype, np.integer),
            'validation truth must be nonempty integer vector')
    require(np.isin(t, np.arange(7)).all(), 'unapproved validation label')
    k = t != 6
    require((t == 0).sum() > 0 and ((t >= 1) & (t <= 5)).sum() > 0 and
            (t == 6).sum() > 0, 'benign/known attack/heldout support missing')
    if expected_counts is not None:
        require((int(k.sum()), int((~k).sum())) == tuple(expected_counts),
                'unexpected cap-sampled K/U support')
    for i in range(1, 6):
        require((t == i).sum() >= 200, f'missing family support: {LABELS[i]}')
    require(set(candidate_scores) == set(EXPECTED), 'candidate score universe incomplete')
    score_digest = {}
    for name in EXPECTED:
        item = candidate_scores[name]
        a, p = np.asarray(item.attack_prob), np.asarray(item.family_prob)
        require(a.shape == (len(t),) and p.shape == (len(t), 5),
                f'{name} incorrect validation score shape')
        require(np.isfinite(a).all() and np.isfinite(p).all(),
                f'{name} nonfinite validation scores')
        require(((a >= 0) & (a <= 1)).all() and ((p >= 0) & (p <= 1)).all(),
                f'{name} probabilities out of [0,1]')
        require(np.allclose(p.sum(axis=1), 1.0, atol=1e-8, rtol=0),
                f'{name} Stage2 family probabilities not normalized')
        score_digest[name] = sha_record({'attack': array_fingerprint(a),
                                        'family': array_fingerprint(p)})
    return t.astype(np.int8, copy=False), k, score_digest


def select_stage1(grid, engine, truth, prob, arm):
    """Validation-only Stage1 adapter; no reference to test files or test paths."""
    take = truth != 6 if arm == 'blind' else np.ones(len(truth), dtype=bool)
    y_stage1 = (truth[take] != 0).astype(np.int8)
    y_family = np.asarray(STAGE1_LABELS, dtype=object)[truth[take]]
    threshold, best, table = grid.select_stage1_threshold(
        engine, y_stage1, y_family, prob[take]
    )
    require(math.isfinite(float(threshold)) and 0 <= threshold <= 1,
            'unacceptable Stage1 threshold')
    require(len(table) == 4, 'historical four Stage1 search configurations changed')
    return float(threshold), best


def select_both_arms(truth, scores, *, grid, engine, policy_sha=POLICY_SHA,
                     source_inventory_sha=BASELINE_SHA, expected_counts=None):
    """Full four-method/four-budget selector with identical K cost denominators.

    Inputs are VALIDATION-only score arrays. Not a model-training runner. Does
    not accept a test loader or test path. No score files are read here.
    """
    require(policy_sha == sha_file(POLICY_PATH) == POLICY_SHA, 'policy changed')
    require(source_inventory_sha == BASELINE_SHA, 'unrecognized dataset snapshot identity')
    t, k, digests = validate_prediction_bundle(truth, scores,
                                               expected_counts=expected_counts)
    known_ids = np.flatnonzero(k)
    # `np.arange` is a reproducible position/index fingerprint, NOT a unique
    # raw flow ID. The future feeder must furnish original deterministic sample
    # provenance and verify it separately.
    candidates = {name: (np.argmax(scores[name].family_prob, axis=1)+1,
                         scores[name].attack_prob)
                  for name in EXPECTED}
    decision = {
        'schema_version': 1,
        'status': 'VALIDATION_ONLY_DECISIONS_NOT_YET_SEALED',
        'policy_sha256': POLICY_SHA,
        'source_inventory_sha256': BASELINE_SHA,
        'validation_truth_sha256': array_fingerprint(t),
        'known_position_sha256': array_fingerprint(known_ids),
        'candidate_validation_score_sha256': digests,
        'arms': {},
    }
    for arm in ('blind', 'visible'):
        mask = k if arm == 'blind' else np.ones(len(t), dtype=bool)
        profile = choose_profile(t, mask, candidates)
        p1 = np.asarray(scores[profile].attack_prob, dtype=np.float64)
        p2 = np.asarray(scores[profile].family_prob, dtype=np.float64)
        threshold, stage1_meta = select_stage1(grid, engine, t, p1, arm)
        gated = p1[mask] >= threshold
        q = np.clip(p2[mask], 1e-12, 1.0)
        # Original Phase3 scorer normalizes clipped probabilities before
        # calculating top2 margin, max confidence and normalized entropy.
        q = q / q.sum(axis=1, keepdims=True)
        pf = np.argmax(q, axis=1).astype(np.int8) + 1
        pmax = np.max(q, axis=1)
        part = np.partition(q, kth=-2, axis=1)
        margin = part[:, -1] - part[:, -2]
        entropy = -(q * np.log(np.clip(q, 1e-12, 1))).sum(axis=1) / np.log(5)
        target = t[mask]
        truep = np.zeros(len(target), dtype=np.float64)
        known = (target >= 1) & (target <= 5)
        known_indices = np.flatnonzero(known)
        truep[known] = q[known_indices, target[known_indices] - 1]
        row = {
            'candidate_profile': profile,
            'stage1_threshold': threshold,
            'stage1_search_best': {str(key): float(value) if isinstance(value,(int,float,np.integer,np.floating)) and not isinstance(value,bool) else str(value)
                                   for key,value in stage1_meta.items() if key in ('thr','objective','fpr','tpr','target_fpr','objective_mode')},
            'primary_budget': PRIMARY_BUDGET,
            'known_validation_count': int(k.sum()),
            'visible_validation_count': int(mask.sum()),
            'selections': {},
        }
        for budget in (0.01, 0.03, 0.05, 0.10):
            bkey = f'{budget:.2f}'
            row['selections'][bkey] = {}
            for method in METHODS:
                uncertainty = {'max_confidence': 1.0-pmax,
                               'margin': 1.0-margin,
                               'entropy': entropy}.get(method)
                if uncertainty is None:  # family_conditional uses pmax/truep
                    uncertainty = np.zeros(len(target), dtype=np.float64)
                val = ValidationScores(target, pf, gated, uncertainty, pmax, truep)
                selected = select_family(val, budget) if method == 'family_conditional' else select_global(val,budget,method)
                row['selections'][bkey][method] = ({'status':'infeasible'} if selected is None
                                                    else asdict(selected))
        decision['arms'][arm] = row
    return decision


class FrozenSelection:
    """Immutable-after-seal JSON decision contract, no test path in this class."""
    def __init__(self, expected_policy_sha=POLICY_SHA):
        self._digest = None
        self._ledger = None
        self._policy_hash = expected_policy_sha

    def seal(self, decision):
        require(self._digest is None, 'attempted reseal')
        require(decision.get('status') == 'VALIDATION_ONLY_DECISIONS_NOT_YET_SEALED',
                'wrong selection ledger status')
        require(decision.get('policy_sha256') == self._policy_hash == sha_file(POLICY_PATH),
                'policy mismatch during seal')
        require(decision.get('source_inventory_sha256') == BASELINE_SHA,
                'snapshot identity mismatch during seal')
        require(set(decision.get('arms',{})) == {'blind','visible'},
                'both arms must be selected')
        for arm in ('blind','visible'):
            row=decision['arms'][arm]
            require(row['candidate_profile'] in EXPECTED and 0 <= row['stage1_threshold'] <= 1,
                    'unapproved selected profile/threshold')
            require(row['primary_budget'] == PRIMARY_BUDGET,
                    'wrong primary budget')
            require(set(row['selections']) == {'0.01','0.03','0.05','0.10'},
                    'missing budget selections')
            for per_method in row['selections'].values():
                require(set(per_method) == set(METHODS), 'missing method selection')
                require(all(x.get('status') in {'feasible','infeasible'} for x in per_method.values()),
                        'unresolved method selection status')
        b, v = decision['arms']['blind'], decision['arms']['visible']
        require(b['known_validation_count'] == v['known_validation_count'] > 0,
                'K population differs between arms')
        require(b['visible_validation_count'] == b['known_validation_count'] and
                v['visible_validation_count'] > b['visible_validation_count'],
                'not blind K vs visible K+U')
        ledger = copy.deepcopy(decision)
        ledger['status'] = 'SEALED_VALIDATION_SELECTION'
        self._ledger = ledger
        self._digest = sha_record(ledger)
        return self._digest

    def verify(self, ledger, seal_sha):
        require(self._digest is not None, 'selection not sealed')
        require(seal_sha == self._digest == sha_record(ledger),
                'decision ledger was modified after seal')
        return True

    def export(self):
        require(self._digest is not None, 'selection not sealed')
        return copy.deepcopy(self._ledger), self._digest


class SyntheticGuardedReader:
    """An explicit test-boundary example; REAL test reading is NOT implemented.

    When a separately approved full runner is built, its evaluator must be a
    distinct stage and must verify the on-disk sealed decisions before it can
    construct or open any test-dataset paths. This is NOT an OS sandbox.
    """
    def __init__(self, sealed: FrozenSelection, fake_loader: Callable):
        self.sealed = sealed
        self.fake_loader = fake_loader
        self.calls = 0

    def open(self, ledger, seal_sha):
        self.sealed.verify(ledger, seal_sha)
        self.calls += 1
        return self.fake_loader()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preflight', action='store_true', required=True,
                        help='Read-only policy, historical profile and code gate; no fits')
    parser.add_argument('--snapshot', type=Path, default=TRAIN_SNAPSHOT)
    parser.add_argument('--verify-bytes', action='store_true',
                        help='Rehash every source file in the existing snapshot (no observation decoding)')
    args = parser.parse_args()
    policy, profiles = check_policy_and_profiles(snapshot_path=args.snapshot)
    if args.verify_bytes:
        import c03_input_content_snapshot as snapshot
        previous = json.loads(args.snapshot.read_text(encoding='utf-8'))
        current = snapshot.inventory(ROOT)
        snapshot.verify_relocation(previous, current)
    grid, engine = history_modules()
    require(callable(grid.select_stage1_threshold) and callable(engine.choose_thr_high_family_aware),
            'missing historical Stage1 selector')
    require(grid.CFG['stage1_threshold_search'] == {
        'target_fpr_list':[.01,.02], 'min_family_support_list':[100],
        'objective_mode_list':['min_family_recall','p10_family_recall'],
        'p10_quantile_list':[.10], 'sweep_points':40,
        'selection_tiebreak':'objective_then_tpr',
    }, 'historical Stage1 search configuration drift')
    require(profiles['rf_inv_family_clipped']['stage2_params'] ==
            profiles['rf_class_weight_balanced']['stage2_params'],
            'RF Stage2 cannot share exact config')
    print('=== C03 PAIRED-RUNNER VALIDATION CORE: READ-ONLY PREFLIGHT ===')
    print('Approved policy hash:', POLICY_SHA)
    print('Historical Git commit:', PINNED_HEAD)
    print('Source inventory manifest fingerprint:', BASELINE_SHA)
    if args.verify_bytes:
        print('Current complete source-content SHA256 recheck: PASSED')
    else:
        print('Current source bytes: NOT rehashed; use --verify-bytes before execution planning')
    print('Historical candidate profiles:', ', '.join(EXPECTED))
    print('RF Stage2 shared model: configuration verified; actual input tensor not yet built')
    print('Stage1 historical grid: callable, four-config pinned search verified')
    print('Core implements: K/K+U candidate selection, Stage1 thresholds, 4 rejectors x 4 budgets, seal contract')
    print('NOT IMPLEMENTED IN THIS FILE: real fitting, checkpointed score producer, real test evaluator, memory-abort')
    print('EXECUTION: NO FIT API/CLI; no real train/val/test rows read or written')
    print('RESULT: PASS production selector/seal core preflight; not end-to-end run authorization')


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print('BLOCKED:', exc, file=sys.stderr)
        sys.exit(2)
