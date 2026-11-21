"""C03 synthetic on-disk test-access smoke tests, with actual historical Stage-1.

Put beside c03_pretest_sealing_gate.py, c03_stage1_integration_gate.py,
c03_selector_reference.py, and C03_APPROVED_SELECTOR_POLICY.json at VERA-IDS root.
Run: python -B -m unittest -v c03_runner_disk_boundary_smoke.py

DOES NOT import or read processed datasets, fit models, or write in repo.
All synthetic test CSVs are created under the OS temporary directory and removed.
This verifies *application-layer gated reader sequencing*, not Windows ACLs,
process sandboxing, or a complete model-fitting C03 runner.
"""
from __future__ import annotations

import copy
import csv
import hashlib
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np
from c03_pretest_sealing_gate import (
    LockedTestAccess, POLICY_FILE, compute_synthetic_decisions,
)
from c03_stage1_integration_gate import source_legacy
from c03_selector_reference import macro_f1_fixed


def write_fake_test(destination: Path, count: int = 73) -> str:
    """Create a deliberately tiny synthetic test surface OUTSIDE the repository."""
    if count < 25:
        raise ValueError('Fixture needs each class present')
    with destination.open('w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['row_id', 'true_label', 'stage2_prediction', 'attack_score', 'uncertainty'])
        for i in range(count):
            t = i % 7
            pf = ((i + 3) % 5) + 1
            a = (0.08 + (i % 11) / 15) if t == 0 else (0.45 + (i % 9) / 19)
            u = 0.04 + (i % 13) / 17
            w.writerow([i, t, pf, min(a, .999), u])
    return hashlib.sha256(destination.read_bytes()).hexdigest()


def read_fake_test(path: Path) -> list[dict]:
    with path.open('r', encoding='utf-8', newline='') as f:
        rows = list(csv.DictReader(f))
    if not rows or any(int(r['row_id']) != i for i, r in enumerate(rows)):
        raise RuntimeError('Synthetic test row order mismatch')
    if any(not 0 <= int(r['true_label']) <= 6 for r in rows):
        raise RuntimeError('Invalid synthetic truth labels')
    return rows


def evaluate_fake_test(rows: list[dict], sealed_record: dict) -> dict:
    """Evaluate independent policies on EXACTLY ONE ordered synthetic test surface.

    This is a miniature stand-in, intentionally not the eventual C03 evaluator.
    It only demonstrates sequencing, identical test identity, and sign convention.
    """
    n = len(rows)
    t = np.fromiter((int(r['true_label']) for r in rows), dtype=np.int64, count=n)
    pf = np.fromiter((int(r['stage2_prediction']) for r in rows), dtype=np.int64, count=n)
    a = np.fromiter((float(r['attack_score']) for r in rows), dtype=float, count=n)
    u = np.fromiter((float(r['uncertainty']) for r in rows), dtype=float, count=n)
    identity = hashlib.sha256(','.join(str(r['row_id']) for r in rows).encode('ascii')).hexdigest()
    output: dict = {'test_identity_sha256': identity, 'test_rows': n, 'arms': {}, 'paired_delta': None}
    for arm in ('blind', 'visible'):
        rec = sealed_record['arms'][arm]
        picked = rec['selections']['entropy']
        if picked['status'] != 'feasible':
            output['arms'][arm] = {'status': 'infeasible', 'test_identity_sha256': identity}
            continue
        gate = a >= rec['stage1_threshold']
        threshold = picked['threshold']
        rej = np.zeros(n, dtype=bool) if math.isinf(threshold) else gate & (u >= threshold)
        pred = np.where(rej, 6, np.where(gate, pf, 0))
        support = np.bincount(t, minlength=7)
        predicted = np.bincount(pred, minlength=7)
        tp = np.bincount(t[t == pred], minlength=7)
        score = float(macro_f1_fixed(support, tp, predicted))
        output['arms'][arm] = {'status': 'feasible', 'macro_f1': score,
                               'test_identity_sha256': identity}
    b = output['arms']['blind']
    v = output['arms']['visible']
    if b['status'] == v['status'] == 'feasible':
        output['paired_delta'] = v['macro_f1'] - b['macro_f1']
    return output


class Gate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.grid, cls.engine = source_legacy()  # real installed source, no fitting
        cls.ledger = compute_synthetic_decisions(cls.grid, cls.engine)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='c03-synthetic-test-')
        self.addCleanup(self.temp.cleanup)
        self.testfile = Path(self.temp.name) / 'test_surface_synthetic.csv'
        self.test_digest = write_fake_test(self.testfile)
        self.loader_calls = 0

        def loader():
            self.loader_calls += 1
            return read_fake_test(self.testfile)

        self.guard = LockedTestAccess(loader)

    def test_01_actual_test_file_exists_but_guard_cannot_read_before_seal(self):
        self.assertTrue(self.testfile.is_file())
        with self.assertRaises(PermissionError):
            self.guard.open_test(self.ledger)
        self.assertEqual(self.guard.loads, 0)
        self.assertEqual(self.loader_calls, 0)

    def test_02_both_arms_sealed_before_guarded_file_reader(self):
        seal = self.guard.seal(copy.deepcopy(self.ledger))
        self.assertEqual(len(seal), 64)
        self.assertEqual(self.guard.loads, 0)
        rows = self.guard.open_test(self.ledger)
        self.assertEqual(self.loader_calls, 1)
        self.assertEqual(self.guard.loads, 1)
        self.assertEqual(len(rows), 73)
        self.assertEqual(self.test_digest, hashlib.sha256(self.testfile.read_bytes()).hexdigest())

    def test_03_incomplete_arm_cannot_unlock_physical_file_reader(self):
        incomplete = copy.deepcopy(self.ledger)
        del incomplete['arms']['visible']
        with self.assertRaises(ValueError):
            self.guard.seal(incomplete)
        with self.assertRaises(PermissionError):
            self.guard.open_test(self.ledger)
        self.assertEqual(self.loader_calls, 0)

    def test_04_policy_swap_blocks_sealing_before_file_reader(self):
        wrong = copy.deepcopy(self.ledger)
        wrong['policy_sha256'] = 'a' * 64
        self.assertNotEqual(wrong['policy_sha256'], hashlib.sha256(POLICY_FILE.read_bytes()).hexdigest())
        with self.assertRaises(ValueError):
            self.guard.seal(wrong)
        self.assertEqual(self.loader_calls, 0)

    def test_05_postseal_mutation_blocks_guarded_file_reader(self):
        original = copy.deepcopy(self.ledger)
        self.guard.seal(original)
        changed = copy.deepcopy(original)
        changed['arms']['blind']['stage1_threshold'] += .0001
        with self.assertRaises(PermissionError):
            self.guard.open_test(changed)
        self.assertEqual(self.loader_calls, 0)

    def test_06_two_arm_outcomes_reuse_same_ordered_test_and_sign(self):
        self.guard.seal(self.ledger)
        rows = self.guard.open_test(self.ledger)
        outcome = evaluate_fake_test(rows, self.ledger)
        self.assertEqual(outcome['test_rows'], 73)
        self.assertEqual(outcome['arms']['blind']['test_identity_sha256'],
                         outcome['arms']['visible']['test_identity_sha256'])
        for arm in ('blind', 'visible'):
            if outcome['arms'][arm]['status'] == 'feasible':
                self.assertTrue(0 <= outcome['arms'][arm]['macro_f1'] <= 1)
        b, v = outcome['arms']['blind'], outcome['arms']['visible']
        if b['status'] == v['status'] == 'feasible':
            self.assertAlmostEqual(outcome['paired_delta'], v['macro_f1'] - b['macro_f1'])
        else:
            self.assertIsNone(outcome['paired_delta'])

    def test_07_deterministic_synthetic_selection_record(self):
        second = compute_synthetic_decisions(self.grid, self.engine)
        self.assertEqual(self.ledger, second)


if __name__ == '__main__':
    unittest.main(verbosity=2)
