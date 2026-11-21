"""Synthetic integration acceptance tests for the C03 paired runner CORE.

Run from VERA-IDS root: python -B -m unittest -v test_c03_paired_runner_core.py
No traffic observations, model fitting, source file writes or real test reader.
Tests exercise the same production core that will accept future fitted scores.
"""
from __future__ import annotations

import copy
import inspect
import unittest
import numpy as np

import c03_paired_runner_core as core


class StrictStage1StandIn:
    """Temporary Stage1 implementation used to test production core interfaces.

    ACTUAL historical Stage1 import is checked by c03_paired_runner_core.py
    --preflight, not by this stand-in.
    """
    calls = 0

    def select_stage1_threshold(self, engine, y1, y2, p):
        self.calls += 1
        assert len(y1) == len(y2) == len(p)
        assert np.isin(y1, [0,1]).all()
        assert ('Botnet' in y2) == (len(y1) == 2575)
        return .40, {'thr': .40, 'fpr': .01, 'objective': .87}, [1,2,3,4]


def fixture():
    rng = np.random.RandomState(213)
    t = np.concatenate([np.zeros(1200, dtype=np.int8),
                        *[np.full(225, i, dtype=np.int8) for i in range(1, 6)],
                        np.full(250, 6, dtype=np.int8)])
    scores = {}
    for name in core.EXPECTED:
        p1 = np.clip(np.where(t==0, .1, .76) + rng.normal(0, .12, len(t)), .0001,.9999)
        p2 = rng.uniform(.02, .06, (len(t),5))
        for i in range(1,6):
            mask = t == i
            p2[mask, i-1] = rng.uniform(.4,.95,int(mask.sum()))
        p2[t==6] = rng.dirichlet(np.ones(5), size=int((t==6).sum()))
        p2[t==0] = rng.dirichlet(np.ones(5), size=int((t==0).sum()))
        p2 /= p2.sum(axis=1, keepdims=True)
        scores[name] = core.CandidateScores(p1,p2)
    return t, scores


class PairedCore(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.truth, cls.scores = fixture()
        cls.grid = StrictStage1StandIn()
        cls.ledger = core.select_both_arms(cls.truth, cls.scores, grid=cls.grid,
                                           engine=object(), expected_counts=(2325,250))

    def test_01_validates_complete_three_candidate_and_both_arms(self):
        self.assertEqual(self.grid.calls, 2)
        self.assertEqual(set(self.ledger['candidate_validation_score_sha256']),set(core.EXPECTED))
        self.assertEqual(self.ledger['arms']['blind']['known_validation_count'],2325)
        self.assertEqual(self.ledger['arms']['blind']['visible_validation_count'],2325)
        self.assertEqual(self.ledger['arms']['visible']['visible_validation_count'],2575)

    def test_02_four_methods_each_of_four_budgets(self):
        for arm in ('blind','visible'):
            row = self.ledger['arms'][arm]
            self.assertIn(row['candidate_profile'], core.EXPECTED)
            self.assertAlmostEqual(row['stage1_threshold'], .4)
            self.assertEqual(set(row['selections']), {'0.01','0.03','0.05','0.10'})
            for options in row['selections'].values():
                self.assertEqual(set(options), set(core.METHODS))
                self.assertTrue(all(x['status'] in ('feasible','infeasible') for x in options.values()))

    def test_03_hidden_U_corruption_cannot_change_blind_selection(self):
        sc = copy.deepcopy(self.scores)
        a = sc['xgb_inv_family_clipped']
        pa = a.attack_prob.copy()
        pa[self.truth==6] *= .8
        sc['xgb_inv_family_clipped'] = core.CandidateScores(pa, a.family_prob)
        changed = core.select_both_arms(self.truth,sc,grid=StrictStage1StandIn(),
                                        engine=object(),expected_counts=(2325,250))
        self.assertEqual(changed['arms']['blind'],self.ledger['arms']['blind'])
        self.assertNotEqual(changed['candidate_validation_score_sha256'],
                            self.ledger['candidate_validation_score_sha256'])

    def test_04_seal_prevents_unsealed_and_mutated_test_reader(self):
        sealed = core.FrozenSelection()
        loadcalls=[]
        guarded = core.SyntheticGuardedReader(sealed,lambda:loadcalls.append(1) or 'synthetic test')
        with self.assertRaises(RuntimeError):
            guarded.open(self.ledger,'0'*64)
        sealhash = sealed.seal(self.ledger)
        record, digest = sealed.export()
        self.assertEqual(digest,sealhash)
        self.assertEqual(guarded.open(record,digest),'synthetic test')
        self.assertEqual(len(loadcalls),1)
        changed=copy.deepcopy(record)
        changed['arms']['blind']['stage1_threshold'] += .001
        with self.assertRaises(RuntimeError):
            guarded.open(changed,digest)
        self.assertEqual(len(loadcalls),1)
        with self.assertRaises(RuntimeError):
            sealed.seal(self.ledger)

    def test_05_incomplete_selection_cannot_seal(self):
        sel=core.FrozenSelection()
        d=copy.deepcopy(self.ledger)
        del d['arms']['visible']
        with self.assertRaises(RuntimeError):sel.seal(d)
        d=copy.deepcopy(self.ledger)
        del d['arms']['blind']['selections']['0.03']['entropy']
        with self.assertRaises(RuntimeError):sel.seal(d)

    def test_06_wrong_policy_blocks_selection_and_seal(self):
        with self.assertRaises(RuntimeError):
            core.select_both_arms(self.truth,self.scores,grid=self.grid,engine=object(),
                                  policy_sha='0'*64)
        d=copy.deepcopy(self.ledger);d['policy_sha256']='0'*64
        with self.assertRaises(RuntimeError):core.FrozenSelection().seal(d)

    def test_07_bad_proba_or_incomplete_profile_blocks(self):
        for mode in ('nan','nonnormal','absent'):
            sc=copy.deepcopy(self.scores)
            if mode=='absent':
                del sc['xgb_inv_family_clipped']
            else:
                v=sc['rf_inv_family_clipped'];p=v.family_prob.copy()
                p[0,0] = np.nan if mode=='nan' else 0.98
                sc['rf_inv_family_clipped']=core.CandidateScores(v.attack_prob,p)
            with self.subTest(mode=mode), self.assertRaises(RuntimeError):
                core.validate_prediction_bundle(self.truth,sc)

    def test_08_wrong_sample_counts_block(self):
        with self.assertRaises(RuntimeError):
            core.validate_prediction_bundle(self.truth,self.scores,expected_counts=(2325,251))

    def test_09_selector_receives_no_test_path_or_loader(self):
        names=set(inspect.signature(core.select_both_arms).parameters)
        self.assertTrue({'truth','scores','grid','engine'} <= names)
        self.assertFalse(any('test' in x for x in names))
        self.assertFalse(any('test' in x for x in inspect.signature(core.select_stage1).parameters))


    def test_10_actual_stage1_with_production_core_when_repo_available(self):
        # Unlike the stand-in tests, this imports the real, pinned VERA-IDS
        # historical Stage1 chooser through the PRODUCTION selector adapter.
        if not (core.ROOT / 'src' / 'ids_eval_framework' / '_native' /
                'protocol_b_grid.py').is_file():
            self.skipTest('real VERA-IDS source absent in isolated test environment')
        grid, engine = core.history_modules()
        ledger = core.select_both_arms(self.truth, self.scores, grid=grid,
                                       engine=engine, expected_counts=(2325,250))
        for arm in ('blind','visible'):
            self.assertTrue(0 <= ledger['arms'][arm]['stage1_threshold'] <= 1)
            self.assertEqual(set(ledger['arms'][arm]['selections']),
                             {'0.01','0.03','0.05','0.10'})
        self.assertEqual(ledger['arms']['blind']['known_validation_count'],
                         ledger['arms']['visible']['known_validation_count'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
