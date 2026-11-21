"""C03 one-cell model-free Stage-1 integration and validation-isolation gate.

From VERA-IDS root (with c03_selector_reference.py also at root):
    python -B -m unittest -v c03_stage1_integration_gate.py

READ ONLY. Imports the REAL repository Stage-1 grid wrapper and historical
family-aware helper. It uses synthetic labels/scores, never loads test data,
fits a preprocessor/model, launches a runner, or writes repository files.

Passing this suite validates only these bounded integration behaviors. It does
not seal a future end-to-end experimental runner or authenticate raw data.
"""
from __future__ import annotations

import copy
import importlib
import inspect
from pathlib import Path
import sys
import unittest

import numpy as np

from c03_paths import ROOT
SOURCE = ROOT / "src" / "ids_eval_framework"
FAMILIES = ("BruteForce", "DDoS", "DoS", "Other", "Scan/Recon")
ALL_LABELS = ("Benign",) + FAMILIES + ("Botnet",)
SEARCH = {
    "target_fpr_list": [0.01, 0.02],
    "min_family_support_list": [100],
    "objective_mode_list": ["min_family_recall", "p10_family_recall"],
    "p10_quantile_list": [0.10],
    "sweep_points": 40,
    "selection_tiebreak": "objective_then_tpr",
}


def make_validation(seed: int = 671):
    """Synthetic ordered validation rows: K benign/5 attacks plus U Botnet."""
    rng = np.random.default_rng(seed)
    truth = np.concatenate((np.zeros(1200, dtype=np.int8),
                            *(np.full(225, idx, dtype=np.int8) for idx in range(1, 6)),
                            np.full(250, 6, dtype=np.int8)))
    scores = np.concatenate((rng.uniform(.01, .42, 1200),
                             rng.uniform(.35, .97, 5 * 225),
                             rng.uniform(.08, .95, 250)))
    return truth, scores


def source_legacy():
    srcdir = ROOT / "src"
    if not SOURCE.is_dir():
        raise RuntimeError(f"Run from VERA-IDS root: source directory missing {SOURCE}")
    if str(srcdir) not in sys.path:
        sys.path.insert(0, str(srcdir))
    grid = importlib.import_module("ids_eval_framework._native.protocol_b_grid")
    engine = importlib.import_module("ids_eval_framework.src.two_stage_engine")
    for mod in (grid, engine):
        location = Path(inspect.getfile(mod)).resolve()
        if not location.is_relative_to(SOURCE.resolve()):
            raise RuntimeError(f"Unexpected historical-source module: {mod.__name__}: {location}")
    return grid, engine


def stage1_selection(grid, helper, truth, scores, view: str):
    """An isolated *validation-only* adapter, no test-data parameter.

    The FULL validation label vector carries underlying Botnet identity for U;
    unknown is not relabeled Benign or dropped in the visible arm.
    """
    if view not in ("blind", "visible"):
        raise ValueError(f"Unapproved selection arm {view}")
    t = np.asarray(truth)
    p = np.asarray(scores, dtype=np.float64)
    if t.ndim != 1 or p.ndim != 1 or len(t) != len(p) or not len(t):
        raise ValueError("Validation arrays must be nonempty aligned one-dimensional vectors")
    if not np.issubdtype(t.dtype, np.integer) or np.any((t < 0) | (t > 6)):
        raise ValueError("Unknown validation class")
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("Invalid Stage-1 attack probability")
    known = t != 6
    if not np.any(t == 6) or not np.any(t == 0):
        raise ValueError("Synthetic validation needs held-out and benign support")
    take = known if view == "blind" else np.ones(len(t), dtype=bool)
    # Stage1 attack is 1 for every represented attack and Botnet H.
    y1 = (t[take] != 0).astype(np.int64)
    # Stage1 threshold's family-aware objective needs NATIVE family names.
    y2 = np.asarray(ALL_LABELS, dtype=object)[t[take]]
    return grid.select_stage1_threshold(helper, y1, y2, p[take])


class SpyHelper:
    """Delegate to the actual historical helper while recording its inputs."""
    def __init__(self, delegate):
        self.delegate = delegate
        self.calls = []

    def choose_thr_high_family_aware(self, y1, y2, scores, *args):
        self.calls.append((np.asarray(y1).copy(), np.asarray(y2).copy(),
                           np.asarray(scores).copy(), tuple(args)))
        return self.delegate.choose_thr_high_family_aware(y1, y2, scores, *args)


class Integration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.grid, cls.engine = source_legacy()

    def test_01_actual_history_loader_and_search_policy(self):
        self.assertEqual(self.grid.CFG["stage1_threshold_search"], SEARCH)
        self.assertTrue(callable(self.engine.choose_thr_high_family_aware))
        self.assertEqual(len(inspect.signature(self.grid.select_stage1_threshold).parameters), 4)
        self.assertIn("helper.choose_thr_high_family_aware", inspect.getsource(self.grid.select_stage1_threshold))
        # No separate test-score input in the adapter or original grid selector.
        self.assertNotIn("test", inspect.signature(stage1_selection).parameters)
        self.assertNotIn("test", inspect.signature(self.grid.select_stage1_threshold).parameters)

    def test_02_actual_stage1_search_k_and_k_plus_u(self):
        t, p = make_validation()
        for arm, n in (("blind", 2325), ("visible", 2575)):
            with self.subTest(arm=arm):
                threshold, chosen, table = stage1_selection(self.grid, self.engine, t, p, arm)
                self.assertTrue(np.isfinite(threshold))
                self.assertTrue(0 <= threshold <= 1)
                self.assertEqual(len(table), 4)  # two target FPRs x two recall objectives
                self.assertIn("thr", chosen)
                self.assertTrue(np.all(np.isfinite(table["thr"].to_numpy(dtype=float))))
                self.assertAlmostEqual(chosen["thr"], threshold)
                # Source side effects on input data would break arm symmetry.
                self.assertEqual(len(t), 2575)
                self.assertEqual(n, (t != 6).sum() if arm == "blind" else len(t))

    def test_03_true_heldout_mask_and_supervised_visible_labels(self):
        t, p = make_validation()
        spy = SpyHelper(self.engine)
        stage1_selection(self.grid, spy, t, p, "blind")
        self.assertEqual(len(spy.calls), 4)
        self.assertTrue(all(len(call[0]) == 2325 for call in spy.calls))
        self.assertTrue(all("Botnet" not in set(call[1]) for call in spy.calls))
        self.assertTrue(all(int(call[0].sum()) == 1125 for call in spy.calls))
        self.assertTrue(all(int((call[0] == 0).sum()) == 1200 for call in spy.calls))
        spy.calls.clear()
        stage1_selection(self.grid, spy, t, p, "visible")
        self.assertEqual(len(spy.calls), 4)
        self.assertTrue(all(len(call[0]) == 2575 for call in spy.calls))
        self.assertTrue(all(int((call[1] == "Botnet").sum()) == 250 for call in spy.calls))
        self.assertTrue(all(int(call[0].sum()) == 1375 for call in spy.calls))
        self.assertTrue(all(int((call[0] == 0).sum()) == 1200 for call in spy.calls))

    def test_04_poison_hidden_scores_blind_unchanged_and_visible_exposed(self):
        t, p = make_validation()
        changed = p.copy()
        changed[t == 6] = np.linspace(0.0001, .9999, int((t == 6).sum()))
        a = SpyHelper(self.engine)
        b = SpyHelper(self.engine)
        v1 = stage1_selection(self.grid, a, t, p, "blind")
        v2 = stage1_selection(self.grid, b, t, changed, "blind")
        self.assertEqual(v1[0], v2[0])
        self.assertEqual(v1[1], v2[1])
        self.assertTrue(v1[2].equals(v2[2]))
        self.assertEqual(len(a.calls), len(b.calls))
        for ca, cb in zip(a.calls, b.calls):
            np.testing.assert_array_equal(ca[2], cb[2])
        a.calls.clear(); b.calls.clear()
        stage1_selection(self.grid, a, t, p, "visible")
        stage1_selection(self.grid, b, t, changed, "visible")
        self.assertEqual(len(a.calls), len(b.calls))
        self.assertTrue(any(not np.array_equal(ca[2],cb[2]) for ca,cb in zip(a.calls,b.calls)))
        for ca, cb in zip(a.calls, b.calls):
            np.testing.assert_array_equal(ca[2][t != 6], cb[2][t != 6])

    def test_05_known_validation_never_perturbed_by_visibility(self):
        t, p = make_validation()
        spy1 = SpyHelper(self.engine); spy2 = SpyHelper(self.engine)
        stage1_selection(self.grid, spy1, t, p, "blind")
        stage1_selection(self.grid, spy2, t, p, "visible")
        self.assertEqual(len(spy1.calls), len(spy2.calls))
        k = t != 6
        for blind, visible in zip(spy1.calls,spy2.calls):
            np.testing.assert_array_equal(blind[0],visible[0][k])
            np.testing.assert_array_equal(blind[1],visible[1][k])
            np.testing.assert_array_equal(blind[2],visible[2][k])
            self.assertEqual(blind[3], visible[3])

    def test_06_invalid_arm_and_invalid_probabilities_block(self):
        t, p = make_validation()
        for arm in ("both", "test", "unknown"):
            with self.assertRaises(ValueError):
                stage1_selection(self.grid,self.engine,t,p,arm)
        for broken in (p[:-1], np.r_[p[:-1],np.nan], np.r_[p[:-1],1.01]):
            with self.assertRaises(ValueError):
                stage1_selection(self.grid,self.engine,t,broken,"blind")
        with self.assertRaises(ValueError):
            stage1_selection(self.grid,self.engine,np.ones(len(p)),p,"blind")

    def test_07_same_inputs_same_legacy_selection(self):
        t, p = make_validation(817)
        for arm in ("blind","visible"):
            a = stage1_selection(self.grid,self.engine,t,p,arm)
            b = stage1_selection(self.grid,self.engine,t,p,arm)
            self.assertEqual(a[0],b[0]);self.assertEqual(a[1],b[1]);self.assertTrue(a[2].equals(b[2]))

    def test_08_full_candidate_profile_selection_is_independent_of_hidden_stage2_predictions(self):
        from c03_selector_reference import choose_profile
        # Every candidate uses the same exact full-validation truth vector.
        t = np.array([0,0,1,2,3,4,5,6,6])
        stage2=np.array([1,1,1,2,3,4,5,1,1])
        data={"rf_A":(stage2,np.array([.15,.19,.8,.8,.8,.8,.8,.01,.01])),
              "rf_B":(stage2,np.array([.11,.21,.5,.5,.5,.5,.5,.99,.99])),
              "xgb_C":(np.array([1,1,2,2,3,4,5,1,2]),np.array([.21,.25,.66,.7,.7,.7,.7,.3,.3]))}
        # Uniqueness and no mutation: Stage2 scores on represented labels identical if H pred changes.
        blind=(t!=6)
        first=choose_profile(t,blind,data)
        altered=copy.deepcopy(data)
        for key in altered:
            s,p=altered[key]
            s=s.copy();s[~blind]=5
            altered[key]=(s,p)
        self.assertEqual(first,choose_profile(t,blind,altered))
        # No hidden label leakage; visible AUC tie can depend on held-out scores.
        self.assertEqual(choose_profile(t,np.ones(len(t),bool),data), "rf_B")


if __name__ == "__main__":
    unittest.main(verbosity=2)
