"""C03 model-free pre-test sealing contract and regression gate.

Run from the VERA-IDS root with the three prior C03 files present:
    python -B -m unittest -v c03_pretest_sealing_gate.py

This is NOT a complete experiment runner. It imports the real historical Stage-1
selector and the approved C03 selector, exercises them on synthetic scores,
and validates a local test-access guard used ONLY in this fixture. It never
loads source/test data, fits models, writes files, commits, or changes configs.
Passing is a pre-integration contract test, NOT proof that a future runner is sealed.
"""
from __future__ import annotations

from dataclasses import asdict
import copy
import hashlib
import json
import math
from pathlib import Path
import unittest

import numpy as np
from c03_selector_reference import (
    ValidationScores, choose_profile, select_global, select_family,
)
from c03_stage1_integration_gate import make_validation, source_legacy, stage1_selection

ROOT = Path(__file__).resolve().parent
POLICY_FILE = ROOT / "C03_APPROVED_SELECTOR_POLICY.json"
METHODS = ("max_confidence", "margin", "entropy", "family_conditional")
BUDGET = 0.03
CANDIDATES = (
    "xgb_inv_family_clipped", "rf_inv_family_clipped", "rf_class_weight_balanced"
)


def strict_json(value):
    """Canonicalize without JSON NaN/Infinity tokens; no private model data."""
    def norm(v):
        if isinstance(v, dict):
            return {str(k): norm(x) for k, x in v.items()}
        if isinstance(v, (tuple, list)):
            return [norm(x) for x in v]
        if isinstance(v, float):
            if math.isnan(v):
                raise ValueError("NaN is forbidden in frozen selection ledger")
            if math.isinf(v):
                return "+inf" if v > 0 else "-inf"
            return v
        if isinstance(v, np.generic):
            return norm(v.item())
        if isinstance(v, (str, int, bool)) or v is None:
            return v
        raise TypeError(f"Cannot seal {type(v).__name__}")
    return json.dumps(norm(value), ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


def fingerprint(value):
    return hashlib.sha256(strict_json(value).encode("utf-8")).hexdigest()


def frozen_policy():
    data = json.loads(POLICY_FILE.read_text(encoding="utf-8"))
    if data.get("claim_id") != "C03" or not data.get("policy_status", "").startswith("SCIENTIFIC_POLICY_APPROVED"):
        raise RuntimeError("Missing/incorrect C03 scientific policy")
    if data["rejector_selection"]["primary_budget"] != BUDGET:
        raise RuntimeError("Unexpected primary budget")
    if tuple(data["rejector_selection"]["methods"]) != METHODS:
        raise RuntimeError("Unexpected method list")
    if tuple(data["common_fit"]["candidate_profiles"]) != CANDIDATES:
        raise RuntimeError("Unexpected candidate list")
    return data


class LockedTestAccess:
    """Fixture contract: no test loader access until TWO selector ledgers sealed.

    The future actual runner MUST independently implement and integration-test
    its own stronger process/file boundary before any test scores are accessed.
    """
    def __init__(self, loader):
        self._loader = loader
        self._sealed_hash = None
        self._frozen_record = None
        self.loads = 0

    def seal(self, decision_record):
        if self._sealed_hash is not None:
            raise RuntimeError("Cannot reseal an existing pilot")
        if set(decision_record.get("arms", {})) != {"blind", "visible"}:
            raise ValueError("Both validation arms must finish before sealing")
        expected_policy_hash = hashlib.sha256(POLICY_FILE.read_bytes()).hexdigest()
        if decision_record.get("policy_sha256") != expected_policy_hash:
            raise ValueError("Selection ledger policy hash does not match approved local policy")
        kb = decision_record["arms"]["blind"].get("known_validation_count")
        kv = decision_record["arms"]["visible"].get("known_validation_count")
        if not isinstance(kb, int) or kb <= 0 or kv != kb:
            raise ValueError("Known validation population must be the same in both arms")
        if not (decision_record["arms"]["blind"].get("visible_validation_count") == kb
                and decision_record["arms"]["visible"].get("visible_validation_count", 0) > kb):
            raise ValueError("Invalid K vs K+U selector view counts")
        for arm in ("blind", "visible"):
            row = decision_record["arms"][arm]
            if row.get("candidate_profile") not in CANDIDATES:
                raise ValueError("Unknown fitted candidate")
            if set(row.get("selections", {})) != set(METHODS):
                raise ValueError("Missing rejector selection, including infeasible statuses")
            if row.get("primary_budget") != BUDGET:
                raise ValueError("Non-primary budget in primary decision ledger")
            if not math.isfinite(row["stage1_threshold"]):
                raise ValueError("Non-finite Stage-1 threshold")
            if any(v.get("status") not in ("feasible", "infeasible")
                   for v in row["selections"].values()):
                raise ValueError("Unresolved rejector selection status")
        self._frozen_record = copy.deepcopy(decision_record)
        self._sealed_hash = fingerprint(self._frozen_record)
        return self._sealed_hash

    def open_test(self, decision_record):
        if self._sealed_hash is None:
            raise PermissionError("TEST LOCKED: arm selections must be sealed first")
        if fingerprint(decision_record) != self._sealed_hash:
            raise PermissionError("Selection ledger changed after sealing")
        self.loads += 1
        return self._loader()


def make_shared_validation_fixture(seed=287):
    """One synthetic score source shared by both selectors, no test source."""
    truth, probability = make_validation(seed)
    n = len(truth)
    rng = np.random.default_rng(seed+1)
    pf = np.where((truth>=1)&(truth<=5), truth, 2).astype(np.int64)
    mis = (rng.random(n)<.07)&(truth>=1)&(truth<=5)
    pf[mis] = ((pf[mis] % 5)+1)
    s2_a = pf.copy(); s2_b = pf.copy()
    s2_x = pf.copy()
    extra = np.flatnonzero((truth>=1)&(truth<=5))[::13]
    s2_x[extra] = (s2_x[extra] % 5)+1
    profiles = {
        "xgb_inv_family_clipped": (s2_x, np.clip(probability * .91 + .05,0,1)),
        "rf_inv_family_clipped": (s2_a, np.clip(probability * .96 + .02,0,1)),
        "rf_class_weight_balanced": (s2_b, probability.copy()),
    }
    pmax = np.where(truth==6, .19, np.where(truth==0,.58,.91)) + rng.uniform(-.02,.02,n)
    margin = np.where(truth==6, .18, np.where(truth==0,.48,.72)) + rng.uniform(-.02,.02,n)
    entropy = np.where(truth==6, .88, np.where(truth==0,.51,.27)) + rng.uniform(-.02,.02,n)
    truep = np.where(truth==6,.20,np.where(truth==0,.55,.85)) + rng.uniform(-.03,.03,n)
    return truth, profiles, pmax, margin, entropy, truep


def compute_synthetic_decisions(grid, engine, seed=287):
    """Only validation arrays flow into real Stage-1 and approved rejector code."""
    t, profiles, pmax, margin, entropy, truep = make_shared_validation_fixture(seed)
    result = {"policy_sha256": hashlib.sha256(POLICY_FILE.read_bytes()).hexdigest(), "arms": {}}
    for arm in ("blind", "visible"):
        take = t!=6 if arm=="blind" else np.ones(len(t), dtype=bool)
        profile = choose_profile(t,take,profiles)
        stage2_predictions, stage1_probs = profiles[profile]
        thr, _, _ = stage1_selection(grid,engine,t,stage1_probs,arm)
        gates = stage1_probs[take] >= thr
        this = {
            "candidate_profile": profile,
            "stage1_threshold": float(thr),
            "primary_budget": BUDGET,
            "visible_validation_count": int(take.sum()),
            "known_validation_count": int((t[take]!=6).sum()),
            "selections": {},
        }
        for method in METHODS:
            values = {"max_confidence":1-pmax,"margin":1-margin,"entropy":entropy,
                      "family_conditional":np.zeros(len(t),dtype=float)}[method]
            sv = ValidationScores(t[take],stage2_predictions[take],gates,values[take],
                                  pmax[take],truep[take])
            selected = select_family(sv,BUDGET) if method=="family_conditional" else select_global(sv,BUDGET,method)
            this["selections"][method] = ({"status":"infeasible"} if selected is None
                                           else asdict(selected))
        result["arms"][arm] = this
    return result


class Gate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.policy = frozen_policy()
        cls.grid, cls.engine = source_legacy()  # actual VERA-IDS modules
        cls.record = compute_synthetic_decisions(cls.grid,cls.engine)

    def test_01_policy_is_approved_and_comparison_is_heldout_visibility(self):
        self.assertIn("held-out",self.policy["single_changed_factor"])
        self.assertEqual(self.policy["rejector_selection"]["budget_scope"],
                         "K-known-only validation rows in BOTH selector arms")

    def test_02_a_single_shared_validation_frame_divides_only_in_visibility(self):
        blind,visible = self.record["arms"]["blind"],self.record["arms"]["visible"]
        self.assertEqual(blind["known_validation_count"], visible["known_validation_count"])
        self.assertEqual(visible["visible_validation_count"] - blind["visible_validation_count"],250)
        self.assertEqual(blind["visible_validation_count"],2325)
        self.assertEqual(visible["visible_validation_count"],2575)

    def test_03_real_stage1_and_rejectors_emit_complete_arms(self):
        for arm in ("blind", "visible"):
            r = self.record["arms"][arm]
            self.assertIn(r["candidate_profile"],CANDIDATES)
            self.assertTrue(0<=r["stage1_threshold"]<=1)
            self.assertEqual(set(r["selections"]),set(METHODS))
            for method,row in r["selections"].items():
                self.assertIn(row["status"],("feasible","infeasible"))
                if row["status"]=="feasible":
                    self.assertLessEqual(row["known_reject"],BUDGET+1e-12)

    def test_04_test_loader_prohibited_before_both_arms_seal(self):
        x=LockedTestAccess(lambda: (_ for _ in ()).throw(AssertionError("test accessed")))
        with self.assertRaises(PermissionError):x.open_test(self.record)
        self.assertEqual(x.loads,0)

    def test_05_incomplete_or_replaced_arm_blocks_seal(self):
        for drop in ("blind","visible"):
            r=copy.deepcopy(self.record); del r["arms"][drop]
            with self.subTest(drop=drop),self.assertRaises(ValueError):LockedTestAccess(lambda:None).seal(r)
        r=copy.deepcopy(self.record)
        del r["arms"]["blind"]["selections"]["entropy"]
        with self.assertRaises(ValueError):LockedTestAccess(lambda:None).seal(r)

    def test_06_wrong_policy_hash_or_K_counts_block_seal(self):
        r=copy.deepcopy(self.record);r["policy_sha256"]="0"*64
        with self.assertRaises(ValueError):LockedTestAccess(lambda:None).seal(r)
        r=copy.deepcopy(self.record);r["arms"]["visible"]["known_validation_count"]+=1
        with self.assertRaises(ValueError):LockedTestAccess(lambda:None).seal(r)

    def test_07_no_reseal_and_postseal_selection_mutation_block(self):
        vault=LockedTestAccess(lambda: 42)
        r=copy.deepcopy(self.record)
        h=vault.seal(r)
        self.assertEqual(len(h),64)
        with self.assertRaises(RuntimeError):vault.seal(r)
        r["arms"]["blind"]["stage1_threshold"]+=.01
        with self.assertRaises(PermissionError):vault.open_test(r)
        self.assertEqual(vault.loads,0)

    def test_08_exact_sealed_ledger_allows_test_loader_afterwards(self):
        trace=[]
        vault=LockedTestAccess(lambda: trace.append("test_read") or "opaque_test_only")
        r=copy.deepcopy(self.record)
        vault.seal(r)
        self.assertEqual(trace,[])
        self.assertEqual(vault.open_test(r),"opaque_test_only")
        self.assertEqual(trace,["test_read"])
        self.assertEqual(vault.loads,1)

    def test_09_inputs_and_selections_are_deterministic(self):
        other=compute_synthetic_decisions(self.grid,self.engine)
        self.assertEqual(fingerprint(self.record),fingerprint(other))


if __name__=="__main__":
    unittest.main(verbosity=2)
