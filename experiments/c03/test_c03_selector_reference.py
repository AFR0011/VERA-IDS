"""Scientific acceptance fixtures for the C03 MODEL-FREE selection kernel.
Run: python -m unittest -v test_c03_selector_reference.py
"""
import unittest
import numpy as np
from c03_selector_reference import (
  ValidationScores, BUDGETS,ALPHAS, EPS, UNKNOWN, _choose,
  choose_profile, auc_tie_aware, evaluate_rejections, macro_f1_fixed,
  profile_stage2_macro_f1,select_global,select_family,
)


def specimen(seed=7,nben=150,natt=275,nunknown=45):
    rng=np.random.default_rng(seed)
    truth=np.r_[np.zeros(nben,dtype=int),
                rng.integers(1,6,size=natt),
                np.full(nunknown,6,dtype=int)]
    n=len(truth)
    pf=rng.integers(1,6,size=n)
    gate=np.r_[np.zeros(nben,dtype=bool),rng.random(natt)<.90,rng.random(nunknown)<.95]
    gate[:2]=True
    score=np.round(rng.random(n),2) # creates many equal score groups
    pmax=rng.uniform(.05,.95,size=n)
    truep=rng.uniform(.05,.95,size=n)
    return ValidationScores(truth,pf,gate,score,pmax,truep)


def brute_global(s,budget):
    # Independent deliberately slow enumerator. Does not use cumulative counters.
    gated=s.score[s.gate]
    thresholds=[float('inf')]+[float(v) for v in sorted(set(gated),reverse=True)]
    rows=[]
    for thr in thresholds:
        rej=s.gate & (s.score>=thr) if np.isfinite(thr) else np.zeros(len(s.truth),dtype=bool)
        m=evaluate_rejections(s,rej)
        ok=(m['fur']<=budget+EPS and m['fur']<=.05+EPS
            and m['ben_rej']<=.10+EPS and m['bfp']<=.02+EPS
            and m['att_rej']<=.10+EPS)
        if ok: rows.append((thr,m))
    if not rows: return None
    f=np.array([x[1]['macro_f1'] for x in rows])
    k=np.array([x[1]['fur'] for x in rows])
    p=np.array([x[0] for x in rows])
    pick=_choose(np.ones(len(rows),bool),f,k,k,p)
    return rows[pick]


class Acceptance(unittest.TestCase):
    def test_macro_fixed_seven_zero_absent_unknown(self):
        s=np.array([4,2,2,2,2,2,0]);p=s.copy();tp=s.copy()
        self.assertAlmostEqual(macro_f1_fixed(s,tp,p),6/7)
        self.assertEqual(macro_f1_fixed(np.zeros(7),np.zeros(7),np.zeros(7)),0.0)

    def test_auc_average_score_ties(self):
        self.assertAlmostEqual(auc_tie_aware([0,0,1,1],[.1,.5,.5,.9]),.875)
        with self.assertRaises(ValueError): auc_tie_aware([1,1],[.3,.4])

    def test_stage2_score_excludes_h_unknown(self):
        truth=np.array([0,1,2,3,4,5,6,6])
        a=np.array([1,1,2,3,4,5,1,2]);b=np.array([1,1,2,3,4,5,5,5])
        self.assertEqual(profile_stage2_macro_f1(truth,a),profile_stage2_macro_f1(truth,b))

    def test_profile_accessible_auc_can_switch_tie(self):
        t=np.array([0,0,1,2,3,4,5,6,6]);s2=np.array([1,1,1,2,3,4,5,1,1])
        # Both candidates have equal K-only Stage2 macro-F1. Stage1 AUC on
        # K prefers A; after adding unseen attacks U prefers B.
        A=np.array([.15,.19,.8,.8,.8,.8,.8,.01,.01])
        B=np.array([.11,.21,.5,.5,.5,.5,.5,.99,.99])
        data={'A':(s2,A),'B':(s2,B)}
        known=t!=6
        self.assertEqual(choose_profile(t,known,data),'A')
        self.assertEqual(choose_profile(t,np.ones(len(t),bool),data),'B')

    def test_exact_tied_uncertainty_group(self):
        s=specimen(seed=42)
        # Group threshold must select either entire exact-equal-score set or none.
        for m in ('entropy','margin','max_confidence'):
            r=select_global(s,.03,m)
            if r is not None and np.isfinite(r.threshold):
                reject=s.gate&(s.score>=r.threshold)
                group=s.gate&(s.score==r.threshold)
                self.assertTrue(np.all(reject[group]))

    def test_no_gated_scores(self):
        s=specimen(seed=15)
        s=ValidationScores(s.truth,s.pred_family,np.zeros(len(s.truth),bool),s.score,s.pmax,s.true_known_family_prob)
        r=select_global(s,.03)
        self.assertIsNotNone(r)
        self.assertEqual(r.threshold,float('inf'))
        self.assertEqual(r.macro_f1,evaluate_rejections(s,np.zeros(len(s.truth),bool))['macro_f1'])

    def test_undefined_feasibility_does_not_impute(self):
        s=specimen(seed=12)
        g=np.ones(len(s.truth),dtype=bool)
        s=ValidationScores(s.truth,s.pred_family,g,s.score,s.pmax,s.true_known_family_prob)
        # Every benign passes attack gate and no more than 1% of K may be rejected.
        # No candidate can remove enough benign->family false positives.
        self.assertIsNone(select_global(s,.01))

    def test_unknown_only_rejection_uses_K_denominator(self):
        nben=250;natt=250;nunk=500
        t=np.r_[np.zeros(nben,int),np.tile(np.arange(1,6),natt//5),np.full(nunk,6,int)]
        pf=np.r_[np.ones(nben,int),np.tile(np.arange(1,6),natt//5),np.ones(nunk,int)]
        gate=np.r_[np.zeros(nben,bool),np.ones(natt+nunk,bool)]
        # K scores small, all 500 unknown score high.
        u=np.r_[np.full(nben,.1),np.full(natt,.1),np.full(nunk,.9)]
        s=ValidationScores(t,pf,gate,u)
        r=select_global(s,.01)
        self.assertIsNotNone(r)
        self.assertEqual(r.unknown_detection,1.0)
        self.assertEqual(r.known_reject,0.0)
        self.assertEqual(r.threshold,.9)

    def test_cumulative_matches_bruteforce_56_cases(self):
        for seed in range(14):
            s=specimen(seed)
            for b in BUDGETS:
                brute=brute_global(s,b)
                fast=select_global(s,b)
                with self.subTest(seed=seed,budget=b):
                    self.assertEqual(fast is None,brute is None)
                    if brute is not None:
                        self.assertEqual(fast.threshold,brute[0])
                        self.assertAlmostEqual(fast.macro_f1,brute[1]['macro_f1'],places=12)
                        self.assertAlmostEqual(fast.known_reject,brute[1]['fur'],places=12)
                        self.assertAlmostEqual(fast.benign_family_fp,brute[1]['bfp'],places=12)

    def test_family_exact_quantile_and_strict_inequality(self):
        s=specimen(seed=51)
        r=select_family(s,.03)
        if r is not None:
            self.assertIn(r.alpha,ALPHAS)
            for f in range(1,6):
                self.assertAlmostEqual(r.family_thresholds[f-1],
                                       np.quantile(s.true_known_family_prob[s.truth==f],r.alpha))
            threshold=np.asarray(r.family_thresholds)[s.pred_family-1]
            rej=s.gate&(s.pmax<threshold)
            m=evaluate_rejections(s,rej)
            self.assertAlmostEqual(r.macro_f1,m['macro_f1'])
            self.assertAlmostEqual(r.known_reject,m['fur'])

    def test_same_inputs_same_selection(self):
        s=specimen(71)
        for b in BUDGETS:
            self.assertEqual(select_global(s,b),select_global(s,b))
            self.assertEqual(select_family(s,b),select_family(s,b))

    def test_missing_benign_fails(self):
        s=specimen(19,nben=0)
        with self.assertRaises(ValueError):select_global(s,.03)

    def test_budget_outside_frozen_set_fails(self):
        with self.assertRaises(ValueError):select_global(specimen(),.02)

if __name__=='__main__':unittest.main(verbosity=2)
