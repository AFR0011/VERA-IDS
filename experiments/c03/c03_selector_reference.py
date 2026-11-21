"""C03 controlled visible-vs-blind selector: pure, model-free reference.

Scientific authorization: unified selector approved 2026-10-25.
This reference deliberately does not import VERA-IDS, read source datasets,
fit models, access test scores, or change repository files.
Python 3.10+; numpy only for rejector-level routines.

Class indices: Benign=0, the five represented attack families=1..5,
Unknown=6. Input scores must already be computed from shared fitted models.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np

N_CLASSES = 7
UNKNOWN = 6
EPS = 1e-12
DEFAULT_CONSTRAINTS = {
    "benign_family_fp": 0.02,
    "benign_rejection": 0.10,
    "fur_known": 0.05,
    "fur_known_attacks": 0.10,
}
BUDGETS = (0.01, 0.03, 0.05, 0.10)
ALPHAS = tuple(round(i * 0.005, 12) for i in range(21))


@dataclass(frozen=True)
class ValidationScores:
    truth: np.ndarray      # 0..6; blind view has no 6s
    pred_family: np.ndarray  # 1..5; family prediction irrespective of gate
    gate: np.ndarray       # Stage1 p_attack >= frozen Stage1 threshold
    score: np.ndarray      # finite uncertainty for global method, e.g. 1-pmax
    pmax: Optional[np.ndarray] = None
    true_known_family_prob: Optional[np.ndarray] = None

    def checked(self) -> "ValidationScores":
        n = len(self.truth)
        for key in ("pred_family", "gate", "score"):
            if len(getattr(self, key)) != n:
                raise ValueError(f"misaligned {key}")
        if n == 0:
            raise ValueError("empty validation view")
        if np.any((self.truth < 0) | (self.truth >= N_CLASSES)):
            raise ValueError("unknown truth index")
        if np.any((self.pred_family < 1) | (self.pred_family > 5)):
            raise ValueError("predicted family must be one of the five represented families")
        if not np.issubdtype(np.asarray(self.truth).dtype, np.integer):
            raise ValueError("truth must be integer labels")
        if self.pmax is not None and len(self.pmax) != n:
            raise ValueError("misaligned pmax")
        if self.true_known_family_prob is not None and len(self.true_known_family_prob) != n:
            raise ValueError("misaligned true_known_family_prob")
        return self


@dataclass(frozen=True)
class Selection:
    method: str
    budget: float
    threshold: Optional[float]
    alpha: Optional[float]
    family_thresholds: Optional[tuple[float, ...]]
    macro_f1: float
    known_reject: float
    benign_reject: float
    benign_family_fp: float
    known_attack_reject: float
    unknown_detection: Optional[float]
    status: str = "feasible"


def _rate(num, den):
    # Denominators needed for constraints must never silently vanish.
    if den <= 0:
        raise ValueError("missing required known/benign/known-attack validation support")
    return np.asarray(num, dtype=float) / den


def macro_f1_fixed(support: np.ndarray, tp: np.ndarray, predicted: np.ndarray):
    """Fixed 7-class macro-F1. 0/0 class F1 is 0, including blind Unknown."""
    den = support + predicted
    return np.divide(2.0 * tp, den, out=np.zeros_like(den, dtype=float), where=den > 0).mean(axis=-1)


def _baseline(scores: ValidationScores):
    s = scores.checked()
    t = np.asarray(s.truth, dtype=np.int64)
    pf = np.asarray(s.pred_family, dtype=np.int64)
    g = np.asarray(s.gate, dtype=bool)
    pred = np.where(g, pf, 0)
    support = np.bincount(t, minlength=N_CLASSES)
    predicted = np.bincount(pred, minlength=N_CLASSES)
    tp = np.bincount(t[t == pred], minlength=N_CLASSES)
    k = t != UNKNOWN
    benign = t == 0
    knownatt = (t >= 1) & (t <= 5)
    nk, nb, na = int(k.sum()), int(benign.sum()), int(knownatt.sum())
    if min(nk, nb, na) <= 0:
        raise ValueError("known, benign and known-attack denominators required")
    base_bfp = int((benign & g).sum())
    return t, pred, support, predicted, tp, nk, nb, na, base_bfp


def _choose(feasible, f1, fur, reject_k, param_preference):
    """Canonical lexicographic order with EPS-ties; first returns None on no feasible."""
    eligible = np.flatnonzero(feasible)
    if not eligible.size:
        return None
    # Compare all values to the extremum (not pairwise nontransitive approximately-equal sorting).
    best = np.max(f1[eligible]); eligible = eligible[f1[eligible] >= best - EPS]
    best = np.min(fur[eligible]); eligible = eligible[fur[eligible] <= best + EPS]
    best = np.min(reject_k[eligible]); eligible = eligible[reject_k[eligible] <= best + EPS]
    best = np.max(param_preference[eligible]); eligible = eligible[param_preference[eligible] >= best - EPS]
    return int(eligible[0])


def select_global(scores: ValidationScores, budget: float, method: str = "entropy") -> Optional[Selection]:
    """Exact unique gated finite scores, no-reject included, tie groups indivisible.

    Candidate macro-F1s use cumulative confusion counts, not per-threshold
    complete recomputation. Feasibility ALWAYS uses K, even when U is visible.
    Input score must already use original method semantics (e.g. entropy normalized).
    """
    if budget not in BUDGETS:
        raise ValueError("unfrozen budget")
    if method not in ("entropy", "margin", "max_confidence"):
        raise ValueError("invalid global method")
    t, pred, support, pred0, tp0, nk, nb, na, bfp0 = _baseline(scores)
    g = np.asarray(scores.gate, dtype=bool)
    uall = np.asarray(scores.score, dtype=np.float64)
    if not np.all(np.isfinite(uall[g])):
        raise ValueError("non-finite gated uncertainty")
    ix = np.flatnonzero(g)
    if ix.size:
        order = np.argsort(-uall[ix], kind="mergesort")
        ix = ix[order]
        us = uall[ix]
        gid = np.cumsum(np.r_[True, us[1:] != us[:-1]]).astype(np.int64) - 1
        ng = int(gid[-1] + 1)
        # One row per score group. Shared exact-equality scores are never split.
        freq_by_pred = np.bincount(gid * N_CLASSES + pred[ix], minlength=ng * N_CLASSES).reshape(ng, N_CLASSES)
        old_correct = (t[ix] == pred[ix])
        tp_removal = np.bincount((gid[old_correct] * N_CLASSES + pred[ix][old_correct]), minlength=ng*N_CLASSES).reshape(ng,N_CLASSES)
        unknown_add = np.bincount(gid[t[ix] == UNKNOWN], minlength=ng)
        known_by_group = np.bincount(gid[t[ix] != UNKNOWN], minlength=ng)
        benign_by_group = np.bincount(gid[t[ix] == 0], minlength=ng)
        attack_by_group = np.bincount(gid[(t[ix] >= 1)&(t[ix] <=5)], minlength=ng)
        pcum = np.vstack([np.zeros((1,N_CLASSES), dtype=np.int64),np.cumsum(freq_by_pred,axis=0)])
        tcum = np.vstack([np.zeros((1,N_CLASSES), dtype=np.int64),np.cumsum(tp_removal,axis=0)])
        ukcum = np.r_[0, np.cumsum(unknown_add)]
        kcum = np.r_[0, np.cumsum(known_by_group)]
        bcum = np.r_[0, np.cumsum(benign_by_group)]
        acum = np.r_[0, np.cumsum(attack_by_group)]
        # The pred-family columns lose rejected rows; Unknown gains all rejected rows.
        all_cum = pcum.sum(axis=1)
        preds = pred0[None,:] - pcum
        preds[:,UNKNOWN] += all_cum
        tps = tp0[None,:] - tcum
        tps[:,UNKNOWN] += ukcum
        thresholds = np.r_[np.inf, us[np.flatnonzero(np.r_[us[:-1] != us[1:], True])]]
    else:
        ng = 0
        preds = pred0[None,:]
        tps = tp0[None,:]
        kcum=bcum=acum=np.array([0])
        ukcum=np.array([0]); thresholds=np.array([np.inf])

    f1 = macro_f1_fixed(support[None,:], tps, preds)
    fur = _rate(kcum,nk)
    benrej = _rate(bcum,nb)
    attrej = _rate(acum,na)
    bfp = _rate(bfp0-bcum, nb)
    ok = ((fur <= budget+EPS) & (fur <= DEFAULT_CONSTRAINTS["fur_known"]+EPS)
          & (benrej <= DEFAULT_CONSTRAINTS["benign_rejection"]+EPS)
          & (bfp <= DEFAULT_CONSTRAINTS["benign_family_fp"]+EPS)
          & (attrej <= DEFAULT_CONSTRAINTS["fur_known_attacks"]+EPS))
    # Higher threshold preferred after all exact metric ties.
    pick = _choose(ok, f1, fur, fur, thresholds)
    if pick is None:
        return None
    nu = int((t == UNKNOWN).sum())
    return Selection(method, budget, float(thresholds[pick]), None, None,
                     float(f1[pick]),float(fur[pick]),float(benrej[pick]),
                     float(bfp[pick]),float(attrej[pick]),
                     float(ukcum[pick]/nu) if nu else None)


def evaluate_rejections(scores: ValidationScores, reject: np.ndarray):
    """Brute-force independent fixture oracle; not intended for 700k-row sweeps."""
    t,pred,*rest = _baseline(scores)
    rej=np.asarray(reject,dtype=bool)
    if len(rej)!=len(t) or np.any(rej & ~np.asarray(scores.gate,dtype=bool)):
        raise ValueError("invalid rejection mask")
    pred=np.where(rej,UNKNOWN,pred)
    support=np.bincount(t,minlength=N_CLASSES)
    predicted=np.bincount(pred,minlength=N_CLASSES)
    tp=np.bincount(t[t==pred],minlength=N_CLASSES)
    k=t!=UNKNOWN; ben=t==0;att=(t>=1)&(t<=5);nU=int((t==UNKNOWN).sum())
    return {
       "macro_f1":float(macro_f1_fixed(support,tp,predicted)),
       "fur":float(rej[k].mean()),
       "ben_rej":float(rej[ben].mean()),
       "att_rej":float(rej[att].mean()),
       "bfp":float((ben & (pred>0) & (pred<UNKNOWN)).sum()/ben.sum()),
       "udr":float(rej[t==UNKNOWN].mean()) if nU else None,
    }


def select_family(scores: ValidationScores,budget:float) -> Optional[Selection]:
    """21 canonical alpha points, thresholds fitted only on represented-family K."""
    if budget not in BUDGETS: raise ValueError("unfrozen budget")
    t, pred, supp, pc, tp, nk, nb, na, bfp = _baseline(scores)
    if scores.pmax is None or scores.true_known_family_prob is None:
        raise ValueError("family-conditional method requires pmax and true-known-family probs")
    pmax=np.asarray(scores.pmax,dtype=np.float64)
    truep=np.asarray(scores.true_known_family_prob,dtype=np.float64)
    if np.any(~np.isfinite(pmax[np.asarray(scores.gate,dtype=bool)])):
        raise ValueError("nonfinite family pmax")
    vals=[truep[t==fam] for fam in range(1,6)]
    if any(len(v)==0 or not np.isfinite(v).all() for v in vals):
        raise ValueError("missing/invalid true-family calibration support")
    rows=[]
    for alpha in ALPHAS:
        thresholds=np.asarray([np.quantile(v,alpha) for v in vals],dtype=np.float64)
        rej=np.asarray(scores.gate,dtype=bool) & (pmax < thresholds[np.asarray(scores.pred_family,dtype=np.int64)-1])
        m=evaluate_rejections(scores,rej)
        if (m["fur"]<=budget+EPS and m["fur"]<=DEFAULT_CONSTRAINTS["fur_known"]+EPS
            and m["ben_rej"]<=DEFAULT_CONSTRAINTS["benign_rejection"]+EPS
            and m["bfp"]<=DEFAULT_CONSTRAINTS["benign_family_fp"]+EPS
            and m["att_rej"]<=DEFAULT_CONSTRAINTS["fur_known_attacks"]+EPS):
            rows.append((alpha,thresholds,m))
    if not rows:return None
    f1=np.array([r[2]["macro_f1"] for r in rows]);fur=np.array([r[2]["fur"] for r in rows]);alpha=np.array([r[0] for r in rows])
    pick=_choose(np.ones(len(rows),dtype=bool),f1,fur,fur,-alpha)
    a,ts,m=rows[pick]
    return Selection("family_conditional",budget,None,float(a),tuple(float(x) for x in ts),
                     m["macro_f1"],m["fur"],m["ben_rej"],m["bfp"],m["att_rej"],m["udr"])


def profile_stage2_macro_f1(truth: np.ndarray,pred:np.ndarray):
    """Five represented-family fixed macro-F1 over K attack validation only."""
    true=np.asarray(truth,dtype=np.int64);out=np.asarray(pred,dtype=np.int64)
    mask=(true>=1)&(true<=5)
    if not mask.any():raise ValueError("known Stage2 evaluation empty")
    if not np.all((out[mask]>=1)&(out[mask]<=5)):raise ValueError("invalid Stage2 predicted family")
    support=np.bincount(true[mask]-1,minlength=5)
    ps=np.bincount(out[mask]-1,minlength=5)
    tp=np.bincount(true[mask][true[mask]==out[mask]]-1,minlength=5)
    den=support+ps
    return float(np.divide(2*tp,den,out=np.zeros(5),where=den>0).mean())


def auc_tie_aware(y_binary:np.ndarray, p_attack:np.ndarray):
    """AUROC by pooled tie ranks; fails on single-class validation."""
    yy=np.asarray(y_binary,dtype=np.int64);pp=np.asarray(p_attack,dtype=np.float64)
    if len(yy)!=len(pp) or not np.isfinite(pp).all() or not np.isin(yy,[0,1]).all():
        raise ValueError("invalid Stage1 AUC inputs")
    p=int(yy.sum());n=len(yy)-p
    if p==0 or n==0:raise ValueError("Stage1 AUC undefined on one class")
    order=np.argsort(pp,kind="mergesort")
    sp=pp[order];sy=yy[order]
    starts=np.r_[0,np.flatnonzero(sp[1:]!=sp[:-1])+1]
    ends=np.r_[starts[1:],len(yy)]
    ranks=np.empty(len(yy),dtype=float)
    for st,en in zip(starts,ends):ranks[st:en]=(st+1+en)/2.0
    return float((ranks[sy==1].sum()-p*(p+1)/2)/(p*n))


def choose_profile(truth:np.ndarray,view:np.ndarray,candidate_data:dict):
    """profile -> (Stage2 predictions, Stage1 attack probabilities), full val arrays.

    Stage2 macro-F1 calculated only using represented-family truth in K;
    AUROC is calculated on accessible view, counting U as attack when present.
    """
    t=np.asarray(truth,dtype=np.int64);access=np.asarray(view,dtype=bool)
    if not access.any():raise ValueError("empty accessible selector view")
    items=[]
    for cid,(s2pred,pattack) in candidate_data.items():
        s2=np.asarray(s2pred);p=np.asarray(pattack,dtype=np.float64)
        if len(s2)!=len(t) or len(p)!=len(t):raise ValueError("candidate arrays misaligned")
        f1=profile_stage2_macro_f1(t[access],s2[access])
        auc=auc_tie_aware((t[access]!=0).astype(int),p[access])
        items.append((str(cid),f1,auc))
    bestF1=max(x[1] for x in items)
    tied=[x for x in items if x[1]>=bestF1-EPS]
    bestauc=max(x[2] for x in tied)
    tied=[x for x in tied if x[2]>=bestauc-EPS]
    return sorted(tied,key=lambda x:x[0])[0][0]
