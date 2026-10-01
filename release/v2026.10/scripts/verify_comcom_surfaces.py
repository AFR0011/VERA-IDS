#!/usr/bin/env python3
import csv
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def rows(rel):
    with (ROOT / rel).open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))

def close(a, b, tol=5e-7):
    return math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=tol)

direct = rows("raw/controlled_direct_test_seed_summary.csv")
def drow(dataset, model, surface):
    return next(r for r in direct if r["dataset"] == dataset and r["model_family"] == model and r["surface"] == surface)
assert close(drow("CICIDS2017", "rf", "argmax")["macro_f1_mean"], 0.9213583846083055)
assert close(drow("CICIoT2023", "rf", "fpr_matched")["macro_f1_mean"], 0.8843900476142148)

blind = rows("raw/blind_phase3b_five_seed_summary.csv")
def brow(holdout, method):
    return next(r for r in blind if r["holdout_family"] == holdout and r["method"] == method and close(r["budget"], 0.03, 1e-12))
assert close(brow("Botnet", "max_confidence")["test_unknown_detection_rate_mean"], 0.993436, 1e-5)
assert close(brow("DDoS", "family_conditional")["test_unknown_detection_rate_mean"], 0.704969, 1e-5)
assert close(brow("BruteForce", "family_conditional")["test_unknown_detection_rate_mean"], 0.0)

common = rows("raw/visible_primary_3pct_common_case_method_summary.csv")
def crow(dataset, method):
    return next(r for r in common if r["dataset"] == dataset and r["method"] == method)
assert close(crow("CICIDS2017", "family_conditional")["unknown_detection_rate_mean"], 0.9413318831391239)
assert close(crow("CICIoT2023", "max_confidence")["macro_f1_mean"], 0.7693052160952947)

feas = rows("raw/visible_budget_feasibility_coverage.csv")
def nfeas(dataset, budget, method):
    return int(next(r for r in feas if r["dataset"] == dataset and close(r["budget"], budget, 1e-12) and r["method"] == method)["n_holdouts_feasible"])
assert nfeas("CICIDS2017", 0.03, "family_conditional") == 3
assert nfeas("CICIoT2023", 0.03, "max_confidence") == 6

resid = rows("raw/residual_failure_structure_primary_fixed_rf.csv")
bot = next(r for r in resid if r["dataset"] == "CICIoT2023" and r["holdout_family"] == "Botnet")
assert bot["modal_dominant_destination"] == "DDoS"
assert close(bot["dominant_destination_share_of_failures_mean"], 0.9922203044947686)

part = rows("raw/cicids_partition_sensitivity_by_holdout.csv")
dos = next(r for r in part if r["holdout_family"] == "DoS")
assert close(dos["absolute_family_shift"], 0.8903666797017903)

competitive = rows("raw/competitive_winner_test_results.csv")
iot = next(r for r in competitive if r["dataset"] == "CICIoT2023" and r["surface"] == "direct_multiclass")
assert close(iot["benign_family_fp_rate"], 0.0811513092144713)

refs = rows("raw/reference_manuscript_table_corrected.csv")
a = next(r for r in refs if r["paper"] == "adewole2025_xgb" and r["dataset"] == "CICIDS2017")
assert close(a["source_inspired_macro_f1_supported"], 0.753401309040285)

transfer = rows("raw/webapp_transfer_summary.csv")
assert len(transfer) == 1
assert close(transfer[0]["test_overall_reject_rate"], 0.1536155, 1e-5)
assert close(transfer[0]["test_false_unknown_rate_all_known"], 0.1532317, 1e-5)

print("PASS: v2026.10 Computer Communications evidence surfaces verified.")
