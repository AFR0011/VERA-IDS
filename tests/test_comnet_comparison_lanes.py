from __future__ import annotations

from ids_eval_framework.src.external_profile_robustness import run_external_profile_robustness
from ids_eval_framework.src.model_competence import build_model, run_model_competence
from ids_eval_framework.src.paths import load_config
from ids_eval_framework.src.reference_framework_eval import build_model as build_reference_model


def test_model_competence_config_and_dry_run() -> None:
    config = load_config("config/model_competence.yml")
    cfg = config["model_competence"]
    assert cfg["model_families"] == ["extra_trees", "lgbm", "catboost"]
    assert cfg["seeds"] == [123, 124, 125, 126, 127]
    out = run_model_competence(config, dry_run=True)
    assert out.name == "12_model_competence"


def test_competence_model_factory_supports_locked_families() -> None:
    extra = build_model(
        "extra_trees",
        {"n_estimators": 5, "max_depth": 2},
        stage="stage1",
        n_classes=2,
        seed=123,
        n_jobs=1,
    )
    lgbm = build_model(
        "lgbm",
        {"n_estimators": 5, "num_leaves": 7},
        stage="stage1",
        n_classes=2,
        seed=123,
        n_jobs=1,
    )
    cat = build_model(
        "catboost",
        {"iterations": 5, "depth": 2},
        stage="stage1",
        n_classes=2,
        seed=123,
        n_jobs=1,
    )
    assert extra.__class__.__name__ == "ExtraTreesClassifier"
    assert lgbm.__class__.__name__ == "LGBMClassifier"
    assert cat.__class__.__name__ == "CatBoostClassifier"


def test_reference_profile_factory_supports_lightgbm() -> None:
    profile = {
        "model_family": "lgbm",
        "stage1_params": {"n_estimators": 5, "learning_rate": 0.05},
        "stage2_params": {"n_estimators": 5, "learning_rate": 0.05},
    }
    model = build_reference_model("test_lgbm", profile, "stage2", 4, 123, 1)
    assert model.__class__.__name__ == "LGBMClassifier"
    assert model.get_params()["num_class"] == 4


def test_external_profile_robustness_config_and_dry_run() -> None:
    config = load_config("config/external_profile_robustness.yml")
    cfg = config["external_profile_robustness"]
    assert cfg["profiles"] == [
        "adewole2025_xgb_profile",
        "keskin2026_lgbm_profile",
        "hung2026_xgb_profile",
        "christy2025_rf_profile",
    ]
    out = run_external_profile_robustness(config, dry_run=True)
    assert out.name == "13_external_profile_robustness"
