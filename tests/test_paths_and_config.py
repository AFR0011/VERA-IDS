from __future__ import annotations

from pathlib import Path

from ids_eval_framework.src.paths import REPO_ROOT, load_config, resolve_repo_path


def test_repository_root_and_config_are_portable() -> None:
    assert (REPO_ROOT / "pyproject.toml").exists()
    config = load_config("config/datasets.yml")
    assert config["paths"]["raw_datasets_root"] == "data/raw"
    assert "Codes" not in resolve_repo_path("outputs")


def test_example_local_paths_exist_but_real_local_file_does_not() -> None:
    assert (REPO_ROOT / "config" / "local_paths.example.yml").exists()
    assert not (REPO_ROOT / "config" / "local_paths.yml").exists()
    assert Path(resolve_repo_path("outputs")).is_absolute()


def test_every_published_yaml_config_loads() -> None:
    for path in sorted((REPO_ROOT / "config").glob("*.yml")):
        assert isinstance(load_config(path), dict)
