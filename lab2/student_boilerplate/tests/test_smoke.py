"""Minimal smoke tests for the Lab 2 student scaffold.

These tests only check that the expected folders and files are present.
Run with:  pytest tests/
"""

from pathlib import Path


def test_project_layout_exists() -> None:
    """Confirm that the core project folders are present."""
    root = Path(__file__).resolve().parents[1]
    assert (root / "src").exists(), "src/ folder is missing"
    assert (root / "outputs").exists(), "outputs/ folder is missing"
    assert (root / "tests").exists(), "tests/ folder is missing"


def test_source_files_exist() -> None:
    """Confirm that all pipeline scripts are present."""
    src = Path(__file__).resolve().parents[1] / "src"
    expected = [
        "settings.py",
        "1_capture_data.py",
        "2_split_data.py",
        "3_train.py",
        "4_inference.py",
    ]
    for filename in expected:
        assert (src / filename).exists(), f"Missing source file: {filename}"


def test_dataset_folder_exists() -> None:
    """Confirm the shared dataset folder is present next to this repo."""
    dataset = Path(__file__).resolve().parents[3] / "dataset"
    assert dataset.exists(), (
        f"Dataset folder not found at {dataset}. "
        "Run 1_capture_data.py first."
    )
