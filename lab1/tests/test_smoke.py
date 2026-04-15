"""Minimal test placeholder for the student scaffold."""

from pathlib import Path


def test_project_layout_exists() -> None:
    """Confirm that the expected project folders are present."""
    root = Path(__file__).resolve().parents[1]
    assert (root / "src").exists()
    assert (root / "assets").exists()
    assert (root / "outputs").exists()
