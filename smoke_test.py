"""Run quick non-camera checks for the face-recognition workflow."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from face_collect import clean_label
from face_train import load_training_data


def assert_raises_runtime_error(path: Path) -> None:
    try:
        load_training_data(path)
    except RuntimeError as exc:
        if "Training data directory" not in str(exc):
            raise AssertionError(f"Unexpected error message: {exc}") from exc
    else:
        raise AssertionError("Expected missing training data directory to fail.")


def main() -> None:
    assert clean_label(" Kylo Dev! ") == "KyloDev"
    assert clean_label("person_01-test") == "person_01-test"

    labels_path = Path("examples/face_labels.sample.json")
    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    assert labels == {"0": "Alice", "1": "Bob"}

    with tempfile.TemporaryDirectory() as temp_dir:
        assert_raises_runtime_error(Path(temp_dir) / "missing-faces")

    print("Smoke checks passed.")


if __name__ == "__main__":
    main()
