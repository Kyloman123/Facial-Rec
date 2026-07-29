"""Run quick non-camera checks for the face-recognition workflow."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from face_collect import clean_label
from face_train import load_training_data
from recognizer import load_labels


def assert_raises_runtime_error(path: Path) -> None:
    try:
        load_training_data(path)
    except RuntimeError as exc:
        if "Training data directory" not in str(exc):
            raise AssertionError(f"Unexpected error message: {exc}") from exc
    else:
        raise AssertionError("Expected missing training data directory to fail.")


def assert_rejects_bad_labels(path: Path) -> None:
    try:
        load_labels(path)
    except RuntimeError as exc:
        if "non-numeric label id" not in str(exc):
            raise AssertionError(f"Unexpected label error message: {exc}") from exc
    else:
        raise AssertionError("Expected invalid label ids to fail.")


def main() -> None:
    assert clean_label(" Kylo Dev! ") == "KyloDev"
    assert clean_label("person_01-test") == "person_01-test"

    labels_path = Path("examples/face_labels.sample.json")
    labels = load_labels(labels_path)
    raw_labels = json.loads(labels_path.read_text(encoding="utf-8"))
    assert raw_labels == {"0": "Alice", "1": "Bob"}
    assert labels == {0: "Alice", 1: "Bob"}

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        assert_raises_runtime_error(temp_path / "missing-faces")

        bad_labels_path = temp_path / "bad-labels.json"
        bad_labels_path.write_text('{"person": "Alice"}', encoding="utf-8")
        assert_rejects_bad_labels(bad_labels_path)

    print("Smoke checks passed.")


if __name__ == "__main__":
    main()
