"""Run quick non-camera checks for the face-recognition workflow."""

from __future__ import annotations

import json
import argparse
import tempfile
from types import SimpleNamespace
from pathlib import Path

from face_collect import clean_label, positive_int
from face_train import load_training_data
from opencv_utils import create_lbph_recognizer, load_frontal_face_cascade
from recognizer import load_labels


def assert_raises_runtime_error(path: Path) -> None:
    try:
        load_training_data(path)
    except RuntimeError as exc:
        if "Training data directory" not in str(exc):
            raise AssertionError(f"Unexpected error message: {exc}") from exc
    else:
        raise AssertionError("Expected missing training data directory to fail.")


def assert_rejects_empty_person_folder(path: Path) -> None:
    person_dir = path / "Alice"
    person_dir.mkdir(parents=True)
    try:
        load_training_data(path)
    except RuntimeError as exc:
        if "No readable training images found" not in str(exc):
            raise AssertionError(f"Unexpected empty-folder error message: {exc}") from exc
    else:
        raise AssertionError("Expected empty person training folder to fail.")


def assert_rejects_non_image_person_folder(path: Path) -> None:
    person_dir = path / "Alice"
    person_dir.mkdir(parents=True)
    (person_dir / "notes.txt").write_text("not a training image", encoding="utf-8")
    try:
        load_training_data(path)
    except RuntimeError as exc:
        if "No readable training images found" not in str(exc):
            raise AssertionError(f"Unexpected non-image error message: {exc}") from exc
    else:
        raise AssertionError("Expected non-image training folder to fail.")


def assert_rejects_bad_labels(path: Path) -> None:
    try:
        load_labels(path)
    except RuntimeError as exc:
        if "non-numeric label id" not in str(exc):
            raise AssertionError(f"Unexpected label error message: {exc}") from exc
    else:
        raise AssertionError("Expected invalid label ids to fail.")


def assert_rejects_empty_labels(path: Path) -> None:
    try:
        load_labels(path)
    except RuntimeError as exc:
        if "does not contain any labels" not in str(exc):
            raise AssertionError(f"Unexpected empty-label error message: {exc}") from exc
    else:
        raise AssertionError("Expected empty label files to fail.")


def assert_argparse_error(func, value: str, expected_message: str) -> None:
    try:
        func(value)
    except argparse.ArgumentTypeError as exc:
        if expected_message not in str(exc):
            raise AssertionError(f"Unexpected parser error: {exc}") from exc
    else:
        raise AssertionError(f"Expected {value!r} to fail validation.")


def assert_explains_missing_lbph() -> None:
    try:
        create_lbph_recognizer(SimpleNamespace())
    except RuntimeError as exc:
        if "opencv-contrib-python" not in str(exc):
            raise AssertionError(f"Unexpected OpenCV setup error: {exc}") from exc
    else:
        raise AssertionError("Expected missing LBPH support to fail.")


def assert_explains_missing_cascade() -> None:
    class EmptyCascade:
        def empty(self) -> bool:
            return True

    fake_cv = SimpleNamespace(
        data=SimpleNamespace(haarcascades="/missing/"),
        CascadeClassifier=lambda path: EmptyCascade(),
    )
    try:
        load_frontal_face_cascade(fake_cv)
    except RuntimeError as exc:
        if "frontal face cascade" not in str(exc):
            raise AssertionError(f"Unexpected cascade setup error: {exc}") from exc
    else:
        raise AssertionError("Expected missing face cascade to fail.")


def main() -> None:
    assert clean_label(" Kylo Dev! ") == "KyloDev"
    assert clean_label("person_01-test") == "person_01-test"
    assert positive_int("1") == 1
    assert_argparse_error(positive_int, "0", "at least 1")

    labels_path = Path("examples/face_labels.sample.json")
    labels = load_labels(labels_path)
    raw_labels = json.loads(labels_path.read_text(encoding="utf-8"))
    assert raw_labels == {"0": "Alice", "1": "Bob"}
    assert labels == {0: "Alice", 1: "Bob"}
    assert_explains_missing_lbph()
    assert_explains_missing_cascade()

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        assert_raises_runtime_error(temp_path / "missing-faces")
        assert_rejects_empty_person_folder(temp_path / "faces")
        assert_rejects_non_image_person_folder(temp_path / "non-image-faces")

        bad_labels_path = temp_path / "bad-labels.json"
        bad_labels_path.write_text('{"person": "Alice"}', encoding="utf-8")
        assert_rejects_bad_labels(bad_labels_path)

        empty_labels_path = temp_path / "empty-labels.json"
        empty_labels_path.write_text("{}", encoding="utf-8")
        assert_rejects_empty_labels(empty_labels_path)

    print("Smoke checks passed.")


if __name__ == "__main__":
    main()
