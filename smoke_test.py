"""Run quick non-camera checks for the face-recognition workflow."""

from __future__ import annotations

import json
import argparse
import tempfile
from types import SimpleNamespace
from pathlib import Path

from face_collect import clean_label, next_sample_path, positive_int, save_face
from face_train import load_training_data, validate_output_paths
from opencv_utils import create_lbph_recognizer, load_frontal_face_cascade
from recognizer import load_labels, load_recognizer


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
        if "invalid label entry" not in str(exc):
            raise AssertionError(f"Unexpected label error message: {exc}") from exc
    else:
        raise AssertionError("Expected invalid label ids to fail.")


def assert_rejects_blank_label_name(path: Path) -> None:
    try:
        load_labels(path)
    except RuntimeError as exc:
        if "invalid label entry" not in str(exc):
            raise AssertionError(f"Unexpected blank-label error message: {exc}") from exc
    else:
        raise AssertionError("Expected blank label names to fail.")


def assert_rejects_duplicate_label_ids(path: Path) -> None:
    try:
        load_labels(path)
    except RuntimeError as exc:
        if "invalid label entry" not in str(exc):
            raise AssertionError(f"Unexpected duplicate-label error message: {exc}") from exc
    else:
        raise AssertionError("Expected duplicate numeric label ids to fail.")


def assert_rejects_empty_labels(path: Path) -> None:
    try:
        load_labels(path)
    except RuntimeError as exc:
        if "does not contain any labels" not in str(exc):
            raise AssertionError(f"Unexpected empty-label error message: {exc}") from exc
    else:
        raise AssertionError("Expected empty label files to fail.")


def assert_rejects_invalid_model_paths(path: Path) -> None:
    path.mkdir()
    model_dir = path / "model.yml"
    labels_dir = path / "labels.json"
    model_dir.mkdir()
    labels_dir.mkdir()
    try:
        load_recognizer(model_dir, labels_dir)
    except FileNotFoundError as exc:
        if "Missing or invalid trained model files" not in str(exc):
            raise AssertionError(f"Unexpected model-path error message: {exc}") from exc
    else:
        raise AssertionError("Expected model directories to fail validation.")


def assert_rejects_shared_model_output(path: Path) -> None:
    try:
        validate_output_paths(path, path.parent / "." / path.name)
    except RuntimeError as exc:
        if "must use different files" not in str(exc):
            raise AssertionError(f"Unexpected output-path error message: {exc}") from exc
    else:
        raise AssertionError("Expected shared model and label output to fail.")


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


def assert_reports_failed_sample_write(path: Path) -> None:
    fake_cv = SimpleNamespace(imwrite=lambda output_path, image: False)
    try:
        save_face(object(), path, fake_cv)
    except RuntimeError as exc:
        if str(path) not in str(exc):
            raise AssertionError(f"Unexpected sample-write error: {exc}") from exc
    else:
        raise AssertionError("Expected a failed face sample write to raise an error.")


def assert_preserves_existing_samples(path: Path) -> None:
    path.mkdir()
    (path / "Alice_0000.png").touch()
    (path / "Alice_0002.png").touch()
    output_path, sample_index = next_sample_path(path, "Alice")
    assert output_path == path / "Alice_0001.png"
    assert sample_index == 1


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
    assert_reports_failed_sample_write(Path("data/faces/Alice/Alice_0000.png"))
    assert_rejects_shared_model_output(Path("models/output.yml"))

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        assert_preserves_existing_samples(temp_path / "existing-samples")
        assert_raises_runtime_error(temp_path / "missing-faces")
        assert_rejects_empty_person_folder(temp_path / "faces")
        assert_rejects_non_image_person_folder(temp_path / "non-image-faces")

        bad_labels_path = temp_path / "bad-labels.json"
        bad_labels_path.write_text('{"person": "Alice"}', encoding="utf-8")
        assert_rejects_bad_labels(bad_labels_path)

        blank_label_path = temp_path / "blank-label.json"
        blank_label_path.write_text('{"0": "  "}', encoding="utf-8")
        assert_rejects_blank_label_name(blank_label_path)

        duplicate_label_path = temp_path / "duplicate-label.json"
        duplicate_label_path.write_text(
            '{"1": "Alice", "01": "Bob"}', encoding="utf-8"
        )
        assert_rejects_duplicate_label_ids(duplicate_label_path)

        negative_label_path = temp_path / "negative-label.json"
        negative_label_path.write_text('{"-1": "Alice"}', encoding="utf-8")
        assert_rejects_bad_labels(negative_label_path)

        padded_label_path = temp_path / "padded-label.json"
        padded_label_path.write_text('{"0": " Alice "}', encoding="utf-8")
        assert load_labels(padded_label_path) == {0: "Alice"}

        empty_labels_path = temp_path / "empty-labels.json"
        empty_labels_path.write_text("{}", encoding="utf-8")
        assert_rejects_empty_labels(empty_labels_path)

        assert_rejects_invalid_model_paths(temp_path / "invalid-model-paths")

    print("Smoke checks passed.")


if __name__ == "__main__":
    main()
