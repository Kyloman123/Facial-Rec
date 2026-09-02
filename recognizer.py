"""Run live webcam face recognition with an LBPH model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from opencv_utils import create_lbph_recognizer, load_frontal_face_cascade


DEFAULT_MODEL_PATH = Path("models/face_model.yml")
DEFAULT_LABELS_PATH = Path("models/face_labels.json")
FACE_SIZE = (150, 150)


def non_negative_int(value: str) -> int:
    camera_index = int(value)
    if camera_index < 0:
        raise argparse.ArgumentTypeError("camera index must be zero or greater")
    return camera_index


def non_negative_float(value: str) -> float:
    threshold = float(value)
    if threshold < 0:
        raise argparse.ArgumentTypeError("threshold must be zero or greater")
    return threshold


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run live face recognition.")
    parser.add_argument("--camera", type=non_negative_int, default=0, help="Webcam index.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS_PATH)
    parser.add_argument(
        "--threshold",
        type=non_negative_float,
        default=80.0,
        help="LBPH confidence cutoff. Lower is stricter.",
    )
    return parser.parse_args()


def load_recognizer(model_path: Path, labels_path: Path):
    if not model_path.exists() or not labels_path.exists():
        raise FileNotFoundError(
            "Missing trained model files. Run `python face_train.py` first."
        )

    recognizer = create_lbph_recognizer()
    recognizer.read(str(model_path))

    labels = load_labels(labels_path)
    return recognizer, labels


def load_labels(labels_path: Path) -> dict[int, str]:
    try:
        raw_labels = json.loads(labels_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Labels file {labels_path} is not valid JSON.") from exc

    if not isinstance(raw_labels, dict):
        raise RuntimeError(f"Labels file {labels_path} must contain a JSON object.")
    if not raw_labels:
        raise RuntimeError(f"Labels file {labels_path} does not contain any labels.")

    labels: dict[int, str] = {}
    for label_id, name in raw_labels.items():
        try:
            clean_name = str(name).strip()
            if not clean_name:
                raise ValueError("label name is empty")
            labels[int(label_id)] = clean_name
        except ValueError as exc:
            raise RuntimeError(
                f"Labels file {labels_path} contains an invalid label entry: {label_id!r}."
            ) from exc

    return labels


def main() -> None:
    args = parse_args()
    recognizer, labels = load_recognizer(args.model, args.labels)

    face_cascade = load_frontal_face_cascade()
    camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        raise RuntimeError(f"Could not open camera index {args.camera}.")

    print("Running recognition. Press q to quit.")

    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

            for x, y, w, h in faces:
                face = cv2.resize(gray[y : y + h, x : x + w], FACE_SIZE)
                label_id, confidence = recognizer.predict(face)

                if confidence <= args.threshold:
                    name = labels.get(label_id, "Unknown")
                    color = (0, 200, 255)
                    display_confidence = max(0, round(100 - confidence))
                else:
                    name = "Unknown"
                    color = (0, 0, 255)
                    display_confidence = 0

                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.rectangle(frame, (x, y - 32), (x + w, y), color, -1)
                cv2.putText(
                    frame,
                    f"{name} ({display_confidence}%)",
                    (x + 4, y - 9),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2,
                )

            cv2.imshow("Facial Recognition", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
