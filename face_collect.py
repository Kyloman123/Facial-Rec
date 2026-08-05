"""Collect face crops from a webcam for LBPH training."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2


DEFAULT_OUTPUT_DIR = Path("data/faces")
FACE_SIZE = (150, 150)


def non_negative_int(value: str) -> int:
    camera_index = int(value)
    if camera_index < 0:
        raise argparse.ArgumentTypeError("camera index must be zero or greater")
    return camera_index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect face images from a webcam.")
    parser.add_argument("--name", required=True, help="Label for the person being recorded.")
    parser.add_argument("--samples", type=int, default=80, help="Number of face samples to save.")
    parser.add_argument("--camera", type=non_negative_int, default=0, help="Webcam index.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be at least 1.")
    return args


def clean_label(label: str) -> str:
    cleaned = "".join(ch for ch in label.strip() if ch.isalnum() or ch in ("-", "_"))
    if not cleaned:
        raise ValueError("Name must contain at least one letter or number.")
    return cleaned


def main() -> None:
    args = parse_args()
    label = clean_label(args.name)
    person_dir = args.output_dir / label
    person_dir.mkdir(parents=True, exist_ok=True)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        raise RuntimeError(f"Could not open camera index {args.camera}.")

    saved = 0
    print("Collecting faces. Press q to stop early.")

    try:
        while saved < args.samples:
            ok, frame = camera.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

            for x, y, w, h in faces:
                face = cv2.resize(gray[y : y + h, x : x + w], FACE_SIZE)
                output_path = person_dir / f"{label}_{saved:04d}.png"
                cv2.imwrite(str(output_path), face)
                saved += 1

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 200, 255), 2)
                cv2.putText(
                    frame,
                    f"{label}: {saved}/{args.samples}",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 200, 255),
                    2,
                )
                if saved >= args.samples:
                    break

            cv2.imshow("Collect Faces", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()

    print(f"Saved {saved} samples to {person_dir}.")


if __name__ == "__main__":
    main()
