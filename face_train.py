"""Train an OpenCV LBPH face recognizer from collected images."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from opencv_utils import create_lbph_recognizer


DEFAULT_DATA_DIR = Path("data/faces")
DEFAULT_MODEL_PATH = Path("models/face_model.yml")
DEFAULT_LABELS_PATH = Path("models/face_labels.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the face recognizer.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS_PATH)
    return parser.parse_args()


def load_training_data(data_dir: Path) -> tuple[list[np.ndarray], list[int], dict[int, str]]:
    images: list[np.ndarray] = []
    label_ids: list[int] = []
    labels: dict[int, str] = {}

    if not data_dir.exists():
        raise RuntimeError(
            f"Training data directory {data_dir} does not exist. "
            "Run `python face_collect.py --name <label>` first."
        )
    if not data_dir.is_dir():
        raise RuntimeError(f"Training data path {data_dir} is not a directory.")

    people = sorted(path for path in data_dir.iterdir() if path.is_dir())
    if not people:
        raise RuntimeError(f"No training folders found in {data_dir}.")

    for label_id, person_dir in enumerate(people):
        person_images: list[np.ndarray] = []
        for image_path in sorted(person_dir.glob("*")):
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue
            person_images.append(cv2.resize(image, (150, 150)))

        if not person_images:
            raise RuntimeError(
                f"No readable training images found in {person_dir}."
            )

        labels[label_id] = person_dir.name
        images.extend(person_images)
        label_ids.extend([label_id] * len(person_images))

    if not images:
        raise RuntimeError(f"No readable training images found in {data_dir}.")

    return images, label_ids, labels


def main() -> None:
    args = parse_args()
    images, label_ids, labels = load_training_data(args.data_dir)

    recognizer = create_lbph_recognizer()
    recognizer.train(images, np.array(label_ids))

    args.model.parent.mkdir(parents=True, exist_ok=True)
    recognizer.write(str(args.model))

    args.labels.parent.mkdir(parents=True, exist_ok=True)
    args.labels.write_text(json.dumps(labels, indent=2), encoding="utf-8")

    print(f"Trained {len(labels)} labels from {len(images)} images.")
    print(f"Model: {args.model}")
    print(f"Labels: {args.labels}")


if __name__ == "__main__":
    main()
