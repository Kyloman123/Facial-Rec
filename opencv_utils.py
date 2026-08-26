"""OpenCV helpers shared by training and recognition entrypoints."""

from __future__ import annotations

import cv2


def load_frontal_face_cascade(cv_module=cv2):
    cascade_path = cv_module.data.haarcascades + "haarcascade_frontalface_default.xml"
    cascade = cv_module.CascadeClassifier(cascade_path)
    if cascade.empty():
        raise RuntimeError(
            "OpenCV could not load the frontal face cascade. "
            f"Expected detector file at {cascade_path}."
        )
    return cascade


def create_lbph_recognizer(cv_module=cv2):
    face_module = getattr(cv_module, "face", None)
    factory = getattr(face_module, "LBPHFaceRecognizer_create", None)
    if factory is None:
        raise RuntimeError(
            "OpenCV LBPH recognizer is unavailable. "
            "Install dependencies with `pip install -r requirements.txt`; "
            "this project needs opencv-contrib-python, not plain opencv-python."
        )
    return factory()
