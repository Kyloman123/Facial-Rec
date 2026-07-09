# Facial-Rec

A small OpenCV face-recognition pipeline for collecting training images, training an LBPH model, and running live webcam recognition.

This project is designed as a local demo. Training images and generated model files stay out of Git so personal face data is not published.

## Features

- Collect face crops from a webcam into a labeled dataset
- Train an OpenCV LBPH face recognizer
- Run live recognition with configurable camera and confidence threshold
- Store generated data under `data/` and `models/`
- Avoid hardcoded machine-specific paths

## Project Structure

```text
.
├── collect.py          # Short wrapper for face collection
├── face_collect.py     # Webcam data collection
├── face_train.py       # Model training
├── recognizer.py       # Live recognition
├── requirements.txt    # Python dependencies
└── .gitignore          # Keeps private data and models out of Git
```

## Setup

Python 3.10+ is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 1. Collect Images

Collect training images for each person. Use a short label such as a first name.

```bash
python face_collect.py --name Kylo --samples 80
```

The script saves cropped grayscale faces to:

```text
data/faces/Kylo/
```

Repeat this step for each person you want the model to recognize.

## 2. Train The Model

```bash
python face_train.py
```

This creates:

```text
models/face_model.yml
models/face_labels.json
```

## 3. Run Recognition

```bash
python recognizer.py
```

Useful options:

```bash
python recognizer.py --camera 1
python recognizer.py --threshold 80
python recognizer.py --model models/face_model.yml --labels models/face_labels.json
```

Press `q` to quit.

## Privacy Notes

Face images and trained model files can contain biometric information. They are intentionally excluded from Git by `.gitignore`.

Do not commit:

- `data/`
- `models/`
- webcam captures
- trained recognizer files

## Limitations

This is a classical computer-vision demo, not a production identity system. Lighting, camera quality, pose, and training data quality strongly affect accuracy.
