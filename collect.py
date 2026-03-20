import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import json
import urllib.request
import os

# --- Download required model files if not present ---
HAND_MODEL = "hand_landmarker.task"
FACE_MODEL = "face_landmarker.task"

if not os.path.exists(HAND_MODEL):
    print("Downloading hand model...")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
        HAND_MODEL
    )
    print("Done.")

if not os.path.exists(FACE_MODEL):
    print("Downloading face model...")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
        FACE_MODEL
    )
    print("Done.")

# --- Setup new-style MediaPipe ---
hand_options = vision.HandLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=HAND_MODEL),
    num_hands=1
)
face_options = vision.FaceLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=FACE_MODEL),
    num_faces=1
)

hand_detector = vision.HandLandmarker.create_from_options(hand_options)
face_detector = vision.FaceLandmarker.create_from_options(face_options)

# --- Gesture labels ---
GESTURES = {
    'a': 'thumbs_up',
    's': 'peace',
    'd': 'open_palm',
    'f': 'smile',
    'g': 'neutral',
}

data = []
current_label = None

print("=== Gesture Collector ===")
for key, label in GESTURES.items():
    print(f"  [{key}] = {label}")
print("  [q] = quit and save\n")

cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    hand_result = hand_detector.detect(mp_image)
    face_result = face_detector.detect(mp_image)

    landmarks = []

    # Hand landmarks (21 points * 3 = 63 numbers)
    if hand_result.hand_landmarks:
        for hand in hand_result.hand_landmarks:
            for lm in hand:
                landmarks += [lm.x, lm.y, lm.z]
            # Draw dots manually
            for lm in hand:
                cx, cy = int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])
                cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)

    # Face landmarks (first 50 points * 3 = 150 numbers)
    if face_result.face_landmarks:
        for face_lms in face_result.face_landmarks:
            for lm in face_lms[:50]:
                landmarks += [lm.x, lm.y, lm.z]
            # Draw dots manually
            for lm in face_lms[:50]:
                cx, cy = int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])
                cv2.circle(frame, (cx, cy), 2, (255, 200, 0), -1)

    # UI
    label_text = f"Recording: {current_label}" if current_label else "Press a key to record"
    cv2.putText(frame, label_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (0, 255, 0) if current_label else (0, 0, 255), 2)
    cv2.putText(frame, f"Samples: {len(data)}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    cv2.imshow("Collector", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif chr(key) in GESTURES:
        current_label = GESTURES[chr(key)]
        print(f"Now recording: {current_label}")

    if current_label and len(landmarks) > 0:
        data.append({'label': current_label, 'landmarks': landmarks})

cap.release()
cv2.destroyAllWindows()

with open('gesture_data.json', 'w') as f:
    json.dump(data, f)
print(f"\nSaved {len(data)} samples to gesture_data.json")