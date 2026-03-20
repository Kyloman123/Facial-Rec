import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import json
import math
import numpy as np
import os

# ---- Load Gesture Model ----
with open('/Users/gulickg/Desktop/gesture_model.json') as f:
    model_data = json.load(f)

labels = model_data['labels']

def sigmoid(x):
    return 1 / (1 + math.exp(-max(-500, min(500, x))))

class NeuralNetwork:
    def __init__(self, data):
        self.layer_sizes = data['layer_sizes']
        self.weights = data['weights']
        self.biases = data['biases']

    def predict(self, inputs):
        activations = inputs
        for i in range(len(self.weights)):
            new_activations = []
            for j in range(len(self.weights[i][0])):
                weighted_sum = sum(
                    activations[k] * self.weights[i][k][j]
                    for k in range(len(activations))
                ) + self.biases[i][j]
                new_activations.append(sigmoid(weighted_sum))
            activations = new_activations
        return activations

nn = NeuralNetwork(model_data)

# ---- Load Face Recognition Model ----
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('/Users/gulickg/Desktop/face_model.yml')

with open('/Users/gulickg/Desktop/face_labels.json') as f:
    face_labels = json.load(f)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# ---- MediaPipe Setup ----
hand_options = vision.HandLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path='/Users/gulickg/Desktop/hand_landmarker.task'),
    num_hands=1
)
face_options = vision.FaceLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path='/Users/gulickg/Desktop/face_landmarker.task'),
    num_faces=1
)

hand_detector = vision.HandLandmarker.create_from_options(hand_options)
face_detector = vision.FaceLandmarker.create_from_options(face_options)

cap = cv2.VideoCapture(1)
print("Running! Press Q to quit.")

frame_count = 0
last_name = "Unknown"
last_confidence = 0
last_box = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    hand_result = hand_detector.detect(mp_image)
    face_result = face_detector.detect(mp_image)

    landmarks = []

    # Hand landmarks
    if hand_result.hand_landmarks:
        for hand in hand_result.hand_landmarks:
            for lm in hand:
                landmarks += [lm.x, lm.y, lm.z]
            for lm in hand:
                cx, cy = int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])
                cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)

    # Face landmarks
    if face_result.face_landmarks:
        for face_lms in face_result.face_landmarks:
            for lm in face_lms[:50]:
                landmarks += [lm.x, lm.y, lm.z]
            for lm in face_lms[:50]:
                cx, cy = int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])
                cv2.circle(frame, (cx, cy), 2, (255, 200, 0), -1)

    # ---- Face Recognition (every 3 frames) ----
    frame_count += 1
    if frame_count % 3 == 0:
        faces_detected = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces_detected:
            face_img = gray[y:y+h, x:x+w]
            face_img = cv2.resize(face_img, (150, 150))
            label_id, confidence = recognizer.predict(face_img)

            # Lower confidence = better match in LBPH
            if confidence < 100:
                last_name = face_labels[str(label_id)]
                last_confidence = round(100 - confidence)
            else:
                last_name = "Unknown"
                last_confidence = 0
            last_box = (x, y, w, h)

    # Draw face recognition box
    if last_box:
        x, y, w, h = last_box
        color = (0, 200, 255) if last_name != "Unknown" else (0, 0, 255)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.rectangle(frame, (x, y-30), (x+w, y), color, -1)
        cv2.putText(frame, f"{last_name} ({last_confidence}%)", (x+4, y-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # ---- Gesture Prediction ----
    if len(landmarks) == model_data['layer_sizes'][0]:
        output = nn.predict(landmarks)
        confidence = max(output)
        gesture = labels[output.index(confidence)]
        color = (0, 255, 0) if confidence > 0.7 else (0, 165, 255)
        cv2.putText(frame, f"{gesture} ({confidence:.0%})", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        for i, (label, prob) in enumerate(zip(labels, output)):
            bar_width = int(prob * 200)
            cv2.rectangle(frame, (10, 70 + i*25), (10 + bar_width, 90 + i*25), color, -1)
            cv2.putText(frame, f"{label}: {prob:.0%}", (220, 87 + i*25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    else:
        cv2.putText(frame, "Show hand + face", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

    cv2.imshow("Gesture + Face Recognizer", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()