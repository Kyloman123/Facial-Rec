import cv2
import os
import numpy as np

FACES_DIR = '/Users/gulickg/Desktop/known_faces'

recognizer = cv2.face.LBPHFaceRecognizer_create()

faces = []
labels = []
label_map = {}
current_label = 0

print("Loading face data...")
for person_name in os.listdir(FACES_DIR):
    person_dir = f"{FACES_DIR}/{person_name}"
    if not os.path.isdir(person_dir):
        continue

    label_map[current_label] = person_name
    print(f"  Loading {person_name}...")

    for img_file in os.listdir(person_dir):
        if img_file.endswith('.jpg'):
            img_path = f"{person_dir}/{img_file}"
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                faces.append(img)
                labels.append(current_label)

    current_label += 1

print(f"\nTraining on {len(faces)} images across {len(label_map)} people...")
recognizer.train(faces, np.array(labels))
recognizer.save('/Users/gulickg/Desktop/face_model.yml')

import json
with open('/Users/gulickg/Desktop/face_labels.json', 'w') as f:
    json.dump({str(k): v for k, v in label_map.items()}, f)

print("Done! Saved face_model.yml and face_labels.json")
print("Now run recognizer.py!")