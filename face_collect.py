import cv2
import os

# ---- Setup ----
FACES_DIR = '/Users/gulickg/Desktop/known_faces'
os.makedirs(FACES_DIR, exist_ok=True)

cap = cv2.VideoCapture(1)

print("=== Face Collector ===")
name = input("Enter person's name: ").strip()
save_dir = f"{FACES_DIR}/{name}"
os.makedirs(save_dir, exist_ok=True)

count = 0
MAX_SAMPLES = 50

print(f"Collecting {MAX_SAMPLES} face samples for {name}...")
print("Look at the camera, move your head slightly side to side.")
print("Press Q to stop early.\n")

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while count < MAX_SAMPLES:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        face_img = gray[y:y+h, x:x+w]
        face_img = cv2.resize(face_img, (150, 150))
        cv2.imwrite(f"{save_dir}/{count}.jpg", face_img)
        count += 1
        cv2.putText(frame, f"Captured: {count}/{MAX_SAMPLES}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Face Collector", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f"\nSaved {count} samples for {name} to {save_dir}")
print("Run face_collect.py again to add another person, then run face_train.py")