# ============================================================
#  testing.py  –  Quick test: verify the trained model works
#
#  CHANGES MADE:
#  1. Added comments explaining every block
#  2. Added camera read check (ret) to avoid crash on bad frame
#  3. Show confidence score on screen so you can tune threshold
#  4. Press Q to quit (same as main app)
# ============================================================

import cv2
import sys

# Confidence threshold – lower value = stricter matching
# If a face's score is BELOW this number it is considered recognised.
CONF_THRESHOLD = 70

# ── Load trained model ─────────────────────────────────────────────────────
try:
    recognizer = cv2.face.LBPHFaceRecognizer_create()
except AttributeError:
    try:
        recognizer = cv2.face.createLBPHFaceRecognizer()
    except AttributeError:
        print("ERROR: Install opencv-contrib-python first.")
        sys.exit()

recognizer.read('TrainingImageLabel/trainner.yml')

# ── Load face detector ─────────────────────────────────────────────────────
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
font         = cv2.FONT_HERSHEY_SIMPLEX

# ── Open webcam ────────────────────────────────────────────────────────────
cam = cv2.VideoCapture(0)
print("Camera started. Press Q to quit.")

while True:
    ret, frame = cam.read()

    # [CHANGE] Skip the loop iteration if the camera frame is empty
    if not ret:
        print("Camera read failed, retrying...")
        continue

    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.2, 5)

    for (x, y, w, h) in faces:
        student_id, confidence = recognizer.predict(gray[y:y + h, x:x + w])

        if confidence < CONF_THRESHOLD:
            # Green box = recognised
            label = f"ID: {student_id}  Conf: {confidence:.1f}"
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
            cv2.putText(frame, label, (x, y - 10),
                        font, 0.8, (0, 255, 0), 2)
        else:
            # Red box = unknown / not confident enough
            label = f"Unknown  Conf: {confidence:.1f}"
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 3)
            cv2.putText(frame, label, (x, y - 10),
                        font, 0.8, (0, 0, 255), 2)

    cv2.imshow('Face Recognition Test – Press Q to quit', frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
print("Camera closed.")
