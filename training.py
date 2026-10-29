# ============================================================
#  training.py  –  Train the LBPH Face Recognition model
#
#  CHANGES MADE:
#  1. Added clear comments explaining every step
#  2. Wrapped ID parsing in try/except so one bad filename
#     doesn't crash the whole training run
#  3. Print progress so you can see what's happening
#  4. Added a check: warn if no face images were found
# ============================================================

import cv2
import os
import numpy as np
from PIL import Image

# ── Step 1: Create the recognizer ─────────────────────────────────────────
# LBPH = Local Binary Patterns Histogram
# It learns what each person's face looks like from your training images.
# Requires:  pip install opencv-contrib-python
try:
    recognizer = cv2.face.LBPHFaceRecognizer_create()
except AttributeError:
    try:
        recognizer = cv2.face.createLBPHFaceRecognizer()
    except AttributeError:
        print("ERROR: cv2.face module not found.")
        print("Fix:   pip install opencv-contrib-python")
        exit()

# ── Step 2: Load the face detector (Haar Cascade) ─────────────────────────
detector = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")


def getImagesAndLabels(path):
    """
    Reads every image in the given folder.
    Returns:
        face_samples – list of cropped grayscale face arrays
        ids          – list of integer IDs matching each face
    """
    image_paths  = [os.path.join(path, f) for f in os.listdir(path)]
    face_samples = []
    ids          = []

    for image_path in image_paths:
        # Open image in grayscale
        pil_image = Image.open(image_path).convert('L')
        image_np  = np.array(pil_image, 'uint8')

        # Filename format: Name.Enrollment.SampleNumber.jpg  → we want index 1
        try:
            student_id = int(os.path.split(image_path)[-1].split(".")[1])
        except (ValueError, IndexError):
            # [CHANGE] Skip files with unexpected names instead of crashing
            print(f"  Skipping (unexpected filename): {image_path}")
            continue

        faces = detector.detectMultiScale(image_np)
        for (x, y, w, h) in faces:
            face_samples.append(image_np[y:y + h, x:x + w])
            ids.append(student_id)

    return face_samples, ids


# ── Run training ───────────────────────────────────────────────────────────
print("Loading training images...")
faces, ids = getImagesAndLabels('TrainingImage')

if len(faces) == 0:
    # [CHANGE] Friendly error instead of a confusing crash
    print("\nERROR: No face images found in 'TrainingImage' folder.")
    print("Run 'Take Images' from the main app first.")
    exit()

print(f"Found {len(faces)} face samples for {len(set(ids))} student(s).")
print("Training... (may take a few minutes)")

recognizer.train(faces, np.array(ids))
recognizer.save('TrainingImageLabel/trainner.yml')
print("Done! Model saved to TrainingImageLabel/trainner.yml")
