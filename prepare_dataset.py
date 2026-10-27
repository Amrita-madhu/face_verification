
from pathlib import Path
import pickle
import cv2
import numpy as np
import random

BASE_DIR = Path(__file__).resolve().parent
CIFAR_DIR = BASE_DIR / "dataset" / "cifar-10-batches-py"
NON_FACE_DIR = BASE_DIR / "dataset" / "non_face"

NON_FACE_DIR.mkdir(parents=True, exist_ok=True)

# CIFAR-10 labels: airplane=0, automobile=1, ship=8, truck=9
ALLOWED_LABELS = {0, 1, 8, 9}
MAX_IMAGES = 500
random.seed(42)

def load_batch(path):
    with open(path, "rb") as file:
        return pickle.load(file, encoding="bytes")

if not CIFAR_DIR.exists():
    raise FileNotFoundError(f"CIFAR-10 folder not found: {CIFAR_DIR}")

candidates = []

for batch_number in range(1, 6):
    batch_path = CIFAR_DIR / f"data_batch_{batch_number}"
    batch = load_batch(batch_path)

    data = batch[b"data"]
    labels = batch[b"labels"]

    for image_data, label in zip(data, labels):
        if label in ALLOWED_LABELS:
            image = image_data.reshape(3, 32, 32).transpose(1, 2, 0)
            candidates.append(image)

random.shuffle(candidates)
selected = candidates[:MAX_IMAGES]

# Remove previously generated CIFAR images only.
for old_image in NON_FACE_DIR.glob("cifar_*.jpg"):
    old_image.unlink()

saved = 0
for index, image in enumerate(selected, start=1):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image = cv2.resize(image, (64, 64))
    output_path = NON_FACE_DIR / f"cifar_{index:04d}.jpg"

    if cv2.imwrite(str(output_path), image):
        saved += 1

print("Dataset preparation complete.")
print(f"Candidate non-face images saved: {saved}")
print("Review labels before model evaluation.")
