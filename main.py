

import cv2
import time
import joblib
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

from skimage.feature import hog

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)


# --------------------------------------------------
# 1. PROJECT CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

FACE_DIR = BASE_DIR / "dataset" / "face"
NON_FACE_DIR = BASE_DIR / "dataset" / "non_face"
RESULTS_DIR = BASE_DIR / "results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = (64, 64)
RANDOM_STATE = 42

print("=" * 55)
print("FACE / NON-FACE CLASSIFIER")
print("HOG + STANDARD SCALER + LOGISTIC REGRESSION")
print("=" * 55)


# --------------------------------------------------
# 2. LOAD LABELED IMAGES
# --------------------------------------------------

def load_images(folder, label):
    samples = []

    if not folder.exists():
        print(f"Warning: Folder not found: {folder}")
        return samples

    valid_extensions = {
        ".jpg", ".jpeg", ".png", ".bmp"
    }

    for path in sorted(folder.iterdir()):

        if path.suffix.lower() not in valid_extensions:
            continue

        image = cv2.imread(
            str(path),
            cv2.IMREAD_GRAYSCALE
        )

        if image is None:
            print(f"Skipping unreadable image: {path.name}")
            continue

        image = cv2.resize(image, IMAGE_SIZE)

        samples.append((image, label))

    return samples


samples = (
    load_images(FACE_DIR, 1)
    + load_images(NON_FACE_DIR, 0)
)

if not samples:
    raise RuntimeError(
        "No images found. Check dataset/face and dataset/non_face."
    )

images = np.array(
    [sample[0] for sample in samples],
    dtype=np.uint8
)

labels = np.array(
    [sample[1] for sample in samples],
    dtype=np.int32
)

face_count = int(np.sum(labels == 1))
non_face_count = int(np.sum(labels == 0))

print(f"Total images: {len(images)}")
print(f"Face images: {face_count}")
print(f"Non-face images: {non_face_count}")

if min(face_count, non_face_count) < 2:
    raise RuntimeError(
        "At least two images per class are required."
    )


# --------------------------------------------------
# 3. SPLIT INTO TRAINING AND TESTING DATA
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    images,
    labels,
    test_size=0.25,
    random_state=RANDOM_STATE,
    stratify=labels
)

print(f"Training images: {len(X_train)}")
print(f"Testing images: {len(X_test)}")


# --------------------------------------------------
# 4. EXTRACT FULL-IMAGE HOG FEATURES
# --------------------------------------------------

def extract_features(image):
    """
    Extract HOG features from a grayscale image.
    Both training and prediction use the same settings.
    """

    image = cv2.resize(image, IMAGE_SIZE)

    features = hog(
        image,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True
    )

    return np.asarray(
        features,
        dtype=np.float32
    )


print("\nExtracting HOG features...")

start_time = time.perf_counter()

X_train_features = np.vstack([
    extract_features(image)
    for image in X_train
])

X_test_features = np.vstack([
    extract_features(image)
    for image in X_test
])

print(
    "Training feature shape:",
    X_train_features.shape
)

print(
    "Testing feature shape:",
    X_test_features.shape
)


# --------------------------------------------------
# 5. STANDARDIZE FEATURES
# --------------------------------------------------

print("\nScaling features...")

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train_features
)

X_test_scaled = scaler.transform(
    X_test_features
)


# --------------------------------------------------
# 6. TRAIN LOGISTIC REGRESSION
# --------------------------------------------------

print("\nTraining Logistic Regression...")

classifier = LogisticRegression(
    C=1.0,
    solver="liblinear",
    class_weight="balanced",
    max_iter=3000,
    random_state=RANDOM_STATE
)

classifier.fit(
    X_train_scaled,
    y_train
)

y_pred = classifier.predict(
    X_test_scaled
)

elapsed = time.perf_counter() - start_time


# --------------------------------------------------
# 7. EVALUATE THE MODEL
# --------------------------------------------------

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    pos_label=1,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    pos_label=1,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    pos_label=1,
    zero_division=0
)

print("\n" + "=" * 55)
print("MODEL EVALUATION")
print("=" * 55)

print(f"Accuracy       : {accuracy:.4f}")
print(f"Precision      : {precision:.4f}")
print(f"Recall         : {recall:.4f}")
print(f"F1-score       : {f1:.4f}")
print(f"Training time  : {elapsed:.3f} seconds")

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["Non-Face", "Face"],
        zero_division=0
    )
)


# --------------------------------------------------
# 8. GENERATE CONFUSION MATRIX
# --------------------------------------------------

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=[0, 1]
)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Non-Face", "Face"]
)

display.plot(
    cmap="Blues",
    values_format="d"
)

plt.title(
    "Face / Non-Face Classification"
)

plt.tight_layout()

confusion_path = (
    RESULTS_DIR / "confusion_matrix.png"
)

plt.savefig(
    confusion_path,
    dpi=200,
    bbox_inches="tight"
)

plt.close()

print(f"\nConfusion matrix saved: {confusion_path}")


# --------------------------------------------------
# 9. SAVE EVALUATION METRICS
# --------------------------------------------------

metrics_path = RESULTS_DIR / "metrics.txt"

with open(
    metrics_path,
    "w",
    encoding="utf-8"
) as file:

    file.write("Face / Non-Face Classification Results\n")
    file.write("=" * 45 + "\n")

    file.write(f"Total images: {len(images)}\n")
    file.write(f"Face images: {face_count}\n")
    file.write(f"Non-face images: {non_face_count}\n")

    file.write(f"Training images: {len(X_train)}\n")
    file.write(f"Testing images: {len(X_test)}\n")

    file.write(f"Image size: {IMAGE_SIZE}\n")
    file.write("Feature extraction: Full-image HOG\n")
    file.write("Feature scaling: StandardScaler\n")
    file.write("Classifier: Logistic Regression\n")
    file.write("Regularization: L2\n")
    file.write("C: 1.0\n")

    file.write(f"Accuracy: {accuracy:.4f}\n")
    file.write(f"Precision: {precision:.4f}\n")
    file.write(f"Recall: {recall:.4f}\n")
    file.write(f"F1-score: {f1:.4f}\n")
    file.write(f"Elapsed seconds: {elapsed:.3f}\n")

    file.write(
        "\nConfusion matrix (Non-Face, Face):\n"
    )

    file.write(f"{cm}\n")

print(f"Metrics saved: {metrics_path}")


# --------------------------------------------------
# 10. SAVE THE TRAINED MODEL
# --------------------------------------------------

model_path = RESULTS_DIR / "hybrid_model.joblib"

model_data = {
    "scaler": scaler,
    "classifier": classifier,
    "image_size": IMAGE_SIZE,
    "feature_method": (
        "Full-image HOG + StandardScaler "
        "+ Logistic Regression"
    ),
    "class_labels": {
        0: "Non-Face",
        1: "Face"
    }
}

joblib.dump(
    model_data,
    model_path
)

print("\n" + "=" * 55)
print("TRAINING COMPLETED SUCCESSFULLY")
print("=" * 55)

print(f"Trained model: {model_path}")
print(f"Metrics: {metrics_path}")
print(f"Confusion matrix: {confusion_path}")

print("\nYou can now run the Streamlit application.")
