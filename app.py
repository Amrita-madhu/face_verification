
import streamlit as st
import cv2
import numpy as np
import joblib

from pathlib import Path
from skimage.feature import hog


# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Face / Non-Face Classifier",
    page_icon="🔍",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "results" / "hybrid_model.joblib"

IMAGE_SIZE = (64, 64)


# --------------------------------------------------
# 2. LOAD TRAINED MODEL
# --------------------------------------------------

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found at: {MODEL_PATH}\n"
            "Run python main.py first."
        )

    saved_model = joblib.load(MODEL_PATH)

    required_keys = {"classifier", "scaler"}
    missing_keys = required_keys - set(saved_model.keys())

    if missing_keys:
        raise ValueError(
            f"The saved model is incompatible. Missing keys: "
            f"{missing_keys}. Run python main.py again."
        )

    return saved_model


try:
    model = load_model()
except Exception as error:
    st.error(f"Could not load the trained model: {error}")
    st.stop()


# --------------------------------------------------
# 3. EXTRACT HOG FEATURES
# Must match main.py exactly
# --------------------------------------------------

def extract_features(image):
    """
    Convert an image to grayscale, resize it to 64x64,
    and extract full-image HOG features.
    """

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, IMAGE_SIZE)

    features = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True
    )

    return np.asarray(
        features,
        dtype=np.float32
    ).reshape(1, -1)


# --------------------------------------------------
# 4. PREDICT FACE / NON-FACE
# --------------------------------------------------

def predict_image(image, saved_model):
    classifier = saved_model["classifier"]
    scaler = saved_model["scaler"]

    features = extract_features(image)

    scaled_features = scaler.transform(features)

    prediction = int(
        classifier.predict(scaled_features)[0]
    )

    probabilities = classifier.predict_proba(
        scaled_features
    )[0]

    classes = list(classifier.classes_)

    if 1 not in classes:
        raise ValueError(
            "The trained model does not contain the Face class."
        )

    face_index = classes.index(1)
    face_score = float(probabilities[face_index])

    return prediction, face_score


# --------------------------------------------------
# 5. DISPLAY PREDICTION
# --------------------------------------------------

def display_prediction(uploaded_image):
    file_bytes = np.frombuffer(
        uploaded_image.getvalue(),
        dtype=np.uint8
    )

    image = cv2.imdecode(
        file_bytes,
        cv2.IMREAD_COLOR
    )

    if image is None:
        st.error(
            "Unable to read the image. Please try another image."
        )
        return

    st.image(
        cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
        caption="Input image",
        use_container_width=True
    )

    try:
        with st.spinner("Analyzing image..."):
            prediction, face_score = predict_image(
                image,
                model
            )

        st.subheader("Prediction Results")

        if prediction == 1:
            st.success("Prediction: FACE")
        else:
            st.warning("Prediction: NON-FACE")

        st.metric(
            label="Model face score",
            value=f"{face_score:.1%}"
        )

        st.progress(
            max(0.0, min(1.0, face_score))
        )

        st.caption(
            "The score is the classifier's estimated probability "
            "for the Face class. It is not a guarantee of correctness."
        )

    except Exception as error:
        st.error(f"Prediction failed: {error}")


# --------------------------------------------------
# 6. APPLICATION USER INTERFACE
# --------------------------------------------------

st.title("🔍 Face / Non-Face Classifier")

st.write(
    "Image classification using HOG feature extraction, "
    "StandardScaler, and Logistic Regression."
)

st.info(
    "This academic prototype classifies an entire image as "
    "Face or Non-Face. It does not identify a person or "
    "verify their identity."
)

tab_upload, tab_camera = st.tabs(
    ["📁 Upload Image", "📷 Webcam Capture"]
)


# --------------------------------------------------
# 7. IMAGE UPLOAD TAB
# --------------------------------------------------

with tab_upload:
    st.subheader("Upload an image")

    uploaded_file = st.file_uploader(
        "Choose a JPG, JPEG, or PNG image",
        type=["jpg", "jpeg", "png"],
        key="face_image_upload"
    )

    if uploaded_file is not None:
        display_prediction(uploaded_file)



# --------------------------------------------------
# 8 WEBCAM CAPTURE TAB
# --------------------------------------------------

with tab_camera:
    st.subheader("Webcam Face Classification")

    st.write(
        "Click the button below when you want to use your camera."
    )

    if "camera_enabled" not in st.session_state:
        st.session_state.camera_enabled = False

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Start Camera"):
            st.session_state.camera_enabled = True

    with col2:
        if st.button("Stop Camera"):
            st.session_state.camera_enabled = False

    if st.session_state.camera_enabled:
        camera_image = st.camera_input(
            "Capture an image",
            key="face_camera_capture"
        )

        if camera_image is not None:
            display_prediction(camera_image)
    else:
        st.info(
            "Camera is off. Click Start Camera to enable camera capture."
        )



# --------------------------------------------------
# 9. FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Academic prototype | HOG + StandardScaler + "
    "Logistic Regression"
)

