"""
Face Detection — Streamlit App
-----------------------------------
Uses OpenCV's pretrained Haar Cascade classifier to detect and draw boxes
around every human face in an uploaded photo. From Project 3 of the
"AI Playground: 4 Real-World AI Projects" notebook.

Run locally:
    pip install -r requirements.txt
    streamlit run app.py
"""

import io
import urllib.request

import cv2
import numpy as np
import streamlit as st
from PIL import Image


# ------------------------------------------------------------------
# Page config
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Face Detection AI",
    page_icon="👤",
    layout="centered",
)


# ------------------------------------------------------------------
# Load the pretrained Haar Cascade detector once, cache it
# ------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading face detector...")
def load_detector():
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(cascade_path)
    return detector


FALLBACK_FACE_URL = "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg"


@st.cache_data(show_spinner=False)
def load_fallback_image():
    with urllib.request.urlopen(FALLBACK_FACE_URL) as response:
        return Image.open(io.BytesIO(response.read())).convert("RGB")


def detect_faces(pil_image, detector, scale_factor=1.1, min_neighbors=5):
    """Run Haar Cascade face detection and return an annotated PIL image + count."""
    # PIL -> OpenCV (RGB -> BGR)
    bgr_image = cv2.cvtColor(np.array(pil_image.convert("RGB")), cv2.COLOR_RGB2BGR)
    gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)

    faces = detector.detectMultiScale(
        gray_image,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        minSize=(30, 30),
    )

    output_image = bgr_image.copy()
    for (x, y, w, h) in faces:
        cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 0), 3)

    # OpenCV (BGR) -> PIL (RGB)
    annotated = Image.fromarray(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
    return annotated, len(faces), faces


# ------------------------------------------------------------------
# UI
# ------------------------------------------------------------------
st.title("👤 Face Detection AI")
st.caption(
    "Powered by OpenCV's Haar Cascade classifier — a pretrained detector "
    "that scans an image for the visual patterns (edges, contrast regions) "
    "typical of a frontal human face, no deep learning required."
)

detector = load_detector()

st.subheader("Detector settings")
col_a, col_b = st.columns(2)
with col_a:
    scale_factor = st.slider(
        "Scale factor", min_value=1.05, max_value=1.5, value=1.1, step=0.05,
        help="How much the image size is reduced at each scan step. "
             "Smaller = more accurate but slower.",
    )
with col_b:
    min_neighbors = st.slider(
        "Min neighbors", min_value=1, max_value=10, value=5, step=1,
        help="How many overlapping detections are required to confirm a face. "
             "Higher = fewer false positives, but may miss real faces.",
    )

st.subheader("Provide an image")
uploaded_file = st.file_uploader("Upload a photo", type=["jpg", "jpeg", "png", "webp"])

use_sample = False
if uploaded_file is None:
    use_sample = st.checkbox("No photo handy? Use a sample image instead")

image = None
if uploaded_file is not None:
    image = Image.open(uploaded_file)
elif use_sample:
    with st.spinner("Downloading sample image..."):
        image = load_fallback_image()

if image is not None:
    with st.spinner("Detecting faces..."):
        annotated_image, face_count, faces = detect_faces(
            image, detector, scale_factor, min_neighbors
        )

    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Original", use_container_width=True)
    with col2:
        st.image(annotated_image, caption="Detected faces", use_container_width=True)

    if face_count == 0:
        st.warning("No faces detected. Try adjusting the sliders above, or use a clearer, front-facing photo.")
    else:
        st.success(f"**{face_count}** face(s) detected.")
        with st.expander("Bounding box coordinates"):
            for i, (x, y, w, h) in enumerate(faces, start=1):
                st.write(f"Face {i}: x={x}, y={y}, width={w}, height={h}")

    with st.expander("What actually happened here?"):
        st.write(
            "The image was first converted to grayscale, since Haar Cascades "
            "look for patterns of light and dark contrast rather than color. "
            "The detector then slides a window across the image at many scales, "
            "checking each region against thousands of simple learned features. "
            "A region is only confirmed as a face once enough overlapping windows "
            "agree — that's what 'min neighbors' controls. This is detection, not "
            "recognition: the model finds where faces are, but has no idea whose "
            "faces they are."
        )
else:
    st.info("Upload a photo above, or check the box to try a sample image.")

st.divider()
st.caption(
    "Built from Project 3 of *AI Playground: 4 Real-World AI Projects*. "
    "Haar Cascades are fast and lightweight, but less robust than modern deep "
    "learning face detectors on side angles, low light, or unusual poses."
)
