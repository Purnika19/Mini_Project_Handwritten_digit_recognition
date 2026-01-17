import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="Digit Recognizer",
    page_icon="✍️",
    layout="centered"
)

# ------------------ CUSTOM CSS ------------------
st.markdown("""
<style>

/* Main background */
.stApp {
    background: linear-gradient(135deg, #0f172a, #020617);
}

/* Headings */
h1 {
    color: #38bdf8;
    text-align: center;
    font-weight: 700;
}
h2 {
    color: #c7d2fe;
    text-align: center;
    font-weight: 400;
}

/* Upload box */
.upload-box {
    border: 2px dashed #22d3ee;
    padding: 22px;
    border-radius: 16px;
    background: linear-gradient(145deg, #020617, #020617);
    box-shadow: 0 0 15px rgba(34, 211, 238, 0.15);
}

/* Prediction box */
.pred-box {
    background: linear-gradient(135deg, #1e293b, #020617);
    padding: 28px;
    border-radius: 18px;
    text-align: center;
    font-size: 30px;
    margin-top: 28px;
    color: #22d3ee;
    font-weight: bold;
    box-shadow: 0 0 20px rgba(56, 189, 248, 0.25);
}

/* Uploaded image border */
img {
    border-radius: 12px;
    border: 2px solid #38bdf8;
}

/* Footer */
.footer {
    text-align: center;
    color: #94a3b8;
    font-size: 14px;
    margin-top: 10px;
}

/* File uploader text */
section[data-testid="stFileUploader"] label {
    color: #e0f2fe !important;
    font-weight: 500;
}

/* Buttons hover (future-proof) */
button:hover {
    background-color: #38bdf8 !important;
    color: black !important;
}

</style>
""", unsafe_allow_html=True)


# ------------------ LOAD MODEL ------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("digit_recognition_model.h5")

model = load_model()

# ------------------ TITLE ------------------
st.markdown("<h1>✍️ Handwritten Digit Recognizer</h1>", unsafe_allow_html=True)
st.markdown("<h2>Upload a digit image to predict</h2>", unsafe_allow_html=True)

# ------------------ UPLOAD ------------------
st.markdown("<div class='upload-box'>", unsafe_allow_html=True)
uploaded_file = st.file_uploader(
    "📤 Upload a handwritten digit image (PNG / JPG)",
    type=["png", "jpg", "jpeg"]
)
st.markdown("</div>", unsafe_allow_html=True)

# ------------------ PROCESS ------------------
if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("L")
    st.image(img, caption="Uploaded Image", width=200)

    # Preprocessing
    img = img.resize((28, 28))
    img_array = np.array(img)
    img_array = 255 - img_array
    img_flat = img_array.reshape(1, 784)

    # Normalize
    img_norm = (img_flat - np.mean(img_flat)) / (np.std(img_flat) + 1e-10)

    # Predict
    prediction = model.predict(img_norm)
    digit = np.argmax(prediction)

    # ------------------ RESULT ------------------
    st.markdown(
        f"<div class='pred-box'> Predicted Digit: <b>{digit}</b></div>",
        unsafe_allow_html=True
    )

# ------------------ FOOTER ------------------
st.markdown("---")
st.markdown(
    "<div class='footer'>Built by Purnika Malhotra</div>",
    unsafe_allow_html=True
)
