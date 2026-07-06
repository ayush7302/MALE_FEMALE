import streamlit as st
import numpy as np
import cv2
import joblib

# -------------------------
# Page Configuration
# -------------------------

st.set_page_config(
    page_title="female vs male Classifier",
    page_icon="🐶",
    layout="centered"
)

# -------------------------
# Load Model
# -------------------------

model = joblib.load("female_male_model.pkl")

IMG_SIZE = 64

st.title("🐱 male and female Image Classifier")

st.write("Upload an image to predict whether it is a female  or male.")

uploaded_file = st.file_uploader(
    "Choose an Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)

    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    st.image(image, channels="BGR", width=300)

    resized = cv2.resize(image, (IMG_SIZE, IMG_SIZE))

    resized = resized.flatten()

    prediction = model.predict([resized])[0]

    probability = model.predict_proba([resized])[0]

    if prediction == 0:
        st.success("🐱 Prediction: female")
    else:
        st.success("🐶 Prediction: male")

    st.write(f"Cat Probability : {probability[0]*100:.2f}%")
    st.write(f"Dog Probability : {probability[1]*100:.2f}%")
