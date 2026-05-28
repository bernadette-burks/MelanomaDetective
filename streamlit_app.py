import streamlit as st
from src.model_prediction import predict_image  # Import the predict_image function from model_prediction.py

st.title("Melanoma Detection CNN")

uploaded_file = st.file_uploader(
    "Upload a skin lesion image",
    type=["jpg", "png"]
)

if uploaded_file:
    st.image(uploaded_file)

    prediction = predict_image(...)

    st.write(f"Prediction: {prediction}")