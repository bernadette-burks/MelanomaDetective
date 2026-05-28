import streamlit as st
from PIL import Image
import torch

from src.model_prediction import predict_image
from src.model_training import SimpleCNN

# Page title
st.title("Melanoma Detective - Skin Lesion Classification")

# Upload image
uploaded_file = st.file_uploader(
    "Upload a skin lesion image",
    type=["jpg", "jpeg", "png"]
)

# If image uploaded
if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image")

    # Load model
    model = SimpleCNN()

    model.load_state_dict(
        torch.load(
            "models/melanoma_cnn_model.pth",
            map_location="cpu"
        )
    )

    # Run prediction
    prediction = predict_image(model, image)

    # Display result
    st.subheader(f"Prediction: {prediction}")