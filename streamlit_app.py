
import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

st.title("Galaxy Morphology Classifier")
st.write("Upload a galaxy image and let the CNN predict its type!")



uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB").resize((69, 69))
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Convert image to array
        image_array = np.array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)

        # Display image shape for debugging
        st.write("Image array shape:", image_array.shape)

        # Load model
        model = load_model("galaxy_cnn_model.h5")

        # Define class names
        class_names = [
            "Completely round smooth",
            "In-between smooth",
            "Cigar-shaped smooth",
            "Barred spiral",
            "Unbarred spiral",
            "Edge-on without bulge",
            "Edge-on with bulge",
            "Spiral arms",
            "Irregular",
            "Merger"
        ]

        # Make prediction
        prediction = model.predict(image_array)
        predicted_class = class_names[np.argmax(prediction)]

        st.success(f"Predicted class: **{predicted_class}**")

    except Exception as e:
        st.error(f"Prediction failed. Error: {e}")
