import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# Load model
model = load_model("galaxy_cnn_model.h5")

# Class names
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


st.title("Galaxy Morphology Classifier")
st.write("Upload a galaxy image and let the CNN guess its type!")

st.markdown("🌌 Need to find a galaxy image? Download one from [Unsplash](https://unsplash.com/s/photos/universe)")
# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).resize((69, 69))
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Preprocess
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    
    # Predict
    prediction = model.predict(image_array)
    class_idx = np.argmax(prediction)
    
    st.success(f"Predicted class: **{class_names[class_idx]}**")
