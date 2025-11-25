import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing import image

# App title
st.title("😷 Face Mask Detection")
st.write("Detects whether a person is **wearing a mask** or **not wearing a mask**.")

# Load model with caching
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model("face-mask-detection-final.keras")
        return model
    except Exception as e:
        st.error(f"Failed to load model. Error: {e}")
        return None

model = load_model()
if model is None:
    st.stop()

# Class names (must match model training order)
class_names = ['With Mask 😷', 'Without Mask ❌']

# Image preprocessing
def preprocess_image(img):
    img = img.convert("RGB")
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

# File uploader
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image_data = Image.open(uploaded_file)
    st.image(image_data, caption="Uploaded Image", use_column_width=True)

    input_img = preprocess_image(image_data)

    # Prediction
    try:
        prediction = model.predict(input_img, verbose=0)[0][0]   # sigmoid value 0-1
        
        if prediction < 0.4:
            pred_class=0 
            confidence=(1-prediction)*100
        else :
            pred_class=1
            confidence=prediction*100

        


        st.write(f"### 🔍 Prediction: **{class_names[pred_class]}**")
        st.write(f"Confidence: `{confidence:.2f}%`")
        st.progress(int(confidence))
    except Exception as e:
        st.error(f"Prediction error: {e}")
