import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

st.set_page_config(page_title="Brain Tumor Classifier", layout="centered")
st.title("🧠 Brain Tumor MRI Image Classification")


# 1. Initialize the model variable in the global scope
model = None

@st.cache_resource
def load_production_model():
    # Double-check this path exactly matches your repository tree
    model_file = 'models/tlm.keras' 
    return tf.keras.models.load_model(model_file, compile=False, safe_mode=False)

try:
    model = load_production_model()
    st.sidebar.success("Native .keras model loaded safely!")
except Exception as e:
    st.sidebar.error("❌ Failed to load the model file.")
    # This will print the exact underlying error to your Hugging Face logs
    st.sidebar.exception(e) 

CLASS_NAMES = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

uploaded_file = st.file_uploader("Upload Brain MRI Image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption='Uploaded MRI', use_container_width=True)
        
    with col2:
        # 2. Strict Guard: Only attempt prediction if the model successfully initialized
        if model is None:
            st.error("⚠️ Prediction unavailable because the machine learning model failed to load. Please check the sidebar error details.")
        else:
            st.write("🔄 **Analyzing MRI scan...**")
            
            img_resized = image.resize((224, 224))
            img_array = np.array(img_resized)
            img_normalized = img_array / 255.0  
            img_tensor = np.expand_dims(img_normalized, axis=0) 
            
            # This line will no longer throw a NameError
            predictions = model.predict(img_tensor)
            best_index = np.argmax(predictions[0])
            predicted_class = CLASS_NAMES[best_index]
            confidence_score = predictions[0][best_index] * 100
            
            st.success(f"Prediction: **{predicted_class}**")
            st.info(f"Confidence: **{confidence_score:.2f}%**")
