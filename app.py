import os
import requests
import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

st.set_page_config(page_title="Brain Tumor Classifier", layout="centered")
st.title("🧠 Brain Tumor MRI Image Classification")

model = None

@st.cache_resource
def load_production_model():
    # Local path name on the Streamlit Linux cloud container
    local_model_path = "tlm.keras"
    
    if not os.path.exists(local_model_path):
        with st.spinner("📥 Downloading AI Model directly from Google Drive... Please wait."):
            # 💡 REPLACE the ID below with your actual Google Drive File ID!
            FILE_ID = "15crJHVBXoqqQh9GAmOJzKRJ3ktdHeRqB" 
            direct_download_url = f"https://docs.google.com/uc?export=download&id={FILE_ID}"
            
            # Use requests to download the large file chunks smoothly
            response = requests.get(direct_download_url, stream=True)
            if response.status_code == 200:
                with open(local_model_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                st.success("🤖 Model binary pulled successfully from Google Drive!")
            else:
                st.error(f"❌ Failed to download from Google Drive. Status Code: {response.status_code}")
                return None
            
    return tf.keras.models.load_model(local_model_path, compile=False, safe_mode=False)

try:
    model = load_production_model()
    if model is not None:
        st.sidebar.success("✅ Deep learning model active!")
except Exception as e:
    st.sidebar.error("❌ Model Initialization Failed.")
    st.sidebar.exception(e)

# --- Keep the rest of your UI and prediction logic exactly the same ---
