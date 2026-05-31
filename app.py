import os
import requests
import streamlit as st
import tensorflow as tf

@st.cache_resource
def load_production_model():
    # Target file path on Streamlit Cloud's local directory
    local_model_path = "tlm_runtime.keras"
    
    if not os.path.exists(local_model_path):
        with st.spinner("📥 Downloading AI Model directly from Google Drive... Please wait."):
            # Your verified Google Drive File ID
            FILE_ID = "15crJHVBXoqqQh9GAmOJzKRJ3ktdHeRqB"
            
            # Base URL for the Google Drive download API
            URL = "https://docs.google.com/uc?export=download"
            
            # Start a persistent session to handle security cookies
            session = requests.Session()
            
            # First request to check for large-file confirmation page
            response = session.get(URL, params={'id': FILE_ID}, stream=True)
            
            # Try to extract the confirmation token if Google Drive prompts a warning page
            token = None
            for key, value in response.cookies.items():
                if key.startswith('download_warning'):
                    token = value
                    break
            
            # If a confirmation token is required, send a second request with the token included
            if token:
                params = {'id': FILE_ID, 'confirm': token}
                response = session.get(URL, params=params, stream=True)
            
            # Save the raw binary chunks to the server disk
            if response.status_code == 200:
                with open(local_model_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                st.success("🤖 Model binary pulled successfully from Google Drive!")
            else:
                st.error(f"❌ Download failed. Google Drive API Status: {response.status_code}")
                return None
                
    # Unpack the verified binary file zip archive smoothly
    return tf.keras.models.load_model(local_model_path, compile=False, safe_mode=False)
