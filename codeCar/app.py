import streamlit as st
import numpy as np
from ultralytics import YOLO
from PIL import Image
import requests

NGROK_URL = "https://autistic-revenge-unending.ngrok-free.dev"

st.set_page_config(page_title="Autonomous Navigation System", layout="wide")
st.title("🛥️ Boat & Water Hyacinth Detection System")

@st.cache_resource
def load_model():
    model = YOLO("best (5).pt")
    model.names[0] = "boat"
    model.names[1] = "water_hyacinth"
    return model

model = load_model()

camera_input = st.camera_input("Capture image for analysis")

if camera_input is not None:
    image = Image.open(camera_input)
    img_array = np.array(image)
    
    results = model(img_array, conf=0.77)[0]
    
    results.names[0] = "boat"
    results.names[1] = "water_hyacinth"
    
    st.image(results.plot(), caption='Analysis Results', use_column_width=True)
    
    detections = results.boxes.data.tolist()
    
    boat_detected = any(int(box[5]) == 0 for box in detections)
    hyacinth_detected = any(int(box[5]) == 1 for box in detections)
    
    if boat_detected:
        st.error("🚨 Boat Detected! Reversing and changing course...")
        try:
            response = requests.get(f"{NGROK_URL}/move_backward", timeout=5)
            if response.status_code == 200:
                st.info("Action: Move Backward command sent")
        except:
            st.error("Connection Error: Check ngrok and Flask server")
            
    elif hyacinth_detected:
        st.success("🌿 Water Hyacinth Detected! Moving forward...")
        try:
            response = requests.get(f"{NGROK_URL}/move_forward", timeout=5)
            if response.status_code == 200:
                st.info("Action: Move Forward command sent")
        except:
            st.error("Connection Error: Check ngrok and Flask server")
            
    else:
        st.warning("⚠️ Clear path - No targets detected")
