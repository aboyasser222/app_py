import streamlit as st
import numpy as np
from ultralytics import YOLO
from PIL import Image
import requests
import time

LOCAL_URL = "http://127.0.0.1:5000"

st.set_page_config(page_title="Autonomous Navigation System", layout="wide")
st.title("🛥️ Local Detection System: Boat, Rock & Water Hyacinth")

if 'last_action_time' not in st.session_state:
    st.session_state.last_action_time = 0

COOLDOWN = 4.0 

@st.cache_resource
def load_model():
    model = YOLO("codeCar/best (7).pt") 
    model.names[0] = "boat"
    model.names[1] = "rock"
    model.names[2] = "water_hyacinth"
    return model

model = load_model()

camera_input = st.camera_input("Capture image for analysis")

if camera_input is not None:
    image = Image.open(camera_input)
    img_array = np.array(image)
    
    results = model(img_array, conf=0.40)[0]
    
    results.names[0] = "boat"
    results.names[1] = "rock"
    results.names[2] = "water_hyacinth"
    
    st.image(results.plot(), caption='Analysis Results', use_column_width=True)
    
    detections = results.boxes.data.tolist()
    
    boat_detected = any(int(box[5]) == 0 for box in detections)
    rock_detected = any(int(box[5]) == 1 for box in detections)
    hyacinth_detected = any(int(box[5]) == 2 for box in detections)
    
    current_time = time.time()
    
    if rock_detected or boat_detected:
        if current_time - st.session_state.last_action_time > COOLDOWN:
            try:
                requests.get(f"{LOCAL_URL}/move_backward", timeout=2)
                st.session_state.last_action_time = current_time
                st.error("🚨 Hazard Detected! Reversing...")
            except:
                st.error("Server Offline")
    elif hyacinth_detected:
        if current_time - st.session_state.last_action_time > COOLDOWN:
            try:
                requests.get(f"{LOCAL_URL}/move_forward", timeout=2)
                st.session_state.last_action_time = current_time
                st.success("🌿 Target Detected! Moving Forward...")
            except:
                st.error("Server Offline")
