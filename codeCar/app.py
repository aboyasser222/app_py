import streamlit as st
import numpy as np
from ultralytics import YOLO
import requests
import time
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

NGROK_URL = "https://autistic-revenge-unending.ngrok-free.dev"

st.set_page_config(page_title="Real-time Navigation System", layout="wide")
st.title("🛥️ Live Detection System: Boat, Rock & Water Hyacinth")

@st.cache_resource
def load_model():
    model = YOLO("codeCar/best (7).pt") 
    model.names[0] = "boat"
    model.names[1] = "rock"
    model.names[2] = "water_hyacinth"
    return model

model = load_model()

if 'last_action_time' not in st.session_state:
    st.session_state.last_action_time = 0

COOLDOWN = 4.0

class VideoProcessor(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        results = model(img, conf=0.40)[0]
        results.names[0] = "boat"
        results.names[1] = "rock"
        results.names[2] = "water_hyacinth"
        
        annotated_img = results.plot()
        
        detections = results.boxes.data.tolist()
        boat_detected = any(int(box[5]) == 0 for box in detections)
        rock_detected = any(int(box[5]) == 1 for box in detections)
        hyacinth_detected = any(int(box[5]) == 2 for box in detections)
        
        current_time = time.time()
        
        if rock_detected or boat_detected:
            if current_time - st.session_state.last_action_time > COOLDOWN:
                try:
                    requests.get(f"{NGROK_URL}/move_backward", timeout=1)
                    st.session_state.last_action_time = current_time
                except:
                    pass
        elif hyacinth_detected:
            if current_time - st.session_state.last_action_time > COOLDOWN:
                try:
                    requests.get(f"{NGROK_URL}/move_forward", timeout=1)
                except:
                    pass
                    
        return annotated_img

webrtc_streamer(key="navigation", video_transformer_factory=VideoProcessor)
