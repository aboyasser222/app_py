import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import requests

# تأكد أن السيرفر بتاع ngrok شغال على اللابتوب
NGROK_URL = "https://autistic-revenge-unending.ngrok-free.dev" 

st.set_page_config(page_title="Local Hyacinth Detector", layout="wide")
st.title("🌿 Real-Time Detection (Local Mode)")

@st.cache_resource
def load_model():
    return YOLO("best (3).pt") 

model = load_model()
frame_window = st.image([])

# حاول تغيير الرقم لـ 1 أو 2 لو عندك كاميرا خارجية
cap = cv2.VideoCapture(0) 

if not cap.isOpened():
    st.error("Cannot open camera. If you are on Streamlit Cloud, this won't work. Run it locally!")

stop_button = st.button("Stop System")

while cap.isOpened() and not stop_button:
    ret, frame = cap.read()
    if not ret:
        break

    # معالجة الفريم
    results = model(frame, conf=0.5, imgsz=320, verbose=False)[0]
    annotated_frame = results.plot()
    
    display_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
    frame_window.image(display_frame)

    if len(results.boxes) > 0:
        try:
            requests.get(f"{NGROK_URL}/move_forward", timeout=0.5)
            st.toast("Object Detected! Signal Sent.", icon="🌱")
        except:
            pass

cap.release()
