import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
from ultralytics import YOLO
import cv2
import numpy as np
import requests

# رابط ngrok الخاص بك
NGROK_URL = "https://autistic-revenge-unending.ngrok-free.dev" 

@st.cache_resource
def load_model():
    return YOLO("best (2).pt")

model = load_model()

class VideoProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # عمل Detect على الفريم الحالي (استخدام النسخة 4 اللي بتثق فيها)
        results = model(img, conf=0.40, imgsz=320, verbose=False)[0] 
        
        annotated_frame = results.plot()

        # لو لقى ورد نيل، يبعت الأمر للأردوينو
        if len(results.boxes) > 0:
            try:
                # بنستخدم timeout صغير جداً عشان البث ما يقطعش
                requests.get(f"{NGROK_URL}/move_forward", timeout=0.1)
            except:
                pass

        return frame.from_ndarray(annotated_frame, format="bgr24")

st.title("🌿 Live Water Hyacinth Detection")

# ده الجزء اللي هيفتح الكاميرا "فيديو" مش "صورة"
webrtc_streamer(key="example", video_processor_factory=VideoProcessor)

st.write("الكاميرا الآن تعمل بشكل مستمر، سيتم إرسال الأوامر تلقائياً عند اكتشاف ورد النيل.")
