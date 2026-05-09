import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import requests

NGROK_URL = "https://autistic-revenge-unending.ngrok-free.dev" 

st.set_page_config(page_title="Live Hyacinth Detector", layout="wide")
st.title("🌿 Real-Time Water Hyacinth Detection")

@st.cache_resource
def load_model():
    # تأكد من رفع ملف الـ best.pt الجديد اللي هيطلع من Kaggle هنا
    return YOLO("best (3).pt") 

model = load_model()

# إعداد مكان عرض الفيديو
frame_window = st.image([])

# فتح الكاميرا
cap = cv2.VideoCapture(0)

stop_button = st.button("Stop System")

while cap.isOpened() and not stop_button:
    ret, frame = cap.read()
    if not ret:
        st.error("Failed to access camera")
        break

    # التحليل باستخدام YOLO
    # imgsz=320 لتسريع المعالجة في البث المباشر
    results = model(frame, conf=0.5, imgsz=320)[0]
    
    # رسم النتائج على الفريم
    annotated_frame = results.plot()
    
    # تحويل اللون من BGR لـ RGB للعرض في Streamlit
    display_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
    frame_window.image(display_frame)

    # التحقق من وجود اكتشافات
    if len(results.boxes) > 0:
        try:
            # إرسال الأمر بمجرد الرصد
            response = requests.get(f"{NGROK_URL}/move_forward", timeout=1)
            if response.status_code == 200:
                st.toast("🚀 Target Detected: Move Command Sent!", icon="✅")
        except:
            pass # تجنب توقف البث في حالة فشل الاتصال المؤقت

cap.release()
