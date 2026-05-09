import streamlit as st
import numpy as np
from ultralytics import YOLO
from PIL import Image
import requests

# 1. إعدادات ngrok والـ Model
NGROK_URL = "https://autistic-revenge-unending.ngrok-free.dev" 

@st.cache_resource
def load_model():
    # تأكد إنك سميت ملف Version 4 باسم "best_v4.pt" وحطيته في الفولدر
    return YOLO("best (2).pt") 

model = load_model()

st.title("🌿 Water Hyacinth Detection (V4 Optimized)")

# 2. إدخال الصورة
camera_input = st.camera_input("Take a photo to analyze")

if camera_input is not None:
    image = Image.open(camera_input)
    img_array = np.array(image)
    
    # --- السطر اللي سألت عليه هنا ---
    # رفعنا الـ conf لـ 0.35 عشان نقلل الغلط
    # فعلنا augment=True عشان الموديل "يدقق" أكتر في تفاصيل الورق
    results = model(img_array, 
                conf=0.25,       # نزلنا الـ conf شوية عشان نضمن إنه يلقط في البداية
                iou=0.45, 
                imgsz=640, 
                augment=False    # اقفل الـ augment حالياً للتأكد من السرعة
               )[0]

# وعشان تتأكد الموديل شايف إيه، اطبع الأسماء في الـ Console عندك:
    print(model.names)
    # -------------------------------
    
    # عرض النتيجة
    st.image(results.plot(), caption='V4 Analysis', use_column_width=True)
    
    # منطق الحركة
    if len(results.boxes) > 0:
        st.success("✅ Water Hyacinth Found!")
        try:
            requests.get(f"{NGROK_URL}/move_forward", timeout=2)
            st.toast("Command Sent: Move Forward")
        except:
            st.error("Connection to ngrok failed")
    else:
        st.warning("⚠️ Area Clear")
