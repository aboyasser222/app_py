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
                conf=0.40,      # ارفعه لـ 0.40 عشان تضمن إنك مش بتلقط "خيال"
                iou=0.45,       # تقليل التداخل
                imgsz=640,      # توحيد المقاس
                classes=[2],    # التركيز على ورد النيل فقط (تأكد من الرقم)
                augment=True    # التحيليل المتعدد للزوايا
               )[0]
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
