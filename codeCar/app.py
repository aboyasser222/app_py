import streamlit as st
import numpy as np
from ultralytics import YOLO
from PIL import Image
import requests

NGROK_URL = "https://autistic-revenge-unending.ngrok-free.dev" 

# 1. تعديل عنوان الصفحة
st.set_page_config(page_title="Boat Detector", layout="wide")
st.title("🛥️ Boat Detection & ngrok Control System")

@st.cache_resource
def load_model():
    # 1. بنعرف الموديل الأول في متغير اسمه model
    model = YOLO("best (5).pt") 
    
    # 2. بنغير اسم الكلاس رقم 0 لـ boat قبل ما نخرج من الدالة
    model.names[0] = "boat" 
    
    # 3. بنرجع الموديل وهو "متعدل" جاهز
    return model

model = load_model()

camera_input = st.camera_input("Take a photo to analyze")

if camera_input is not None:
    image = Image.open(camera_input)
    img_array = np.array(image)
    
    # تشغيل الموديل
    results = model(img_array, conf=0.77)[0]
    
    # 💡 السطر ده "إجباري" لتغيير الاسم قبل الرسم
    results.names[0] = "boat" 
    
    # عرض النتائج بالاسم الجديد
    st.image(results.plot(), caption='Analysis Results', use_column_width=True)
    
    # باقي الكود...
    detections = results.boxes.data.tolist()
    
    if len(detections) > 0:
        # 3. تعديل رسالة النجاح
        st.success(f"✅ Boat Detected!") 
        try:
            response = requests.get(f"{NGROK_URL}/move_forward", timeout=5)
            if response.status_code == 200:
                st.info("🚀 Move command (F) sent successfully via ngrok")
        except Exception as e:
            st.error(f"❌ Connection failed: Ensure ngrok and Flask server are running")
    else:
        # 4. تعديل رسالة التحذير
        st.warning(f"⚠️ No boat detected - Robot will not move")
