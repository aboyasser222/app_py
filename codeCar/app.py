import streamlit as st
import numpy as np
from ultralytics import YOLO
from PIL import Image
import requests

# 1. الإعدادات الأساسية
# تأكد من تحديث الرابط لو فتحت نفق ngrok جديد
NGROK_URL = "https://autistic-revenge-unending.ngrok-free.dev" 

st.set_page_config(page_title="Water Hyacinth Detector", layout="wide")
st.title("🌿 Water Hyacinth Detection System")

# 2. تحميل الموديل (تأكد أن اسم الملف مطابق تماماً لما هو مرفوع على GitHub)
@st.cache_resource
def load_model():
    # استعمل اسم الملف اللي كان شغال معاك (سواء v4 أو غيره)
    return YOLO("best_v4.pt") 

model = load_model()

# 3. مكون الكاميرا
camera_input = st.camera_input("Take a photo to analyze")

if camera_input is not None:
    # تحويل الصورة المعطاة من المتصفح
    image = Image.open(camera_input)
    img_array = np.array(image)
    
    # عملية التوقع البسيطة (Inference)
    results = model(img_array, conf=0.25)[0] 
    
    # عرض الصورة وعليها مربعات الاكتشاف
    st.image(results.plot(), caption='Analysis Results', use_column_width=True)
    
    # 4. منطق إرسال الأوامر للروبوت
    if len(results.boxes) > 0:
        st.success("✅ Water Hyacinth Detected!")
        try:
            # إرسال طلب التحرك للأمام
            response = requests.get(f"{NGROK_URL}/move_forward", timeout=5)
            if response.status_code == 200:
                st.toast("🚀 Signal Sent to Robot", icon="✅")
        except Exception:
            st.error("❌ Connection failed: Check ngrok and local server")
    else:
        st.warning("⚠️ No targets detected")
