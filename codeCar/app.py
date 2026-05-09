import streamlit as st
import numpy as np
from ultralytics import YOLO
from PIL import Image
import requests

NGROK_URL = "https://autistic-revenge-unending.ngrok-free.dev" 

st.set_page_config(page_title="Water Hyacinth Detector", layout="wide")
st.title("🌿 Water Hyacinth Detection & ngrok Control System")

@st.cache_resource
def load_model():
    return YOLO("best (2).pt") 

model = load_model()

camera_input = st.camera_input("Take a photo to analyze")

if camera_input is not None:
    image = Image.open(camera_input)
    img_array = np.array(image)
    
    results = model(
        img_array, 
        conf=0.40,      # رفع الثقة لتقليل الاكتشافات الوهمية
        iou=0.45,       # دمج المربعات المتداخلة للحصول على أدق نتيجة
        imgsz=640,      # توحيد مقاس الصورة ليتناسب مع تدريب V4
        augment=True    # تفعيل التحيليل المتعدد (TTA) لتدقيق تفاصيل الورق
    )[0]
    
    detections = results.boxes.data.tolist()
    
    st.image(results.plot(), caption='Analysis Results', use_column_width=True)
    
    if len(detections) > 0:
        st.success(f"✅ Water Hyacinth Detected!")
        try:
            response = requests.get(f"{NGROK_URL}/move_forward", timeout=5)
            if response.status_code == 200:
                st.info("🚀 Move command (F) sent successfully via ngrok")
        except Exception as e:
            st.error(f"❌ Connection failed: Ensure ngrok and Flask server are running")
    else:
        st.warning(f"⚠️ No targets detected - Robot will not move")
