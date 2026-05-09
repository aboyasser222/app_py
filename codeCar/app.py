import streamlit as st
import numpy as np
from ultralytics import YOLO
from PIL import Image
import requests
import cv2

# --- إعدادات ngrok ---
NGROK_URL = "https://autistic-revenge-unending.ngrok-free.dev" 

st.set_page_config(page_title="Water Hyacinth Detector", layout="wide")
st.title("🌿 Water Hyacinth Detection & ngrok Control System")

@st.cache_resource
def load_model():
    return YOLO("best (2).pt") 

model = load_model()

# --- خيارات التشغيل ---
app_mode = st.sidebar.selectbox("Choose Mode", ["Manual (Take Photo)", "Auto-Detect (Experimental)"])

if app_mode == "Manual (Take Photo)":
    # الجزء القديم كما هو بدون تغيير
    camera_input = st.camera_input("Take a photo to analyze")

    if camera_input is not None:
        image = Image.open(camera_input)
        img_array = np.array(image)
        
        # تحسين الدقة
        results = model(img_array, conf=0.40, iou=0.45, imgsz=640, augment=True)[0]
        
        st.image(results.plot(), caption='Analysis Results', use_column_width=True)
        
        if len(results.boxes) > 0:
            st.success(f"✅ Water Hyacinth Detected!")
            try:
                requests.get(f"{NGROK_URL}/move_forward", timeout=5)
                st.info("🚀 Move command (F) sent successfully")
            except:
                st.error("❌ Connection failed")

elif app_mode == "Auto-Detect (Experimental)":
    st.warning("Note: Continuous detection works best when running the app locally.")
    # ملاحظة: Streamlit Cloud بيواجه قيود في الـ Live Video
    # هذا الجزء سيحتاج مكتبة streamlit-webrtc لو أردت فيديو حقيقي مستمر
    st.write("To enable real-time detection without clicking, please use the Local Version or 'streamlit-webrtc'.")
