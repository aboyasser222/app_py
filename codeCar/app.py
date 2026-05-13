import streamlit as st
import numpy as np
from ultralytics import YOLO
from PIL import Image
import requests

NGROK_URL = "https://autistic-revenge-unending.ngrok-free.dev"

st.set_page_config(page_title="Autonomous Navigation System", layout="wide")
st.title("🛥️ Detection System: Boat, Rock & Water Hyacinth")

@st.cache_resource
def load_model():
    model = YOLO("codeCar/best (7).pt") 
    # إرجاع تسمية الكلاسات هنا
    model.names[0] = "boat"
    model.names[1] = "rock"
    model.names[2] = "water_hyacinth"
    return model

model = load_model()

camera_input = st.camera_input("Capture image for analysis")

if camera_input is not None:
    image = Image.open(camera_input)
    img_array = np.array(image)
    
    # استخدام conf=0.40 لضمان ظهور المربعات
    results = model(img_array, conf=0.40)[0]
    
    # إرجاع تسمية الكلاسات في النتائج قبل الرسم
    results.names[0] = "boat"
    results.names[1] = "rock"
    results.names[2] = "water_hyacinth"
    
    # استخدام channels="BGR" لضبط ألوان الصورة
    st.image(results.plot(), caption='Analysis Results', use_column_width=True, channels="BGR")
    
    detections = results.boxes.data.tolist()
    
    boat_detected = any(int(box[5]) == 0 for box in detections)
    rock_detected = any(int(box[5]) == 1 for box in detections)
    hyacinth_detected = any(int(box[5]) == 2 for box in detections)
    
    # أولوية الحركة: تجنب العقبات (الصخور والمراكب) قبل التحرك نحو الهدف
    if rock_detected:
        st.error("🪨 Rock Detected! Danger! Reversing...")
        try:
            response = requests.get(f"{NGROK_URL}/move_backward", timeout=5)
            if response.status_code == 200:
                st.info("Action: Move Backward command sent")
        except:
            st.error("Connection Error: Check ngrok and Flask server")
            
    elif boat_detected:
        st.error("🚨 Boat Detected! Reversing and changing course...")
        try:
            response = requests.get(f"{NGROK_URL}/move_backward", timeout=5)
            if response.status_code == 200:
                st.info("Action: Move Backward command sent")
        except:
            st.error("Connection Error: Check ngrok and Flask server")
            
    elif hyacinth_detected:
        st.success("🌿 Water Hyacinth Detected! Moving forward...")
        try:
            response = requests.get(f"{NGROK_URL}/move_forward", timeout=5)
            if response.status_code == 200:
                st.info("Action: Move Forward command sent")
        except:
            st.error("Connection Error: Check ngrok and Flask server")
            
    else:
        st.warning("⚠️ Clear path - No targets detected")
