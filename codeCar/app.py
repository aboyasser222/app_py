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
    import streamlit as st
import numpy as np
from ultralytics import YOLO
from PIL import Image
import requests

# --- Settings ---
# اتأكد إن الـ URL ده هو اللي طالع لك من الـ ngrok دلوقتي
NGROK_URL = "https://autistic-revenge-unending.ngrok-free.dev" 

st.set_page_config(page_title="Water Hyacinth Detector", layout="wide")
st.title("🌿 Water Hyacinth Detection System")

# Load Model
@st.cache_resource
def load_model():
    # استعمل اسم ملف Version 4 اللي شغال معاك كويس
    return YOLO("best_v4.pt") 

model = load_model()

# Camera Input Component
camera_input = st.camera_input("Take a photo to analyze")

if camera_input is not None:
    # 1. تحويل الصورة
    image = Image.open(camera_input)
    img_array = np.array(image)
    
    # 2. عملية التوقع (Inference) بالكود القديم البسيط
    results = model(img_array, conf=0.25)
    
    # 3. عرض النتائج
    st.image(results.plot(), caption='Analysis Results', use_column_width=True)
    
    # 4. منطق الحركة
    if len(results.boxes) > 0:
        st.success(f"✅ Water Hyacinth Detected!")
        try:
            # إرسال الأمر للأردوينو
            response = requests.get(f"{NGROK_URL}/move_forward", timeout=5)
            if response.status_code == 200:
                st.toast("🚀 Move command sent successfully!", icon="✅")
        except Exception as e:
            st.error("❌ Connection failed: Check ngrok and Flask server")
    else:
        st.warning("⚠️ No targets detected")
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
