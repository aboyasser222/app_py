import streamlit as st
import numpy as np
from ultralytics import YOLO
from PIL import Image
import requests
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration, WebRtcMode

# 1. إعدادات ngrok والـ STUN Servers
NGROK_URL = "https://autistic-revenge-unending.ngrok-free.dev" 
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302", "stun:stun1.l.google.com:19302"]}]}
)

st.set_page_config(page_title="Water Hyacinth Detector", layout="wide")
st.title("🌿 Water Hyacinth Detection System")

# 2. تحميل الموديل
@st.cache_resource
def load_model():
    return YOLO("best (3).pt") 

model = load_model()

# 3. اختيار الوضع (Mode Selection)
app_mode = st.sidebar.selectbox("Choose Operation Mode", ["Manual (Take Photo)", "Live (Auto-Detect)"])

# --- الوضع الأول: Manual (الكود القديم بتاعك) ---
if app_mode == "Manual (Take Photo)":
    camera_input = st.camera_input("Take a photo to analyze")
    if camera_input is not None:
        image = Image.open(camera_input)
        img_array = np.array(image)
        results = model(img_array, conf=0.40, imgsz=640, augment=True)[0]
        st.image(results.plot(), caption='Analysis Results', use_column_width=True)
        
        if len(results.boxes) > 0:
            st.success("✅ Detected!")
            try: requests.get(f"{NGROK_URL}/move_forward", timeout=2)
            except: st.error("ngrok connection failed")

# --- الوضع الثاني: Live (الرصد التلقائي اللي طلبته) ---
elif app_mode == "Live (Auto-Detect)":
    st.info("الرصد المباشر يعمل الآن.. أول ما يظهر ورد نيل، الروبوت هيتحرك لوحده.")
    
    class VideoProcessor(VideoProcessorBase):
        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            # معالجة الفريم (تقليل imgsz للسرعة في البث المباشر)
            results = model(img, conf=0.40, imgsz=320, verbose=False)[0]
            
            if len(results.boxes) > 0:
                try: requests.get(f"{NGROK_URL}/move_forward", timeout=0.1)
                except: pass
            
            return frame.from_ndarray(results.plot(), format="bgr24")

    webrtc_streamer(
        key="hyacinth-live",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=VideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True
    )
