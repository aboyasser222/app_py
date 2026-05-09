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
# --- الوضع الثاني: Live (الرصد التلقائي المعدل) ---
elif app_mode == "Live (Auto-Detect)":
    st.info("الرصد المباشر يعمل.. تأكد من وجود إضاءة جيدة.")
    
    class VideoProcessor(VideoProcessorBase):
        def __init__(self):
            # تحميل الموديل جوه الـ Processor عشان يفضل جاهز
            self.model = model 

        def recv(self, frame):
            # 1. تحويل الفريم لـ Array
            img = frame.to_ndarray(format="bgr24")
            
            # 2. تصغير الصورة جداً للسرعة (هيرفع الـ FPS ويخلي الـ Detect يظهر)
            # قللنا الـ conf لـ 0.30 عشان يلقط أسرع في الفيديو
            results = self.model(img, conf=0.30, imgsz=256, verbose=False)[0]
            
            # 3. رسم المربعات
            annotated_frame = results.plot()
            
            # 4. إرسال الإشارة لـ ngrok لو فيه اكتشاف
            if len(results.boxes) > 0:
                try:
                    # استخدمنا timeout صغير جداً عشان الفيديو ما يوقفش
                    requests.get(f"{NGROK_URL}/move_forward", timeout=0.01)
                except:
                    pass
            
            # 5. إرجاع الصورة المرسومة للمتصفح
            return frame.from_ndarray(annotated_frame, format="bgr24")

    webrtc_streamer(
        key="hyacinth-live",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=VideoProcessor, # تأكد إن الاسم مطابق هنا
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True
    )
