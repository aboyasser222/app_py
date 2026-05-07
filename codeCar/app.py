import streamlit as st
import numpy as np
from ultralytics import YOLO
from PIL import Image
import requests

# --- الإعدادات ---
# استبدل هذا الرابط بالرابط الذي يظهر لك في شاشة ngrok
NGROK_URL = "https://autistic-revenge-unending.ngrok-free.dev" 

st.set_page_config(page_title="Water Hyacinth Detector", layout="wide")
st.title("🌿 نظام كشف ورد النيل والتحكم عبر ngrok")

# تحميل الموديل
@st.cache_resource
def load_model():
    return YOLO("codeCar/water_hyacinth.pt") 

model = load_model()

uploaded_file = st.file_uploader("ارفع صورة ورد النيل للتحليل", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    
    # التحليل
    results = model(img_array, conf=0.3)[0]
    max_conf = 0
    
    for box in results.boxes:
        conf = float(box.conf)
        if conf > max_conf:
            max_conf = conf

    # عرض النتيجة
    st.image(results.plot(), caption='نتائج التحليل', use_column_width=True)
    
    if max_conf > 0.80:
        st.success(f"✅ تم اكتشاف ورد نيل بدقة ({max_conf*100:.0f}%)")
        try:
            # إرسال الأمر للابتوب عبر نفق ngrok
            response = requests.get(f"{NGROK_URL}/move_forward", timeout=5)
            if response.status_code == 200:
                st.info("🚀 تم إرسال أمر التحرك (F) بنجاح عبر ngrok")
        except Exception as e:
            st.error(f"❌ فشل الاتصال باللابتوب: تأكد أن ngrok وسيرفر Flask يعملان")
    else:
        st.warning(f"⚠️ الدقة منخفضة ({max_conf*100:.0f}%) - لن يتم تحريك الكار")
