import streamlit as st
import os
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import paho.mqtt.client as mqtt

# إعدادات MQTT
# 1. عدل الإعدادات في بداية الملف
MQTT_BROKER = "mqtt.eclipseprojects.io"
MQTT_PORT = 80 # استخدم بورت 80 أو 443 للهروب من الـ Firewall
MQTT_TOPIC = "water_hyacinth_robot_yasser"

def send_mqtt_command(cmd):
    try:
        # الإرسال باستخدام Websockets
        import paho.mqtt.publish as publish
        publish.single(
            MQTT_TOPIC, 
            payload=cmd, 
            hostname=MQTT_BROKER, 
            port=MQTT_PORT,
            transport="websockets" # مهم جداً
        )
        return True
    except Exception as e:
        st.error(f"❌ فشل الإرسال: {e}")
        return False
# --- واجهة Streamlit ---
st.set_page_config(page_title="Water Hyacinth Detector", layout="wide")
st.title("🌿 نظام كشف ورد النيل والتحكم عن بُعد")

# تحميل الموديل
@st.cache_resource
def load_model():
    return YOLO("codeCar/water_hyacinth.pt") 

model = load_model()

uploaded_file = st.file_uploader("ارفع صورة ورد النيل هنا للتحليل", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # معالجة الصورة
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    
    # التحليل (Inference)
    results = model(img_array, conf=0.3)[0]
    
    max_conf = 0
    
    # الحصول على أعلى دقة كشف
    for box in results.boxes:
        conf = float(box.conf)
        if conf > max_conf:
            max_conf = conf

    # عرض صورة التحليل
    st.image(results.plot(), caption='نتائج تحليل الموديل', use_column_width=True)
    
    # اتخاذ القرار بناءً على الدقة
    if max_conf > 0.80:
        st.success(f"✅ تم اكتشاف ورد نيل بدقة ({max_conf*100:.0f}%)")
        if send_mqtt_command("F"):
            st.info("🚀 تم إرسال أمر التحرك (F) للكار عبر السحاب")
    else:
        st.warning(f"⚠️ لم يتم اكتشاف ورد نيل بدقة كافية ({max_conf*100:.0f}%)")
        send_mqtt_command("S")
        st.info("🛑 تم إرسال أمر التوقف (S)")
