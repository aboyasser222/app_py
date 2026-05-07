import streamlit as st
import os
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import paho.mqtt.client as mqtt

# --- إعدادات MQTT (الجسر اللاسلكي) ---
MQTT_BROKER = "broker.hivemq.com" # نفس الوسيط اللي فتحته في الصورة
MQTT_TOPIC = "water_hyacinth_robot_yasser"

def send_mqtt_command(cmd):
    try:
        client = mqtt.Client()
        client.connect(MQTT_BROKER, 1883, 60)
        client.publish(MQTT_TOPIC, cmd)
        client.disconnect()
        return True
    except Exception as e:
        st.error(f"Error sending MQTT: {e}")
        return False

# --- واجهة Streamlit ---
st.set_page_config(page_title="Water Hyacinth Detector", layout="wide")
st.title("🌿 نظام كشف ورد النيل والتحكم عن بُعد")

# تحميل الموديل (تأكد من وجود الملف في GitHub)
@st.cache_resource
def load_model():
    # استخدم اسم الملف اللي عندك في الكود (water_hyacinth.pt)
    return YOLO("codeCar/water_hyacinth.pt") 

model = load_model()

uploaded_file = st.file_uploader("codeCar/static/results/result_0.jpeg.", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # معالجة الصورة
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    
    # التحليل (Inference)
    results = model(img_array, conf=0.3)[0]
    
    found = False
    max_conf = 0
    
    # رسم النتائج
    for box in results.boxes:
        conf = float(box.conf)
        if conf > max_conf:
            max_conf = conf
        
        # إذا تجاوزت الدقة 80% (حسب طلبك في الكود الأصلي)
        if conf >= 0.80:
            found = True
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(img_array, (x1, y1), (x2, y2), (255, 0, 0), 5)

    # عرض النتيجة
    st.image(img_array, caption='تحليل الروبوت', use_column_width=True)
    
    if found:
        st.success(f"✅ تم اكتشاف ورد نيل بدقة ({max_conf:.0%})")
        send_mqtt_command("F") # إرسال أمر Move Forward للابتوب
        st.info("🚀 تم إرسال أمر التحرك للكار عبر السحاب")
    else:
        st.warning(f"⚠️ لم يتم الكشف بدقة كافية ({max_conf:.0%})")
        send_mqtt_command("S") # إرسال أمر Stop
