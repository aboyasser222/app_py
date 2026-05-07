from flask import Flask, render_template, request, send_from_directory
import os
import logging
import time
import atexit
import cv2
import serial
from ultralytics import YOLO
from werkzeug.utils import secure_filename

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__)

SERIAL_PORT = "COM3"
BAUD_RATE = 9600
TIMEOUT = 1
arduino = None

UPLOAD_FOLDER = "static/uploads"
RESULTS_FOLDER = "static/results"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["RESULTS_FOLDER"] = RESULTS_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

CONFIDENCE_THRESHOLD = 0.80

model = YOLO("best (1).pt")
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def initialize_serial():
    global arduino
    try:
        arduino = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=TIMEOUT)
        time.sleep(2)
        arduino.reset_input_buffer()
        arduino.reset_output_buffer()
        logger.info(f"Successfully connected to serial port: {SERIAL_PORT}")
    except serial.SerialException as e:
        logger.error(f"Failed to open serial port: {e}")
        arduino = None


def close_serial():
    global arduino
    if arduino and arduino.is_open:
        try:
            arduino.close()
            logger.info("Serial port closed successfully.")
        except Exception as e:
            logger.error(f"Error while closing serial port: {e}")


def send_command(cmd: bytes):
    if arduino and arduino.is_open:
        try:
            arduino.reset_input_buffer()
            arduino.write(cmd)
            time.sleep(0.15)
            logger.info(f"Command sent: {cmd.decode()}")
            return True
        except serial.SerialException as e:
            logger.error(f"Failed to send command: {e}")
            return False
    logger.warning("Serial port is not connected or is closed.")
    return False


initialize_serial()
atexit.register(close_serial)


@app.route("/", methods=["GET", "POST"])
def upload_and_detect():
    original_filename = None
    result_filename = None
    message = ""

    if request.method == "POST":
        if "file" not in request.files:
            message = "No file was uploaded."
        else:
            file = request.files["file"]
            if file.filename == "":
                message = "Please select an image to upload."
            elif file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                original_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
                file.save(original_path)

                results = model(original_path, conf=0.3)[0]
                img = cv2.imread(original_path)
                confidences = []

                for box in results.boxes:
                    if int(box.cls) == 0:
                        conf = float(box.conf)
                        confidences.append(conf)
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        label = f"{conf:.0%}"
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
                        text_y = max(20, y1 - 10)
                        cv2.putText(
                            img,
                            label,
                            (x1, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.9,
                            (0, 0, 255),
                            3,
                        )

                result_filename = f"result_{filename}"
                result_path = os.path.join(
                    app.config["RESULTS_FOLDER"], result_filename
                )
                cv2.imwrite(result_path, img)
                original_filename = filename

                if confidences:
                    max_conf = max(confidences)
                    print(f"Max confidence: {max_conf:.2%}")
                    if max_conf >= CONFIDENCE_THRESHOLD:
                        send_command(b"F")
                        message = f"Water hyacinth detected ({max_conf:.0%})"
                    else:
                        send_command(b"S")
                        message = f"Low confidence ({max_conf:.0%} < 80%) - Ignored."
                else:
                    send_command(b"S")
                    message = "No water hyacinth detected - Robot stopped."

    return render_template(
        "index.html",
        original=original_filename,
        result=result_filename,
        message=message,
    )


@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


@app.route("/results/<filename>")
def result_file(filename):
    return send_from_directory(app.config["RESULTS_FOLDER"], filename)


if __name__ == "__main__":
    app.run(debug=False, use_reloader=False)
