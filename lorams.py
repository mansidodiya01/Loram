import cv2
import os
import time
from ultralytics import YOLO
from mediapipe.tasks import python
from mediapipe.tasks.python.audio.core import audio_record
from mediapipe.tasks.python.components import containers
from mediapipe.tasks.python import audio
import Adafruit_DHT
import serial

# Set up LoRa communication
lora_serial = serial.Serial('/dev/ttyUSB0', 115200)  # Adjust the port to your setup

# Shared variable for audio results
audio_result_message = "Baby is not crying."

# Function to send messages via LoRa
def send_via_lora(message):
    if lora_serial.is_open:
        lora_serial.write(f"{message}\n".encode("utf-8"))

# Function to check temperature and humidity
def check_environment_conditions():
    sensor = Adafruit_DHT.DHT11
    pin = 21
    humidity, temperature = Adafruit_DHT.read_retry(sensor, pin)
    warnings = ""

    if humidity is not None and temperature is not None:
        if not (20.0 <= temperature <= 25.0):
            warnings += " Warning: Temperature is outside normal range!"
        if not (30.0 <= humidity <= 60.0):
            warnings += " Warning: Humidity is outside normal range!"
    else:
        warnings += " Failed to get temperature and humidity readings."

    return warnings

# Audio result callback function
def audio_result_callback(result: audio.AudioClassifierResult, timestamp_ms: int):
    global audio_result_message
    for category in result.classifications[0].categories:
        if ("baby cry" in category.category_name.lower() or
            "infant cry" in category.category_name.lower()):
            audio_result_message = "Baby is crying."
            break
    else:
        audio_result_message = "Baby is not crying."

def process_frame(ncnn_model, frame):
    results = ncnn_model(frame)
    best_box = None
    best_score = 0

    for result in results:
        boxes = result.boxes.xyxy.numpy()
        scores = result.boxes.conf.numpy()

        for i, box in enumerate(boxes):
            score = scores[i]
            if score > best_score:  # Keep the box with the highest confidence score
                best_box = box
                best_score = score

    detected_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
    message = f"{detected_time} - "

    if best_box is not None:
        warnings = check_environment_conditions()
        return f"Baby is detected with confidence {best_score:.2f}.{warnings}"
    else:
        return "Baby is not there."

# Main function
def main():
    # Path to the NCNN model and audio classification model
    model_path = "best_ncnn_model"
    audio_model_path = "yamnet.tflite"

    # Initialize the YOLO model
    ncnn_model = YOLO(model_path, task="detect")

    # Initialize the audio classification model
    base_options = python.BaseOptions(model_asset_path=audio_model_path)
    options = audio.AudioClassifierOptions(
        base_options=base_options,
        running_mode=audio.RunningMode.AUDIO_STREAM,
        max_results=5,
        score_threshold=0.3,
        result_callback=audio_result_callback  # Attach the callback function
    )
    audio_classifier = audio.AudioClassifier.create_from_options(options)

    # Initialize the audio recorder
    buffer_size, sample_rate, num_channels = 15600, 16000, 1
    record = audio_record.AudioRecord(num_channels, sample_rate, buffer_size)

    # Open the camera stream
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        error_message = "Error: Unable to access the camera."
        print(error_message)
        send_via_lora(error_message)
        return

    # Start recording audio
    record.start_recording()

    print("Starting synchronized baby detection and cry detection. Press 'q' to exit.")

    while True:
        start_time = time.time()
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
        message = f"{timestamp} - "

        # Capture a frame from the camera
        ret, frame = cap.read()
        if not ret:
            error_message = "Error: Unable to read frame from the camera."
            print(error_message)
            send_via_lora(error_message)
            break

        # Process frame
        frame_result = process_frame(ncnn_model, frame)

        # Combine results
        combined_message = f"Audio: {audio_result_message} | Frame: {frame_result}"
        send_via_lora(combined_message)
        print(combined_message)

        # Target ~2 FPS for synchronized processing
        elapsed_time = time.time() - start_time
        time.sleep(max(0, 0.5 - elapsed_time))

        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            print("Exiting the script...")
            send_via_lora("Exiting the script...")
            break

    # Release resources
    cap.release()
    record.stop_recording()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
