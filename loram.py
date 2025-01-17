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

# Function to send messages via LoRa
def send_via_lora(message):
    if lora_serial.is_open:
        lora_serial.write(f"{message}\n".encode('utf-8'))

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

# Define a result callback function for audio classification
def audio_result_callback(result: audio.AudioClassifierResult, timestamp_ms: int):
    detected_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
    message = f"Timestamp: {detected_time} - "

    detected = False
    for category in result.classifications[0].categories:
        if ("baby cry" in category.category_name.lower() or
            "infant cry" in category.category_name.lower()):
            message += "Your baby is crying."
            detected = True
            break

    if not detected:
        message += "Baby is not crying."

    send_via_lora(message)
    print(message)

def main():
    # Path to the NCNN model and audio classification model
    model_path = "best_ncnn_model"
    audio_model_path = "yamnet.tflite"

    # Directories for saving images
    base_dir = "images"
    original_dir = os.path.join(base_dir, "original")
    cropped_dir = os.path.join(base_dir, "cropped")
    detected_dir = os.path.join(base_dir, "detected")
    os.makedirs(original_dir, exist_ok=True)
    os.makedirs(cropped_dir, exist_ok=True)
    os.makedirs(detected_dir, exist_ok=True)

    # Initialize the YOLO model
    ncnn_model = YOLO(model_path, task="detect")

    # Initialize the audio classification model
    base_options = python.BaseOptions(model_asset_path=audio_model_path)
    options = audio.AudioClassifierOptions(
        base_options=base_options,
        running_mode=audio.RunningMode.AUDIO_STREAM,
        max_results=5,
        score_threshold=0.3,
        result_callback=audio_result_callback,  # Pass the callback function here
    )
    audio_classifier = audio.AudioClassifier.create_from_options(options)

    # Initialize the audio recorder
    buffer_size, sample_rate, num_channels = 15600, 16000, 1
    audio_format = containers.AudioDataFormat(num_channels, sample_rate)
    record = audio_record.AudioRecord(num_channels, sample_rate, buffer_size)
    audio_data = containers.AudioData(buffer_size, audio_format)

    # Open the camera stream
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        error_message = "Error: Unable to access the camera."
        print(error_message)
        send_via_lora(error_message)
        return

    # Synchronization interval in seconds
    interval_between_inference = 0.5

    # Start audio recording
    record.start_recording()

    print("Starting baby detection and cry detection. Press 'q' to exit.")

    while True:
        start_time = time.time()

        # Generate a timestamp for all operations
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
        message = f"Timestamp: {timestamp} - "

        # Capture a frame from the camera
        ret, frame = cap.read()
        if not ret:
            error_message = "Error: Unable to read frame from the camera."
            print(error_message)
            send_via_lora(error_message)
            break

        # Run inference on the captured frame
        results = ncnn_model(frame)

        # Save the original frame resized to 224x224
        resized_original = cv2.resize(frame, (224, 224))
        original_image_path = os.path.join(original_dir, f"original_{timestamp.replace(':', '_')}.jpg")
        cv2.imwrite(original_image_path, resized_original)

        baby_detected = False
        best_box = None
        best_score = 0

        # Find the bounding box with the highest confidence score
        for result in results:
            boxes = result.boxes.xyxy.numpy()
            scores = result.boxes.conf.numpy()

            for i, box in enumerate(boxes):
                score = scores[i]
                if score > best_score:
                    best_box = box
                    best_score = score
                    baby_detected = True

        # Generate message based on detection
        if baby_detected:
            message += "Baby is detected."
            warnings = check_environment_conditions()
            if warnings:
                message += warnings
        else:
            message += "Baby is not there."

        send_via_lora(message)
        print(message)

        # Update the reference_time and classify audio
        reference_time = time.monotonic_ns() // 1_000_000  # Monotonic timestamp in milliseconds
        data = record.read(buffer_size)
        audio_data.load_from_array(data)
        audio_classifier.classify_async(audio_data, reference_time)

        elapsed_time = time.time() - start_time
        sleep_time = max(0, interval_between_inference - elapsed_time)
        time.sleep(sleep_time)

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
