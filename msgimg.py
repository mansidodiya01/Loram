import cv2
import time
import serial
import os
from ultralytics import YOLO
from mediapipe.tasks import python
from mediapipe.tasks.python.audio.core import audio_record
from mediapipe.tasks.python.components import containers
from mediapipe.tasks.python import audio
import Adafruit_DHT

# Set up LoRa communication
lora_serial = serial.Serial('/dev/ttyUSB0', 115200)  # Adjust the port to your setup

# Directories for saving images
image_dir = "images"
cropped_dir = os.path.join(image_dir, "cropped")
original_dir = os.path.join(image_dir, "original")
os.makedirs(cropped_dir, exist_ok=True)
os.makedirs(original_dir, exist_ok=True)

# Shared variables
audio_result_message = "Audio classification not processed."

# Function to send messages via LoRa
def send_via_lora(message):
    if lora_serial.is_open:
        lora_serial.write(f"{message}\n".encode("utf-8"))

# Function to send images via LoRa
def send_image_via_lora(image_path):
    if lora_serial.is_open:
        with open(image_path, "rb") as img_file:
            while chunk := img_file.read(128):  # Send in chunks of 128 bytes
                lora_serial.write(chunk)
            lora_serial.write(b"\nEND\n")  # Mark the end of the image
    print(f"Image {image_path} sent via LoRa.")

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

# Audio classification callback
def audio_result_callback(result: audio.AudioClassifierResult, timestamp_ms: int):
    global audio_result_message
    detected_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
    print(f"Callback triggered at {detected_time} with result: {result}")

    detected = False
    for category in result.classifications[0].categories:
        print(f"Category: {category.category_name}, Score: {category.score}")
        if ("baby cry" in category.category_name.lower() or
            "infant cry" in category.category_name.lower()) and category.score > 0.3:
            audio_result_message = "Baby is crying."
            detected = True
            break

    if not detected:
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
            if score > best_score:
                best_box = box
                best_score = score

    detected_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
    message = f"{detected_time} - "

    if best_box is not None:
        # Crop and save the image
        x_min, y_min, x_max, y_max = map(int, best_box)
        cropped_image = frame[y_min:y_max, x_min:x_max]
        cropped_path = os.path.join(cropped_dir, f"cropped_{int(time.time())}.jpg")
        cv2.imwrite(cropped_path, cropped_image)
        send_image_via_lora(cropped_path)  # Send the cropped image via LoRa
        warnings = check_environment_conditions()
        return f"Baby is detected.{warnings}"
    else:
        # Save the original image
        original_path = os.path.join(original_dir, f"original_{int(time.time())}.jpg")
        send_image_via_lora(original_path)  # Send the original image via LoRa
        cv2.imwrite(original_path, frame)
        return "Baby is not there."

def main():
    # Initialize YOLO model
    model_path = "best_ncnn_model"
    ncnn_model = YOLO(model_path, task="detect")

    # Initialize the audio classification model
    audio_model_path = "yamnet.tflite"
    base_options = python.BaseOptions(model_asset_path=audio_model_path)
    options = audio.AudioClassifierOptions(
        base_options=base_options,
        running_mode=audio.RunningMode.AUDIO_STREAM,
        max_results=5,
        score_threshold=0.3,
        result_callback=audio_result_callback,
    )
    audio_classifier = audio.AudioClassifier.create_from_options(options)

    # Initialize audio recorder
    buffer_size, sample_rate, num_channels = 15600, 16000, 1
    record = audio_record.AudioRecord(num_channels, sample_rate, buffer_size)

    if not record:
        print("Error: Audio recorder initialization failed.")
        return

    # Open the camera stream
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Unable to access the camera.")
        return

    # Start audio recording
    record.start_recording()

    print("Starting synchronized detection. Press 'q' to exit.")

    while True:
        start_time = time.time()

        # Process video frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to read frame from the camera.")
            break

        frame_result = process_frame(ncnn_model, frame)

        # Process audio input
        data = record.read(buffer_size)
        audio_data = containers.AudioData(buffer_size, containers.AudioDataFormat(1, 16000))
        audio_data.load_from_array(data)
        audio_classifier.classify_async(audio_data, time.time_ns() // 1_000_000)

        # Combine results
        combined_message = f"Audio: {audio_result_message} | Frame: {frame_result}"
        send_via_lora(combined_message)
        print(combined_message)

        # Synchronize processing rate
        elapsed_time = time.time() - start_time
        time.sleep(max(0, buffer_size / sample_rate - elapsed_time))

        # Exit condition
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            print("Exiting...")
            break

    # Release resources
    cap.release()
    record.stop_recording()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
