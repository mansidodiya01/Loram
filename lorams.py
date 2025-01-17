import cv2
import os
import time
from ultralytics import YOLO
from mediapipe.tasks import python
from mediapipe.tasks.python.audio.core import audio_record
from mediapipe.tasks.python.components import containers
from mediapipe.tasks.python import audio

def save_audio_result(result: audio.AudioClassifierResult, timestamp_ms: int, score_threshold: float):
    detected_time = time.time()  # High-precision timestamp
    detected = False
    for category in result.classifications[0].categories:
        if ("baby cry" in category.category_name.lower() or
            "infant cry" in category.category_name.lower()) and category.score > score_threshold:
            print(f"[{detected_time:.3f}] Baby is crying!")
            detected = True
    if not detected:
        print(f"[{detected_time:.3f}] No baby crying detected.")

def main():
    # Path to the NCNN model and audio classification model
    model_path = "best_ncnn_model"
    audio_model_path = "yamnet.tflite"

    # Initialize directories for saving images
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
        result_callback=lambda result, timestamp_ms: save_audio_result(result, timestamp_ms, 0.3),
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
        print("Error: Unable to access the camera.")
        return

    # Start audio recording
    record.start_recording()
    reference_time = time.monotonic_ns() // 1_000_000  # Current time in milliseconds
    buffer_duration = buffer_size / sample_rate

    print("Starting combined baby and cry detection. Press 'q' to exit.")

    while True:
        start_time = time.time()

        # Capture a frame from the camera
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to read frame from the camera.")
            break

        # Run inference on the captured frame
        results = ncnn_model(frame)

        # Find the bounding box with the highest confidence score
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

        # Crop and save the frame if a bounding box is detected
        if best_box is not None:
            x_min, y_min, x_max, y_max = map(int, best_box)
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(frame.shape[1], x_max)
            y_max = min(frame.shape[0], y_max)

            cropped_image = frame[y_min:y_max, x_min:x_max]
            resized_cropped = cv2.resize(cropped_image, (224, 224))
            cropped_image_path = os.path.join(cropped_dir, f"cropped_{time.time()}.jpg")
            cv2.imwrite(cropped_image_path, resized_cropped)

            print(f"Detected object saved: {cropped_image_path}")

        # Process audio data
        data = record.read(buffer_size)
        audio_data.load_from_array(data)
        audio_timestamp = reference_time
        reference_time += int(buffer_duration * 1000)
        audio_classifier.classify_async(audio_data, audio_timestamp)

        elapsed_time = time.time() - start_time
        time.sleep(max(0, 0.5 - elapsed_time))  # Maintain ~2 FPS

        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            print("Exiting the script...")
            break

    # Release resources
    cap.release()
    record.stop_recording()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
