import gradio as gr
import cv2
from ultralytics import YOLO
from collections import defaultdict
import numpy as np
import time
import os
import tempfile
import torch

# --- Configuration ---
MODEL_PATH = 'yolov8s_thermal_prototype.pt'
STATIONARY_THRESHOLD_SECONDS = 5.0
MOVEMENT_PIXEL_THRESHOLD = 10
FRAME_SKIP = 5 # Optimize for speed

# --- Device Check ---
device = 0 if torch.cuda.is_available() else 'cpu'
print(f"Running on device: {device}")

# --- Debug: Check for Model File ---
if not os.path.exists(MODEL_PATH):
    print(f"\n FATAL ERROR: The model file '{MODEL_PATH}' was not found!")
    print(f"Current Working Directory: {os.getcwd()}")
    print("Files in current directory:", os.listdir('.'))
    print(" PLEASE ACTION: Upload 'yolov8s_thermal_prototype.pt' to the Files tab on the left.\n")
    # We stop execution here so you don't get a confusing error later
    raise FileNotFoundError(f"Model file '{MODEL_PATH}' not found. Please upload it.")

# --- Load Model ---
try:
    model = YOLO(MODEL_PATH)
    model.to(device) # Move model to GPU
    print("Thermal model loaded successfully on GPU.")
except Exception as e:
    print(f" Error loading model: {e}")
    model = None 

def process_thermal_video(video_path):
    print(f"Processing video: {video_path}")

    if model is None:
        raise gr.Error("Model failed to load. Check the logs above for 'FATAL ERROR'.")

    if video_path is None:
        return None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise gr.Error("Could not open video file.")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0: fps = 30

    # Temp file for output
    output_video_file = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
    output_path = output_video_file.name
    output_video_file.close()

    # Setup VideoWriter
    output_fps = fps / FRAME_SKIP
    # In Colab, 'mp4v' or 'avc1' codecs are usually safe
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(output_path, fourcc, output_fps, (width, height))

    track_history = defaultdict(lambda: {'positions': [], 'timestamps': [], 'is_stationary': False})
    stationary_check_frames = max(1, int(STATIONARY_THRESHOLD_SECONDS * output_fps))

    frame_count = 0
    
    while True:
        success, frame = cap.read()
        if not success:
            break
        
        if frame_count % FRAME_SKIP != 0:
            frame_count += 1
            continue

        video_time = frame_count / fps 
        frame_count += 1

        # Run tracking on GPU
        # verbose=False keeps the console clean
        results = model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False, device=device)
        
        annotated_frame = frame.copy()

        if results[0].boxes is not None and results[0].boxes.id is not None:
            # Move results to CPU for drawing
            boxes = results[0].boxes.xywh.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().tolist()

            annotated_frame = results[0].plot(img=annotated_frame, line_width=2, labels=False, conf=False)

            for box, track_id in zip(boxes, track_ids):
                center_x, center_y, w, h = box
                current_position = (center_x, center_y)

                history = track_history[track_id]
                history['positions'].append(current_position)
                history['timestamps'].append(video_time)

                while len(history['timestamps']) > 0 and video_time - history['timestamps'][0] > STATIONARY_THRESHOLD_SECONDS:
                    history['positions'].pop(0)
                    history['timestamps'].pop(0)

                if len(history['positions']) >= stationary_check_frames:
                    start_pos = np.array(history['positions'][0])
                    current_pos = np.array(current_position)
                    distance_moved = np.linalg.norm(current_pos - start_pos)

                    if distance_moved < MOVEMENT_PIXEL_THRESHOLD:
                        if not history['is_stationary']:
                             history['is_stationary'] = True
                    else:
                        history['is_stationary'] = False

                if history['is_stationary']:
                    x1, y1 = int(center_x - w / 2), int(center_y - h / 2)
                    x2, y2 = int(center_x + w / 2), int(center_y + h / 2)
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    cv2.putText(annotated_frame, f"ID:{track_id} STATIONARY", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        out.write(annotated_frame)

    cap.release()
    out.release()
    
    return output_path

# --- Launch App ---
iface = gr.Interface(
    fn=process_thermal_video,
    inputs=gr.Video(label="Upload Thermal Video", format="mp4"),
    outputs=gr.Video(label="Processed Output"),
    title="Thermal Chick Monitor",
    description="Upload a thermal video. Detects and flags stationary warm objects.",
    allow_flagging="never"
)

# share=True creates a public link
iface.launch(share=True, debug=True)