import cv2
import torch
import numpy as np
import time
import os
from torchvision.models.optical_flow import raft_small
from urllib.parse import quote

# --- THE FIX: Force OpenCV to use TCP instead of UDP for RTSP ---
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

# Clamp PyTorch threads to avoid CPU context-switching overhead
torch.set_num_threads(4) 

def flow_to_color(flow_uv):
    """Converts a 2D optical flow tensor (u, v) into an RGB image for visualization."""
    u, v = flow_uv[0], flow_uv[1]
    rad = np.sqrt(u**2 + v**2)
    a = np.arctan2(-v, -u) / np.pi
    
    fk = (a + 1) / 2 * 180  
    hsv = np.zeros((flow_uv.shape[1], flow_uv.shape[2], 3), dtype=np.uint8)
    hsv[..., 0] = fk
    hsv[..., 1] = 255
    hsv[..., 2] = np.minimum(rad * 255 / 10.0, 255)  
    
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def preprocess_frame(frame, target_size=(320, 240)):
    """Resizes, converts to RGB, and normalizes a frame to [-1, 1] for RAFT."""
    frame = cv2.resize(frame, target_size)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    tensor = torch.from_numpy(frame).permute(2, 0, 1).float()
    tensor = (tensor / 127.5) - 1.0  
    
    return tensor.unsqueeze(0)  

CAMERA_HOST = "10.101.0.7"
CAMERA_USERNAME = "admin"
CAMERA_PASSWORD = "admin123"

def build_rtsp_candidates(host, username, password):
    """Return common RTSP URLs to try for a camera at the given host."""
    auth = f"{quote(username, safe='')}:{quote(password, safe='')}"
    return [
        # Put the known working path first! 
        # /1/2 targets the Sub-Stream (H.264) which OpenCV can decode.
        f"rtsp://{auth}@{host}:554/snl/live/1/2", 
        # Fallback to main stream just in case
        f"rtsp://{auth}@{host}:554/snl/live/1/1",
    ]

def get_video_source():
    """Return RTSP candidates for the camera, or a webcam fallback on demand."""
    print("Trying RTSP stream URLs for the Sparsh camera...")
    return build_rtsp_candidates(CAMERA_HOST, CAMERA_USERNAME, CAMERA_PASSWORD)

def open_capture(source):
    if source == 0:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
        return cap

    if isinstance(source, str):
        cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
        if cap.isOpened():
            return cap
        cap.release()

    if isinstance(source, (list, tuple)):
        for candidate in source:
            print(f"Testing connection to: {candidate}")
            cap = cv2.VideoCapture(candidate, cv2.CAP_FFMPEG)
            if cap.isOpened():
                print(f">>> Successfully connected to: {candidate}")
                return cap
            cap.release()

    print("Warning: Could not open any RTSP stream, falling back to webcam.")
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    return cap

def overlay_metrics(frame, latency_ms, end_pixel_error):
    """Draw runtime metrics over the output frame."""
    annotated = frame.copy()
    cv2.putText(annotated, f"Latency: {latency_ms:.1f} ms", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(annotated, f"End Pixel Error: {end_pixel_error:.3f}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    return annotated

# ... KEEP YOUR EXISTING main() FUNCTION DOWN HERE ...