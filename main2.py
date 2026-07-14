import cv2
import torch
import numpy as np
import time
from torchvision.models.optical_flow import raft_small
from urllib.parse import quote

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

# OPTIMIZATION: Reduced target size to 320x240 (still a multiple of 8)
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
        f"rtsp://{auth}@{host}:554/stream1",
        f"rtsp://{auth}@{host}:554/ch01/0",
        f"rtsp://{auth}@{host}:554/cam/realmonitor?channel=1&subtype=0",
        f"rtsp://{auth}@{host}:554/live",
        f"rtsp://{auth}@{host}:554/snl/live/1/1",
        f"rtsp://{auth}@{host}:554/Streaming/Channels/101",
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
            print(f"Trying RTSP URL: {candidate}")
            cap = cv2.VideoCapture(candidate, cv2.CAP_FFMPEG)
            if cap.isOpened():
                print(f"Connected to RTSP stream: {candidate}")
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
    cv2.putText(
        annotated,
        f"Latency: {latency_ms:.1f} ms",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        annotated,
        f"End Pixel Error: {end_pixel_error:.3f}",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return annotated

def main():
    print("Initializing RAFT-Small architecture...")
    device = torch.device('cpu')
    
    model = raft_small(weights=None)
    
    print("Loading fine-tuned weights...")
    state_dict = torch.load("best_raft_model.pth", map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    
    # Optional: fuse modules if applicable, but eval() is the crucial step
    model.eval()
    model.to(device)
    print("Model loaded successfully.")

    cap = open_capture(get_video_source())
    
    if not cap.isOpened():
        print("Error: Could not open video source.")
        return

    ret, prev_frame = cap.read()
    if not ret:
        print("Error: Could not read frame from camera.")
        return
        
    prev_tensor = preprocess_frame(prev_frame)

    print("Starting live inference. Press 'q' to quit.")
    
    # Use torch.inference_mode() - it is slightly faster than no_grad()
    with torch.inference_mode():
        while True:
            frame_start = time.perf_counter()

            ret, current_frame = cap.read()
            if not ret:
                break
                
            current_tensor = preprocess_frame(current_frame)
            
            # OPTIMIZATION: Hardcap the GRU iterations to 4 (default is 12)
            flow_predictions = model(prev_tensor, current_tensor, num_flow_updates=4)
            
            final_flow = flow_predictions[-1][0].numpy()

            flow_vis = flow_to_color(final_flow)
            end_pixel_error = float(np.linalg.norm(final_flow, axis=0).mean())
            
            resized_current = cv2.resize(current_frame, (320, 240))
            combined_view = np.hstack((resized_current, flow_vis))
            latency_ms = (time.perf_counter() - frame_start) * 1000.0
            combined_view = overlay_metrics(combined_view, latency_ms, end_pixel_error)
            
            cv2.imshow('Left: Live Feed | Right: Optical Flow', combined_view)
            
            prev_tensor = current_tensor
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()