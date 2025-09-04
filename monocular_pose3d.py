import cv2
import numpy as np
import mediapipe as mp
import time
import argparse
import csv
from collections import deque


ap = argparse.ArgumentParser()
ap.add_argument("--camera", type=int, default=0, help="Webcam index (default 0)")
ap.add_argument("--width", type=int, default=1280, help="Capture width")
ap.add_argument("--height", type=int, default=720, help="Capture height")
ap.add_argument("--smooth", type=float, default=0.25, help="EMA smoothing 0..1 (higher = more smoothing)")
ap.add_argument("--height_m", type=float, default=0.0, help="Known person height in meters to scale pose_world to meters (optional)")
ap.add_argument("--log_csv", type=str, default="pose3d_mono.csv", help="CSV output file")
args = ap.parse_args()

SMOOTH_ALPHA = float(np.clip(args.smooth, 0.0, 1.0))
KNOWN_HEIGHT_M = max(0.0, args.height_m)

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

LM_NAMES = [
    "nose","left_eye_inner","left_eye","left_eye_outer","right_eye_inner","right_eye","right_eye_outer",
    "left_ear","right_ear","mouth_left","mouth_right",
    "left_shoulder","right_shoulder","left_elbow","right_elbow","left_wrist","right_wrist",
    "left_pinky","right_pinky","left_index","right_index","left_thumb","right_thumb",
    "left_hip","right_hip","left_knee","right_knee","left_ankle","right_ankle",
    "left_heel","right_heel","left_foot_index","right_foot_index"
]


TOP_IDX = 0  # nose 
FOOT_L = 31  # left_foot_index
FOOT_R = 32  # right_foot_index


def ema(prev, cur, alpha):
    if prev is None: return cur.copy()
    return alpha * cur + (1 - alpha) * prev

def estimate_height_and_scale(X):
    """
    Estimate a body 'height' from world landmarks (relative units) and return a scale to map to KNOWN_HEIGHT_M.
    We use the vertical distance between nose and average of both foot indices in world coordinates.
    """
    if KNOWN_HEIGHT_M <= 0:
        return 1.0, None
    if np.any(np.isnan(X[TOP_IDX])) or np.any(np.isnan(X[FOOT_L])) or np.any(np.isnan(X[FOOT_R])):
        return 1.0, None
    foot_avg = 0.5 * (X[FOOT_L] + X[FOOT_R])
    est_height_rel = np.linalg.norm(X[TOP_IDX] - foot_avg)
    if est_height_rel < 1e-6:
        return 1.0, None
    scale = KNOWN_HEIGHT_M / est_height_rel
    return scale, est_height_rel

def draw_axes(img, origin=(30,80)):
    cv2.putText(img, "coords: +X right, +Y up, +Z forward (relative)", origin, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)


cap = cv2.VideoCapture(args.camera)
if not cap.isOpened():
    raise RuntimeError(f"Camera {args.camera} not available")
cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

# CSV logger
csvfile = open(args.log_csv, "w", newline="")
log = csv.writer(csvfile)
log.writerow(["ts","joint","x","y","z","units"])  

# Pose
pose = mp_pose.Pose(model_complexity=1, enable_segmentation=False, smooth_landmarks=True)

prev_world = None
scale_info_txt = "scale: relative (no height set)"
last_scale_compute = 0.0
scale_factor = 1.0

print("[i] press ESC to quit")
try:
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose.process(frame_rgb)

        if res.pose_landmarks:
            mp_draw.draw_landmarks(
                frame,
                res.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_styles.get_default_pose_landmarks_style()
            )

       
        X_world = None
        if res.pose_world_landmarks and len(res.pose_world_landmarks.landmark) == 33:
            X_world = np.array([[lm.x, lm.y, lm.z] for lm in res.pose_world_landmarks.landmark], dtype=np.float32)
            
            X_world = ema(prev_world, X_world, SMOOTH_ALPHA)
            prev_world = X_world

            if KNOWN_HEIGHT_M > 0 and (time.time() - last_scale_compute > 0.3):
                scale_factor, est_h = estimate_height_and_scale(X_world)
                if est_h is not None:
                    scale_info_txt = f"scale: ~meters (KNOWN {KNOWN_HEIGHT_M:.2f} m)"
                last_scale_compute = time.time()

          
            X_out = X_world * scale_factor

           
            ts = time.time()
            units = "m" if KNOWN_HEIGHT_M > 0 else "rel"
            for i, name in enumerate(LM_NAMES):
                x,y,z = X_out[i].tolist()
                log.writerow([ts, name, float(x), float(y), float(z), units])

          
            cv2.putText(frame, f"3D joints: {units}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
            cv2.putText(frame, scale_info_txt, (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)
            draw_axes(frame, (10,90))
        else:
            cv2.putText(frame, "no pose yet…", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)

        cv2.imshow("Monocular Pose 3D (MediaPipe)", frame)
        if (cv2.waitKey(1) & 0xFF) == 27:
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    pose.close()
    csvfile.close()
    print(f"[✓] CSV saved to {args.log_csv}")
