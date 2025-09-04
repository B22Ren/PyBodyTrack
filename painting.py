import cv2
import time
import math
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple


import mediapipe as mp
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

WINDOW_TITLE = "Air Painting in 3D Space "


@dataclass
class Camera:
    yaw: float = 0.0
    pitch: float = 0.0
    dist: float = 1200.0
    fov: float = 900.0

    def project(self, P: np.ndarray, w: int, h: int) -> Tuple[int, int, float]:
        cy, sy = math.cos(self.yaw), math.sin(self.yaw)
        cp, sp = math.cos(self.pitch), math.sin(self.pitch)
        Px =  cy*P[0] + sy*P[2]
        Pz = -sy*P[0] + cy*P[2]
        Py  =  cp*P[1] - sp*Pz
        Pz2 =  sp*P[1] + cp*Pz
        z_cam = -(Pz2 + self.dist)
        eps = 1e-6
        s = self.fov / max(eps, (self.fov - z_cam))
        u = int(w/2 + Px * s)
        v = int(h/2 - Py * s)
        return u, v, z_cam

# Strokes
@dataclass
class Stroke:
    color: Tuple[int,int,int]
    thickness: int
    points: List[np.ndarray] = field(default_factory=list)


def lerp(a, b, t): return a*(1-t) + b*t

def pinch_distance_px(hand_lms, w, h) -> float:
    lm = hand_lms.landmark
    ix, iy = int(lm[8].x*w), int(lm[8].y*h)
    tx, ty = int(lm[4].x*w), int(lm[4].y*h)
    return math.hypot(ix-tx, iy-ty)

def fingertip_xyz(hand_lms, w, h, z_scale_px: float) -> np.ndarray:
    lm = hand_lms.landmark
    lx = lm[8].x * w
    ly = lm[8].y * h
    lz = lm[8].z
    x = lx - w/2
    y = -(ly - h/2)
    z = -lz * z_scale_px
    return np.array([x, y, z], dtype=float)

def render_strokes(img, cam: Camera, strokes: List[Stroke]):
    h, w = img.shape[:2]
    for stroke in strokes:
        pts2d, depths = [], []
        for P in stroke.points:
            u, v, z = cam.project(P, w, h)
            pts2d.append((u, v)); depths.append(z)
        for i in range(1, len(pts2d)):
            p1, p2 = pts2d[i-1], pts2d[i]
            z = 0.5*(depths[i-1] + depths[i])
            t = int(max(1, stroke.thickness * (1.0 + 0.0006*z)))
            cv2.line(img, p1, p2, stroke.color, t, cv2.LINE_AA)

def draw_grid(img, cam: Camera):
    h, w = img.shape[:2]
    y_plane = -200
    grid_color = (60,60,60)
    xs = np.arange(-1200, 1201, 200)
    zs = np.arange(-1200, 1201, 200)
    for x in xs:
        u1,v1,_ = cam.project(np.array([x, y_plane, -1200.0]), w, h)
        u2,v2,_ = cam.project(np.array([x, y_plane,  1200.0]), w, h)
        cv2.line(img, (u1,v1), (u2,v2), grid_color, 1, cv2.LINE_AA)
    for z in zs:
        u1,v1,_ = cam.project(np.array([-1200.0, y_plane, z]), w, h)
        u2,v2,_ = cam.project(np.array([ 1200.0, y_plane, z]), w, h)
        cv2.line(img, (u1,v1), (u2,v2), grid_color, 1, cv2.LINE_AA)

    origin = np.array([0.0, 0.0, 0.0])
    X = np.array([300.0, 0.0, 0.0])
    Y = np.array([0.0, 300.0, 0.0])
    Z = np.array([0.0, 0.0, 300.0])
    uo, vo, _ = cam.project(origin, w, h)
    ux, vx, _ = cam.project(X, w, h)
    uy, vy, _ = cam.project(Y, w, h)
    uz, vz, _ = cam.project(Z, w, h)
    cv2.line(img, (uo,vo), (ux,vx), (0,0,255), 2, cv2.LINE_AA)
    cv2.line(img, (uo,vo), (uy,vy), (0,255,0), 2, cv2.LINE_AA)
    cv2.line(img, (uo,vo), (uz,vz), (255,0,0), 2, cv2.LINE_AA)
    cv2.circle(img, (uo,vo), 4, (200,200,200), -1, cv2.LINE_AA)

KEY_LEFT  = {2424832, 81, 65361}
KEY_RIGHT = {2555904, 83, 65363}
KEY_UP    = {2490368, 82, 65362}
KEY_DOWN  = {2621440, 84, 65364}

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Camera not found.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_TITLE, 1280, 720)

    cam = Camera()

    palette = [
        (255,255,255),
        (0,200,255),
        (0,120,255),
        (0,255,0),
        (255,0,0),
        (255,0,255),
    ]
    color_idx = 0
    thickness = 4

    strokes: List[Stroke] = []
    drawing = False
    last_draw_pt = None
    smooth_curr = None
    z_scale_px = 1200.0

    with mp_hands.Hands(static_image_mode=False,
                        max_num_hands=1,
                        min_detection_confidence=0.6,
                        min_tracking_confidence=0.6) as hands:

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = hands.process(rgb)

            if res.multi_hand_landmarks:
                hand = res.multi_hand_landmarks[0]
                pinch_thresh = max(22, int(0.045 * h))
                is_pinch = pinch_distance_px(hand, w, h) < pinch_thresh

                raw_pt = fingertip_xyz(hand, w, h, z_scale_px)
                smooth_curr = raw_pt if smooth_curr is None else lerp(smooth_curr, raw_pt, 0.35)

                if is_pinch and not drawing:
                    drawing = True
                    strokes.append(Stroke(color=palette[color_idx], thickness=thickness))
                    strokes[-1].points.append(smooth_curr.copy())
                    last_draw_pt = smooth_curr.copy()
                elif is_pinch and drawing:
                    if last_draw_pt is None or np.linalg.norm(smooth_curr - last_draw_pt) > 1.2:
                        strokes[-1].points.append(smooth_curr.copy())
                        last_draw_pt = smooth_curr.copy()
                elif (not is_pinch) and drawing:
                    drawing = False
                    last_draw_pt = None
            else:
                drawing = False
                smooth_curr = None
                last_draw_pt = None

            canvas = frame
            draw_grid(canvas, cam)
            render_strokes(canvas, cam, strokes)

            cv2.putText(canvas, f"Color[{color_idx+1}]  Thickness: {thickness}",
                        (16, h-48), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)
            cv2.putText(canvas,
                        "Pinch=paint | Arrows: rotate  [ / ]: zoom  1-6: colors  +/-: size  c: clear  u: undo  s: save  R: reset  Q: quit",
                        (16, h-18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230,230,230), 1, cv2.LINE_AA)

            
            if res.multi_hand_landmarks:
                mp_drawing.draw_landmarks(canvas, res.multi_hand_landmarks[0],
                                          mp_hands.HAND_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=(160,160,160), thickness=1, circle_radius=2),
                                          mp_drawing.DrawingSpec(color=(100,100,100), thickness=1))

            cv2.imshow(WINDOW_TITLE, canvas)

            
            k = cv2.waitKeyEx(1)

            if k in (ord('q'), ord('Q')):
                break
            elif k in (ord('c'), ord('C')):
                strokes.clear()
            elif k in (ord('u'), ord('U')):
                if strokes: strokes.pop()
            elif k in (ord('s'), ord('S')):
                ts = int(time.time()); cv2.imwrite(f"air_paint_{ts}.png", canvas)
            elif k in (ord('r'), ord('R')):
                cam = Camera()  # reset camera
            elif k in (ord('+'), ord('=')):
                thickness = min(24, thickness + 1)
            elif k == ord('-'):
                thickness = max(1, thickness - 1)
            elif k in [ord(str(d)) for d in range(1,7)]:
                color_idx = int(chr(k)) - 1
            elif k in KEY_LEFT:
                cam.yaw -= 0.05
            elif k in KEY_RIGHT:
                cam.yaw += 0.05
            elif k in KEY_UP:
                cam.pitch += 0.05
            elif k in KEY_DOWN:
                cam.pitch -= 0.05
            elif k == ord('['):
                cam.dist = max(200.0, cam.dist - 50.0)
            elif k == ord(']'):
                cam.dist = min(4000.0, cam.dist + 50.0)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
