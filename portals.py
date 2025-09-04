import cv2
import time
import math
import random
import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

import mediapipe as mp
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

TITLE = "Air Portal Opening 🌀"

PINCH_FRAC           = 0.045           
CIRCLE_BUF           = 60               
CIRCLE_STD_FRAC      = 0.25             
CIRCLE_COVERAGE      = math.radians(270)
GRAB_NEAR_FRAC       = 1.3             
PUSH_DZ_THRESH       = 0.035            
PUSH_COOLDOWN_S      = 0.35
PORTAL_GROW_ON_PUSH  = 90.0             
PORTAL_INERTIA       = 0.85            
BASE_RADIUS          = 90
RING_THICK           = 6
RING_ROT_SPEED       = 1.6              
SWIRL_SPEED          = 0.4             
SLOWMO_SCALE         = 0.33


@dataclass
class HandState:
    label: str
    palm: Tuple[int,int] = (0,0)
    z: float = 0.0
    prev_pts: deque = field(default_factory=lambda: deque(maxlen=6))
    prev_zs: deque = field(default_factory=lambda: deque(maxlen=6))
    circle_buf: deque = field(default_factory=lambda: deque(maxlen=CIRCLE_BUF))
    last_push: float = 0.0
    last_circle: float = 0.0

@dataclass
class Portal:
    center: Tuple[float,float]
    radius: float
    hue_offset: float = 0.0
    angle: float = 0.0
    grabbed: bool = False
    alive: bool = True


def palm_center_px(hand_lms, w, h) -> Tuple[int,int]:
    lm = hand_lms.landmark
    idxs = [0,5,9,13,17]
    xs = [(lm[i].x * w) for i in idxs]
    ys = [(lm[i].y * h) for i in idxs]
    return int(sum(xs)/len(xs)), int(sum(ys)/len(ys))

def palm_depth(hand_lms) -> float:
    return hand_lms.landmark[9].z  

def detect_circle(buf: deque, last_trigger: float, now: float) -> Optional[Tuple[Tuple[int,int], int]]:
    if len(buf) < int(CIRCLE_BUF*0.6): return None
    pts = np.array(buf, dtype=float)
    cx, cy = pts[:,0].mean(), pts[:,1].mean()
    rs = np.hypot(pts[:,0]-cx, pts[:,1]-cy)
    r_mean = rs.mean()
    if r_mean < 25: return None
    std_ok = (rs.std()/r_mean) < CIRCLE_STD_FRAC
    angs = np.unwrap(np.arctan2(pts[:,1]-cy, pts[:,0]-cx))
    cov = angs.max() - angs.min()
    if std_ok and cov >= CIRCLE_COVERAGE and (now - last_trigger) > 0.9:
        return (int(cx), int(cy)), int(r_mean)
    return None

def theme_colors(idx: int):

    themes = [
        ((0,220,255), (0,160,255)),    # cyan / gold
        ((255,120,0), (255,200,0)),    # orange
        ((120,255,120), (0,255,100)),  # green
        ((255,80,180), (255,200,240)), # magenta
        ((180,180,255), (255,255,255)) # blue-white
    ]
    return themes[max(0,min(len(themes)-1, idx))]

def gen_swirl_tile(size: int, t: float, hue_shift: float, tint_bgr=(255,255,255)):
    s = size
    y, x = np.mgrid[0:s, 0:s].astype(np.float32)
    cx, cy = s/2, s/2
    dx, dy = (x - cx)/s, (y - cy)/s
    r = np.sqrt(dx*dx + dy*dy) * 2.2
    ang = np.arctan2(dy, dx) + t*SWIRL_SPEED*2*np.pi
    hue = (ang/(2*np.pi) + hue_shift) % 1.0
    val = 0.6 + 0.4*np.sin(8*r - t*3.0)
    sat = np.clip(0.6 + 0.4*r, 0, 1)
    hsv = np.dstack((hue*179.0, sat*255.0, np.clip(val,0,1)*255.0)).astype(np.uint8)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    # gentle tint
    tb, tg, tr = tint_bgr
    bgr = np.clip(bgr*0.9 + np.array([tb, tg, tr], dtype=np.float32)*0.1, 0, 255).astype(np.uint8)
    return bgr

def composite_circle(frame, center, radius, overlay_img):
    h, w = frame.shape[:2]
    x, y = int(center[0]), int(center[1])
    r = int(max(2, radius))
    x0, y0 = max(0, x-r), max(0, y-r)
    x1, y1 = min(w, x+r), min(h, y+r)
    if x1<=x0 or y1<=y0: return
    roi = frame[y0:y1, x0:x1]
    # resize tile to roi
    tile = cv2.resize(overlay_img, (x1-x0, y1-y0), interpolation=cv2.INTER_LINEAR)
    # circular mask
    mask = np.zeros((y1-y0, x1-x0), dtype=np.uint8)
    cv2.circle(mask, (x-x0, y-y0), min(x1-x0, y1-y0)//2, 255, -1, lineType=cv2.LINE_AA)
    mask_f = (mask/255.0)[...,None]
    roi[:] = (tile*mask_f + roi*(1-mask_f)).astype(np.uint8)


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Camera not found."); return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cv2.namedWindow(TITLE, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(TITLE, 1280, 720)
    hands_state = {"Left": HandState(label="Left"), "Right": HandState(label="Right")}
    portal: Optional[Portal] = None
    theme_idx = 0
    ring_col, tint_col = theme_colors(theme_idx)
    paused = False
    prev = time.time()

    with mp_hands.Hands(static_image_mode=False,
                        max_num_hands=2,
                        min_detection_confidence=0.6,
                        min_tracking_confidence=0.6) as hands:
        while True:
            ok, frame = cap.read()
            if not ok: break
            frame = cv2.flip(frame, 1)
            H, W = frame.shape[:2]
            MAX_R = int(min(H, W)*0.45)

            now = time.time()
            dt = max(1e-3, min(0.05, now - prev))
            prev = now
            if paused: dt *= SLOWMO_SCALE

            # detect
            res = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            # update hands
            pinch_thresh = max(22, int(PINCH_FRAC*H))
            for hs in hands_state.values():
                hs.circle_buf.append(hs.palm)  

            if res.multi_hand_landmarks:
                if res.multi_handedness and len(res.multi_handedness)==len(res.multi_hand_landmarks):
                    pairs = zip(res.multi_hand_landmarks, res.multi_handedness)
                    for lms, hd in pairs:
                        label = hd.classification[0].label
                        hs = hands_state[label]
                        hs.palm = palm_center_px(lms, W, H)
                        hs.z = palm_depth(lms)
                        hs.prev_pts.append(hs.palm)
                        hs.prev_zs.append(hs.z)
                        hs.circle_buf.append(hs.palm)
                else:
                    lmss = res.multi_hand_landmarks
                    xs = [lm.landmark[0].x for lm in lmss]
                    order = np.argsort(xs)
                    labels = ["Left","Right"] if len(lmss)==2 else ["Right"]
                    for i,lms in enumerate(lmss):
                        label = labels[int(np.where(order==i)[0][0])] if len(lmss)==2 else "Right"
                        hs = hands_state[label]
                        hs.palm = palm_center_px(lms, W, H)
                        hs.z = palm_depth(lms)
                        hs.prev_pts.append(hs.palm)
                        hs.prev_zs.append(hs.z)
                        hs.circle_buf.append(hs.palm)

            #  spawn portal
            for hs in hands_state.values():
                det = detect_circle(hs.circle_buf, hs.last_circle, now)
                if det:
                    (cx,cy), r = det
                    r = max(40, min(MAX_R, int(r*1.05)))
                    portal = Portal(center=(cx,cy), radius=float(r))
                    hs.last_circle = now
                    hs.circle_buf.clear()

 
            if portal and portal.alive:
                for hs in hands_state.values():
                    if len(hs.prev_zs) >= 2 and (now - hs.last_push) > PUSH_COOLDOWN_S:
                        if (hs.prev_zs[-2] - hs.prev_zs[-1]) > PUSH_DZ_THRESH:
                            portal.radius = min(MAX_R, portal.radius + PORTAL_GROW_ON_PUSH)
                            hs.last_push = now

                left, right = hands_state["Left"], hands_state["Right"]
                two_hands = (left.prev_pts and right.prev_pts)
                if two_hands:
                    dL = math.hypot(left.palm[0]-portal.center[0], left.palm[1]-portal.center[1])
                    dR = math.hypot(right.palm[0]-portal.center[0], right.palm[1]-portal.center[1])
                    near = (dL < portal.radius*GRAB_NEAR_FRAC) and (dR < portal.radius*GRAB_NEAR_FRAC)
                    if near:
                        target_center = ((left.palm[0]+right.palm[0])/2.0, (left.palm[1]+right.palm[1])/2.0)
                        target_radius = 0.55*math.hypot(right.palm[0]-left.palm[0], right.palm[1]-left.palm[1])
                        target_radius = max(40, min(MAX_R, target_radius))
                        cx, cy = portal.center
                        portal.center = (cx*PORTAL_INERTIA + target_center[0]*(1-PORTAL_INERTIA),
                                         cy*PORTAL_INERTIA + target_center[1]*(1-PORTAL_INERTIA))
                        portal.radius = portal.radius*PORTAL_INERTIA + target_radius*(1-PORTAL_INERTIA)
                        portal.grabbed = True
                    else:
                        portal.grabbed = False
                portal.angle += RING_ROT_SPEED*dt
                portal.hue_offset = (portal.hue_offset + dt*0.12) % 1.0

           
            canvas = frame

            
            if res.multi_hand_landmarks:
                for lms in res.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(canvas, lms, mp_hands.HAND_CONNECTIONS,
                                              mp_drawing.DrawingSpec(color=(140,140,140), thickness=1, circle_radius=2),
                                              mp_drawing.DrawingSpec(color=(80,80,80), thickness=1))

            if portal and portal.alive:
                tile = gen_swirl_tile(size=512, t=now, hue_shift=portal.hue_offset, tint_bgr=tint_col)
                composite_circle(canvas, portal.center, portal.radius, tile)
                cx, cy = int(portal.center[0]), int(portal.center[1])
                r = int(portal.radius)
                spokes = 12
                for s in range(spokes):
                    ang = portal.angle + s*(math.tau/spokes)
                    x1 = int(cx + r*math.cos(ang))
                    y1 = int(cy + r*math.sin(ang))
                    cv2.line(canvas, (cx,cy), (x1,y1), ring_col, 1, cv2.LINE_AA)
                cv2.circle(canvas, (cx,cy), r, ring_col, RING_THICK, cv2.LINE_AA)
                cv2.circle(canvas, (cx,cy), int(r*0.67), ring_col, 2, cv2.LINE_AA)
                cv2.circle(canvas, (cx,cy), int(r*0.36), ring_col, 1, cv2.LINE_AA)
                if portal.grabbed:
                    cv2.circle(canvas, (cx,cy), r+8, (255,255,255), 1, cv2.LINE_AA)


            cv2.putText(canvas, "Draw a circle to CREATE portal | Two palms near it to DRAG/RESIZE | Push palm forward to OPEN",
                        (16, H-48), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (230,230,230), 1, cv2.LINE_AA)
            cv2.putText(canvas, "1-5 theme  +/- size  SPACE slow  C clear  R reset  Q quit",
                        (16, H-18), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (230,230,230), 1, cv2.LINE_AA)

            cv2.imshow(TITLE, canvas)
            k = cv2.waitKeyEx(1)
            if k in (ord('q'), ord('Q')):
                break
            elif k in (ord('c'), ord('C')):
                portal = None
            elif k in (ord('r'), ord('R')):
                portal = None
                for hs in hands_state.values():
                    hs.prev_pts.clear(); hs.prev_zs.clear(); hs.circle_buf.clear()
            elif k in [ord('1'),ord('2'),ord('3'),ord('4'),ord('5')]:
                theme_idx = int(chr(k)) - 1
                ring_col, tint_col = theme_colors(theme_idx)
            elif k in (ord('+'), ord('=')):
                global BASE_RADIUS
                BASE_RADIUS = min(int(min(H,W)*0.45), BASE_RADIUS + 6)
                if portal: portal.radius = max(portal.radius, BASE_RADIUS)
            elif k == ord('-'):
                BASE_RADIUS = max(40, BASE_RADIUS - 6)
                if portal: portal.radius = max(40, min(portal.radius, BASE_RADIUS))
            elif k in (ord('s'), ord('S')):
                ts = int(time.time()); cv2.imwrite(f"portal_{ts}.png", canvas)
            elif k == 32:  # SPACE
                paused = not paused

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
