import cv2
import time
import math
import random
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

# -------- Hand tracking (MediaPipe) ----------
import mediapipe as mp
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

WINDOW_TITLE = "Air Bow & Arrow 🎯"

# ---- Gameplay ----
GRAVITY = 1800.0           # pixels/s^2 downward
AIR_DRAG = 0.02            # small drag
MAX_POWER_PX = 350.0       # cap draw length used for power
VEL_SCALE = 4.0            # px draw -> px/s multiplier
PINCH_THRESH_FRAC = 0.045  # pinch threshold relative to frame height
DEBOUNCE_S = 0.18          # release debounce to avoid double shots
TRAIL_LEN = 18
ARROW_LEN_FRAC = 0.06      # arrow drawn length as fraction of frame width

# ---- UI----
STRING_ALPHA = 0.4
TARGET_R_FRAC = 0.09
FONT = cv2.FONT_HERSHEY_SIMPLEX

@dataclass
class HandInfo:
    label: str                         # "Left" or "Right" (subject's hands)
    lm: List[Tuple[int,int]]
    index_tip: Tuple[int,int]
    thumb_tip: Tuple[int,int]
    wrist: Tuple[int,int]
    pinch: bool

@dataclass
class Arrow:
    pos: np.ndarray                    
    vel: np.ndarray                    
    trail: List[Tuple[int,int]] = field(default_factory=list)
    alive: bool = True

@dataclass
class Target:
    center: Tuple[int,int]
    radius: int

def norm(v):
    n = math.hypot(v[0], v[1])
    return (v[0]/n, v[1]/n) if n > 1e-6 else (0.0, 0.0)

def hand_pixel_list(hand_landmarks, w, h) -> List[Tuple[int,int]]:
    return [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks.landmark]

def parse_hands(res, w, h, pinch_px_thresh: int) -> List[HandInfo]:
    """Return list of HandInfo with pixel coordinates & pinch state."""
    hands: List[HandInfo] = []
    if not res.multi_hand_landmarks:
        return hands

    
    handedness = getattr(res, "multi_handedness", None)
    if handedness and len(handedness) == len(res.multi_hand_landmarks):
        pairs = zip(res.multi_hand_landmarks, handedness)
        for lms, handed in pairs:
            label = handed.classification[0].label  # "Left" / "Right"
            pts = hand_pixel_list(lms, w, h)
            ix, iy = pts[8]
            tx, ty = pts[4]
            pinch = math.hypot(ix - tx, iy - ty) < pinch_px_thresh
            hands.append(HandInfo(
                label=label, lm=pts, index_tip=(ix,iy), thumb_tip=(tx,ty),
                wrist=pts[0], pinch=pinch
            ))
    else:
        
        lmss = res.multi_hand_landmarks
        xs = [lm.landmark[0].x for lm in lmss]
        order = np.argsort(xs)
        labels = ["Left","Right"] if len(lmss)==2 else ["Right"]
        mapping = {int(order[i]): labels[i] for i in range(len(labels))}
        for i,lms in enumerate(lmss):
            label = mapping.get(i, "Right")
            pts = hand_pixel_list(lms, w, h)
            ix, iy = pts[8]; tx, ty = pts[4]
            pinch = math.hypot(ix - tx, iy - ty) < pinch_px_thresh
            hands.append(HandInfo(
                label=label, lm=pts, index_tip=(ix,iy), thumb_tip=(tx,ty),
                wrist=pts[0], pinch=pinch
            ))
    return hands

def draw_target(img, tgt: Target):
    cx, cy = tgt.center
    r = tgt.radius
    rings = [1.00, 0.80, 0.60, 0.40, 0.20]
    colors = [(40,40,40), (70,70,180), (40,40,40), (70,70,180), (40,40,40)]
    for frac, col in zip(rings, colors):
        cv2.circle(img, (cx,cy), int(r*frac), col, 2, lineType=cv2.LINE_AA)
    cv2.circle(img, (cx,cy), int(r*0.08), (0,220,255), -1, lineType=cv2.LINE_AA)

def score_hit(dist, r):
    """Return points for a hit distance from center."""
    if dist <= r*0.20: return 10
    if dist <= r*0.40: return 8
    if dist <= r*0.60: return 6
    if dist <= r*0.80: return 4
    if dist <= r*1.00: return 2
    return 0

def draw_arrow(img, pos, vel, w):
    
    length = max(30, int(w * ARROW_LEN_FRAC))
    vx, vy = vel
    nx, ny = norm((vx, vy))
    tip = (int(pos[0] + nx*length), int(pos[1] + ny*length))
    tail = (int(pos[0] - nx*length*0.4), int(pos[1] - ny*length*0.4))
    cv2.line(img, tail, tip, (230,230,230), 3, cv2.LINE_AA)
    
    left = (int(tip[0] - ny*8), int(tip[1] + nx*8))
    right = (int(tip[0] + ny*8), int(tip[1] - nx*8))
    cv2.fillConvexPoly(img, np.array([tip,left,right]), (230,230,230))

def draw_string_and_bow(img, anchor, pull, power, w):
    # Bow: small arc near anchor; String: line to pull
    ax, ay = anchor; px, py = pull
    # string
    overlay = img.copy()
    cv2.line(overlay, (ax,ay), (px,py), (0,255,255), 2, cv2.LINE_AA)
    cv2.addWeighted(overlay, STRING_ALPHA, img, 1-STRING_ALPHA, 0, img)

    # simple bow arc (perpendicular to pull dir)
    dx, dy = px-ax, py-ay
    nx, ny = norm((dx,dy))
    # perpendicular
    bx, by = -ny, nx
    bow_size = 26 + int(min(power, MAX_POWER_PX) * 0.10)
    p1 = (ax + int(bx*bow_size), ay + int(by*bow_size))
    p2 = (ax - int(bx*bow_size), ay - int(by*bow_size))
    cv2.ellipse(img, (ax,ay), (bow_size, int(bow_size*0.35)), 0, 0, 360, (200,200,200), 2, cv2.LINE_AA)
    cv2.circle(img, anchor, 5, (0,255,255), -1, cv2.LINE_AA)

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Camera not found.")
        return

    # ask for 1280x720 (camera may ignore)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_TITLE, 1280, 720)

    # game state
    arrows: List[Arrow] = []
    total_score, shots = 0, 0
    last_release = 0.0
    is_drawing = False
    anchor_pt: Optional[Tuple[int,int]] = None
    pull_pt: Optional[Tuple[int,int]] = None

    prev_time = time.time()
    target: Optional[Target] = None

    with mp_hands.Hands(static_image_mode=False,
                        max_num_hands=2,
                        min_detection_confidence=0.6,
                        min_tracking_confidence=0.6) as hands:

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

          
            if target is None:
                target = Target(center=(int(w*0.82), int(h*0.45)), radius=int(w*TARGET_R_FRAC))

            
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = hands.process(rgb)

            pinch_px = max(22, int(PINCH_THRESH_FRAC * h))
            hands_info = parse_hands(res, w, h, pinch_px)

            # the pinching hand is the pull hand; the other is bow hand.
            pull_hand = next((hd for hd in hands_info if hd.pinch), None)
            bow_hand = None
            if pull_hand and len(hands_info) >= 2:
                # pick the other hand
                bow_hand = next((hd for hd in hands_info if hd is not pull_hand), None)
            elif len(hands_info) == 2:
                # no pinch yet: by default, left = bow, right = pull
                left = next((hd for hd in hands_info if hd.label == "Left"), None)
                right = next((hd for hd in hands_info if hd.label == "Right"), None)
                bow_hand, pull_hand = left, right

           
            current_anchor = None
            current_pull = None
            if bow_hand:
                
                current_anchor = bow_hand.lm[5] if len(bow_hand.lm) > 5 else bow_hand.wrist
            if pull_hand:
                current_pull = pull_hand.index_tip

           
            now = time.time()
            dt = max(1e-3, min(0.05, now - prev_time)) 
            prev_time = now

            # 3) update all arrows
            for arr in arrows:
                if not arr.alive: 
                    continue
                # gravity & drag
                arr.vel[1] += GRAVITY * dt
                arr.vel *= (1.0 - AIR_DRAG * dt)
                arr.pos += arr.vel * dt

                # trail
                arr.trail.append((int(arr.pos[0]), int(arr.pos[1])))
                if len(arr.trail) > TRAIL_LEN:
                    arr.trail.pop(0)

                # bounds
                if arr.pos[0] < -50 or arr.pos[0] > w+50 or arr.pos[1] > h+50:
                    arr.alive = False

                # hit test
                if target and arr.alive:
                    dx = arr.pos[0] - target.center[0]
                    dy = arr.pos[1] - target.center[1]
                    d = math.hypot(dx, dy)
                    if d <= target.radius:
                        points = score_hit(d, target.radius)
                        total_score += points
                        shots += 1
                        arr.alive = False

            
            if pull_hand and current_anchor and current_pull:
                
                if pull_hand.pinch:
                    is_drawing = True
                    anchor_pt = current_anchor
                    pull_pt = current_pull
                elif is_drawing:
                    
                    if now - last_release > DEBOUNCE_S:
                        ax, ay = anchor_pt
                        px, py = pull_pt
                        dx = ax - px
                        dy = ay - py
                        power = min(MAX_POWER_PX, math.hypot(dx, dy))
                        if power > 25.0:  # minimal draw to register a shot
                            vel = np.array([dx, dy], dtype=float) * VEL_SCALE
                            arrows.append(Arrow(pos=np.array([ax, ay], dtype=float),
                                                vel=vel))
                            shots += 1
                            last_release = now
                    is_drawing = False
            else:
                is_drawing = False

            
            draw_target(frame, target)

            # power meter & string
            if is_drawing and anchor_pt and pull_pt:
                ax, ay = anchor_pt
                px, py = pull_pt
                power = min(MAX_POWER_PX, math.hypot(ax - px, ay - py))
                draw_string_and_bow(frame, anchor_pt, pull_pt, power, w)

                # power bar
                bar_w = int(w * 0.22)
                bar_h = 14
                x0 = int(w * 0.05)
                y0 = int(h * 0.08)
                cv2.rectangle(frame, (x0, y0), (x0 + bar_w, y0 + bar_h), (200, 200, 200), 1)
                fill = int(bar_w * (power / MAX_POWER_PX))
                fill = max(0, min(bar_w, fill))
                cv2.rectangle(frame, (x0, y0), (x0 + fill, y0 + bar_h), (0, 220, 255), -1)
                cv2.putText(frame, "Power", (x0, y0 - 6), FONT, 0.6, (230,230,230), 1, cv2.LINE_AA)

        
            for arr in arrows:
                if not arr.alive and not arr.trail:
                    continue
                # trail
                if len(arr.trail) >= 2:
                    for i in range(1, len(arr.trail)):
                        cv2.line(frame, arr.trail[i-1], arr.trail[i], (160,160,160), 2, cv2.LINE_AA)
                if arr.alive:
                    draw_arrow(frame, arr.pos, arr.vel, w)

            # draw landmarks last (for feedback)
            if res.multi_hand_landmarks:
                for lms in res.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, lms, mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(120,120,120), thickness=1, circle_radius=2),
                        mp_drawing.DrawingSpec(color=(80,80,80), thickness=1)
                    )

            # HUD
            cv2.putText(frame, f"Score: {total_score}", (int(w*0.04), int(h*0.95)),
                        FONT, 0.8, (255,255,255), 2, cv2.LINE_AA)
            cv2.putText(frame, "Pinch to grab, pull, release to shoot | q: quit  c: clear  r: reset  t: move target",
                        (int(w*0.04), int(h*0.98)), FONT, 0.55, (230,230,230), 1, cv2.LINE_AA)

            cv2.imshow(WINDOW_TITLE, frame)

            # input keys
            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'):
                break
            elif k == ord('c'):
                arrows.clear()
            elif k == ord('r'):
                arrows.clear(); total_score = 0; shots = 0
            elif k == ord('t') and target:
                tx = random.randint(int(w*0.65), int(w*0.88))
                ty = random.randint(int(h*0.25), int(h*0.70))
                target.center = (tx, ty)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
