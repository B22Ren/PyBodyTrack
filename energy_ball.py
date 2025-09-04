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

WINDOW_TITLE = "Air Energy Ball (palm charge) ⚡️🌀"


PINCH_FRAC      = 0.045         
CHARGE_GROW_S   = 55.0         
CHARGE_SHRINK_S = 120.0         
MOTION_STEADY_PX_S = 90.0       
PUSH_DZ_THRESH  = 0.035         
PUSH_COOLDOWN_S = 0.35
VEL_SCALE       = 0.30        
ORB_MIN_R       = 16
ORB_MAX_R       = 90
BASE_ORB_R      = 28

DRAG            = 0.06         
GRAVITY         = 0.0         
TRAIL_LEN       = 26
PARTICLE_BURST  = 24
RING_LIFETIME   = 2.8
RING_THICK      = 6
CIRCLE_BUF      = 60
CIRCLE_STD_FRAC = 0.25
CIRCLE_COVERAGE = math.radians(270)
SLOWMO_SCALE    = 0.33


@dataclass
class HandState:
    label: str
    palm: Tuple[int,int] = (0,0)
    palm_z: float = 0.0
    prev_pts: deque = field(default_factory=lambda: deque(maxlen=6))   # for velocity
    prev_zs: deque = field(default_factory=lambda: deque(maxlen=6))
    circle_buf: deque = field(default_factory=lambda: deque(maxlen=CIRCLE_BUF))
    charging_r: float = BASE_ORB_R
    last_push_time: float = 0.0
    last_circle_time: float = 0.0

@dataclass
class Particle:
    pos: np.ndarray
    vel: np.ndarray
    life: float
    color: Tuple[int,int,int]

@dataclass
class Orb:
    pos: np.ndarray
    vel: np.ndarray
    r: float
    color: Tuple[int,int,int]
    trail: deque = field(default_factory=lambda: deque(maxlen=TRAIL_LEN))
    alive: bool = True

@dataclass
class RingSpell:
    center: Tuple[int,int]
    radius: int
    color: Tuple[int,int,int]
    t: float = 0.0
    life: float = RING_LIFETIME
    angle: float = 0.0


def palm_center_px(hand_lms, w, h) -> Tuple[int,int]:
    """Simple palm center: average of wrist + (index/middle/ring/pinky MCP)."""
    lm = hand_lms.landmark
    idxs = [0, 5, 9, 13, 17]
    xs = [(lm[i].x * w) for i in idxs]
    ys = [(lm[i].y * h) for i in idxs]
    return int(sum(xs)/len(xs)), int(sum(ys)/len(ys))

def palm_depth(hand_lms) -> float:
    return hand_lms.landmark[9].z

def add_glow(img, center, radius, color, alpha=0.35, rings=3):
    x,y = center
    overlay = img.copy()
    for i in range(rings):
        r = int(radius*(1.0 + 0.35*i))
        a = max(0, alpha - i*0.12)
        cv2.circle(overlay, (x,y), r, color, -1, lineType=cv2.LINE_AA)
        cv2.addWeighted(overlay, a, img, 1-a, 0, img)

def draw_orb(img, orb: Orb):
    add_glow(img, (int(orb.pos[0]), int(orb.pos[1])), int(orb.r), orb.color, 0.30, 3)
    cv2.circle(img, (int(orb.pos[0]), int(orb.pos[1])), int(orb.r*0.35), (255,255,255), 2, cv2.LINE_AA)
    tr = list(orb.trail)
    for i in range(1, len(tr)):
        a = i / len(tr)
        col = tuple(int(a*c + (1-a)*40) for c in orb.color)
        cv2.line(img, tr[i-1], tr[i], col, max(1, int(orb.r*0.12)), cv2.LINE_AA)

def burst_particles(particles: List[Particle], center, base_color):
    cx, cy = center
    for _ in range(PARTICLE_BURST):
        ang = random.random()*math.tau
        spd = random.uniform(120, 380)
        vel = np.array([math.cos(ang)*spd, math.sin(ang)*spd], dtype=float)
        color = tuple(int(min(255, c + random.randint(-20,40))) for c in base_color)
        particles.append(Particle(pos=np.array([cx,cy], dtype=float), vel=vel, life=random.uniform(0.35,0.8), color=color))

def draw_ring(img, ring: RingSpell):
    cx, cy = ring.center
    r = ring.radius
    a = ring.angle
    spokes = 10
    for s in range(spokes):
        ang = a + s*(math.tau/spokes)
        x1 = int(cx + r*math.cos(ang))
        y1 = int(cy + r*math.sin(ang))
        cv2.line(img, (cx,cy), (x1,y1), ring.color, 1, cv2.LINE_AA)
    cv2.circle(img, (cx,cy), r, ring.color, RING_THICK, cv2.LINE_AA)
    cv2.circle(img, (cx,cy), int(r*0.65), ring.color, 2, cv2.LINE_AA)
    cv2.circle(img, (cx,cy), int(r*0.35), ring.color, 1, cv2.LINE_AA)

def detect_circle(buf: deque, last_trigger: float, now: float) -> Optional[Tuple[Tuple[int,int], int]]:
    if len(buf) < int(CIRCLE_BUF*0.6):
        return None
    pts = np.array(buf, dtype=float)
    cx, cy = pts[:,0].mean(), pts[:,1].mean()
    rs = np.hypot(pts[:,0]-cx, pts[:,1]-cy)
    r_mean = rs.mean()
    if r_mean < 25:  # too tiny
        return None
    std_ok = (rs.std()/r_mean) < CIRCLE_STD_FRAC
    angs = np.unwrap(np.arctan2(pts[:,1]-cy, pts[:,0]-cx))
    cov = angs.max()-angs.min()
    if std_ok and cov >= CIRCLE_COVERAGE and (now - last_trigger) > 0.9:
        return (int(cx), int(cy)), int(r_mean)
    return None

def theme_colors(idx: int) -> Tuple[Tuple[int,int,int], Tuple[int,int,int]]:
    themes = [
        ((0,220,255), (0,160,255)),     # cyan / gold
        ((255,120,0), (255,200,0)),     # orange
        ((120,255,120), (0,255,100)),   # green
        ((255,80,180), (255,200,240)),  # magenta
        ((180,180,255), (255,255,255)), # blue-white
    ]
    idx = max(0, min(len(themes)-1, idx))
    return themes[idx]


def main():
    global BASE_ORB_R  

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Camera not found."); return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_TITLE, 1280, 720)

    hands_state = {"Left": HandState(label="Left"), "Right": HandState(label="Right")}
    orbs: List[Orb] = []
    rings: List[RingSpell] = []
    particles: List[Particle] = []

    theme_idx = 0
    orb_col, ring_col = theme_colors(theme_idx)
    score = 0
    slowmo = False

    prev = time.time()

    with mp_hands.Hands(static_image_mode=False,
                        max_num_hands=2,
                        min_detection_confidence=0.6,
                        min_tracking_confidence=0.6) as hands:
        while True:
            ok, frame = cap.read()
            if not ok: break
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            now = time.time()
            dt = now - prev; prev = now
            if slowmo: dt *= SLOWMO_SCALE
            dt = max(1e-3, min(0.05, dt))

           
            res = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            # Map hands 
            for hs in hands_state.values():
                hs.charging_r = max(ORB_MIN_R, hs.charging_r - CHARGE_SHRINK_S*0.25*dt)

            if res.multi_hand_landmarks:

                if res.multi_handedness and len(res.multi_handedness) == len(res.multi_hand_landmarks):
                    pairs = zip(res.multi_hand_landmarks, res.multi_handedness)
                    for lms, hd in pairs:
                        label = hd.classification[0].label  # "Left" or "Right"
                        update_hand(hands_state[label], lms, w, h, dt, now, orbs, particles, orb_col)
                else:
                    lmss = res.multi_hand_landmarks
                    xs = [lm.landmark[0].x for lm in lmss]
                    order = np.argsort(xs)
                    labels = ["Left","Right"] if len(lmss)==2 else ["Right"]
                    mapping = {int(order[i]): labels[i] for i in range(len(labels))}
                    for i,lms in enumerate(lmss):
                        update_hand(hands_state[mapping.get(i,"Right")], lms, w, h, dt, now, orbs, particles, orb_col)

            # Circle detection
            for hs in hands_state.values():
                det = detect_circle(hs.circle_buf, hs.last_circle_time, now)
                if det:
                    center, rad = det
                    rings.append(RingSpell(center=center, radius=int(rad*1.05), color=ring_col))
                    burst_particles(particles, center, ring_col)
                    score += 5
                    hs.last_circle_time = now
                    hs.circle_buf.clear()

            # orbs
            for orb in orbs:
                if not orb.alive: continue
                orb.vel *= (1.0 - DRAG*dt)
                orb.vel[1] += GRAVITY*dt
                orb.pos += orb.vel*dt
                orb.trail.append((int(orb.pos[0]), int(orb.pos[1])))
                if orb.pos[0] < -120 or orb.pos[0] > w+120 or orb.pos[1] < -120 or orb.pos[1] > h+120:
                    orb.alive = False

            # rings
            for ring in rings:
                ring.t += dt
                ring.angle += 1.8 * dt
            rings = [r for r in rings if r.t <= r.life]

            # particles - decay life
            for p in particles:
                p.life -= dt
            particles = [p for p in particles if p.life > 0]
            for p in particles:
                p.vel *= (1.0 - 0.8 * dt)
                p.pos += p.vel * dt

          
            canvas = frame

            
            if res.multi_hand_landmarks:
                for lms in res.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(canvas, lms, mp_hands.HAND_CONNECTIONS,
                                              mp_drawing.DrawingSpec(color=(140,140,140), thickness=1, circle_radius=2),
                                              mp_drawing.DrawingSpec(color=(80,80,80), thickness=1))

            
            for hs in hands_state.values():
                add_glow(canvas, hs.palm, int(hs.charging_r), orb_col, 0.30, 3)
                cv2.circle(canvas, hs.palm, int(hs.charging_r*0.35), (255,255,255), 2, cv2.LINE_AA)

            # Orbs
            for orb in orbs:
                if orb.alive: draw_orb(canvas, orb)

            # Rings
            for ring in rings:
                draw_ring(canvas, ring)

            # Particles
            for p in particles:
                a = max(0.0, min(1.0, p.life / 0.8))
                col = tuple(int(a*c + (1-a)*40) for c in p.color)
                cv2.circle(canvas, (int(p.pos[0]), int(p.pos[1])), 3, col, -1, cv2.LINE_AA)

            # HUD
            cv2.putText(canvas, f"Score: {score}", (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2, cv2.LINE_AA)
            cv2.putText(canvas, "Hold palm steady to CHARGE | Push forward to LAUNCH | Draw a circle to SUMMON RING",
                        (16, h-48), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230,230,230), 1, cv2.LINE_AA)
            cv2.putText(canvas, "1-5 theme  +/- size  SPACE slow-mo  C clear  R reset  Q quit",
                        (16, h-18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230,230,230), 1, cv2.LINE_AA)

            cv2.imshow(WINDOW_TITLE, canvas)

          
            k = cv2.waitKeyEx(1)
            if k in (ord('q'), ord('Q')):
                break
            elif k in (ord('c'), ord('C')):
                orbs.clear(); rings.clear(); particles.clear()
                for hs in hands_state.values(): hs.charging_r = BASE_ORB_R
            elif k in (ord('r'), ord('R')):
                orbs.clear(); rings.clear(); particles.clear()
                for hs in hands_state.values():
                    hs.charging_r = BASE_ORB_R; hs.prev_pts.clear(); hs.prev_zs.clear(); hs.circle_buf.clear()
                score = 0
            elif k in [ord('1'),ord('2'),ord('3'),ord('4'),ord('5')]:
                theme_idx = int(chr(k)) - 1
                orb_col, ring_col = theme_colors(theme_idx)
            elif k in (ord('+'), ord('=')):
                BASE_ORB_R = min(ORB_MAX_R-4, BASE_ORB_R + 4)
                for hs in hands_state.values(): hs.charging_r = max(hs.charging_r, BASE_ORB_R)
            elif k == ord('-'):
                BASE_ORB_R = max(ORB_MIN_R+4, BASE_ORB_R - 4)
                for hs in hands_state.values(): hs.charging_r = max(hs.charging_r, BASE_ORB_R)
            elif k == 32:  # SPACE
                slowmo = not slowmo

    cap.release()
    cv2.destroyAllWindows()


def update_hand(hs: HandState, lms, w, h, dt, now, orbs, particles, orb_col):
    
    palm = palm_center_px(lms, w, h)
    z = palm_depth(lms)
    hs.palm = palm
    hs.palm_z = z
    hs.prev_pts.append(palm)
    hs.prev_zs.append(z)
    hs.circle_buf.append(palm)

   
    if len(hs.prev_pts) >= 2:
        (x1,y1) = hs.prev_pts[-1]; (x0,y0) = hs.prev_pts[-2]
        speed = math.hypot(x1-x0, y1-y0) / max(dt,1e-3)  
    else:
        speed = 0.0

    
    if speed < MOTION_STEADY_PX_S:
        hs.charging_r = min(ORB_MAX_R, hs.charging_r + CHARGE_GROW_S * dt)
    else:
        hs.charging_r = max(ORB_MIN_R, hs.charging_r - CHARGE_SHRINK_S * dt)

  
    if len(hs.prev_zs) >= 2 and (now - hs.last_push_time) > PUSH_COOLDOWN_S:
        z0 = hs.prev_zs[-2]; z1 = hs.prev_zs[-1]
        dz = z0 - z1  
        if dz > PUSH_DZ_THRESH and hs.charging_r > ORB_MIN_R + 4:
            
            if len(hs.prev_pts) >= 2:
                (x1,y1) = hs.prev_pts[-1]; (x0,y0) = hs.prev_pts[-2]
                vel = np.array([(x1-x0)/max(dt,1e-3), (y1-y0)/max(dt,1e-3)], dtype=float) * VEL_SCALE
            else:
                vel = np.array([0.0,0.0], dtype=float)
            orbs.append(Orb(pos=np.array([palm[0], palm[1]], dtype=float),
                            vel=vel, r=hs.charging_r, color=orb_col))
            burst_particles(particles, palm, orb_col)
            hs.charging_r = BASE_ORB_R 
            hs.last_push_time = now

if __name__ == "__main__":
    main()
