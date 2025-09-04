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

TITLE = "Air Pet Interaction"


PINCH_FRAC           = 0.045       
PINCH_HOLD_TREAT_S   = 0.25
PINCH_V_THROW_PX_S   = 900.0      
NEAR_PET_PX          = 140         
PET_MAX_SPEED        = 520.0       
PET_ACCEL            = 1200.0     
PET_RADIUS           = 56
HAPPINESS_DECAY_S    = 0.006
HUNGER_INCREASE_S    = 0.008
TREAT_FEED           = 32
HAPPY_TREAT_BONUS    = 22
HAPPY_PET_BONUS      = 6
TOY_HAPPY_BONUS      = 18
TOY_DRAG             = 0.15
HEART_LIFE           = 1.0
SLOW_DT_MAX          = 0.05

THEMES = [
    dict( name="Sky Pup",
          fur=(180,210,255), fur_dark=(140,170,225), ear_in=(205,190,255), belly=(245,246,252),
          blush=(190,170,240), line=(80,90,120), accent=(255,255,255) ),
    dict( name="Peach Cat",
          fur=(180,180,220), fur_dark=(150,150,200), ear_in=(180,180,255), belly=(250,240,230),
          blush=(150,160,255), line=(70,70,110), accent=(255,255,255) ),
    dict( name="Mint Bun",
          fur=(200,245,220), fur_dark=(160,210,190), ear_in=(220,230,240), belly=(245,250,246),
          blush=(180,200,255), line=(70,100,90), accent=(255,255,255) ),
    dict( name="Sun Fox",
          fur=(120,190,255), fur_dark=(100,160,220), ear_in=(190,220,255), belly=(250,245,235),
          blush=(140,170,255), line=(70,80,110), accent=(255,255,255) ),
    dict( name="Grape Pup",
          fur=(220,190,255), fur_dark=(190,160,230), ear_in=(235,210,255), belly=(248,246,252),
          blush=(190,170,240), line=(90,70,120), accent=(255,255,255) ),
]

@dataclass
class HandState:
    label: str
    pinch: bool = False
    was_pinch: bool = False
    pinch_pt: Tuple[int,int] = (0,0)
    pinch_start_t: float = 0.0
    prev_pts: deque = field(default_factory=lambda: deque(maxlen=6))

@dataclass
class Treat:
    pos: np.ndarray
    life: float = 6.0

@dataclass
class Toy:
    pos: np.ndarray
    vel: np.ndarray
    life: float = 8.0

@dataclass
class Heart:
    pos: np.ndarray
    vel: np.ndarray
    life: float = HEART_LIFE
    color: Tuple[int,int,int] = (255,180,210)

@dataclass
class Pet:
    pos: np.ndarray
    vel: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=float))
    happiness: float = 65.0
    hunger: float = 35.0
    target: Optional[np.ndarray] = None
    skin: int = 0
    wag_phase: float = 0.0

    def want_target(self, treats: List[Treat], toys: List[Toy]):
        if treats:
            i = int(np.argmin([np.linalg.norm(t.pos - self.pos) for t in treats]))
            self.target = treats[i].pos.copy(); return
        if toys:
            i = int(np.argmin([np.linalg.norm(t.pos - self.pos) for t in toys]))
            self.target = toys[i].pos.copy(); return
        self.target = None

    def update(self, dt: float, w: int, h: int):
        self.happiness = max(0.0, min(100.0, self.happiness - HAPPINESS_DECAY_S*dt*100))
        self.hunger    = max(0.0, min(100.0, self.hunger + HUNGER_INCREASE_S*dt*100))

        if self.target is not None:
            dirv = self.target - self.pos
            dist = float(np.linalg.norm(dirv)) + 1e-6
            dirv /= dist
            self.vel += dirv * PET_ACCEL * dt
        else:
            self.vel *= (1.0 - 0.9*dt)

        sp = float(np.linalg.norm(self.vel))
        if sp > PET_MAX_SPEED:
            self.vel *= (PET_MAX_SPEED / sp)

        self.pos += self.vel * dt
        r = PET_RADIUS
        self.pos[0] = float(np.clip(self.pos[0], r, w - r))
        self.pos[1] = float(np.clip(self.pos[1], r, h - r))
        self.wag_phase += dt * (6.0 + 6.0*(self.happiness/100.0))

    def draw(self, img, t: float, look_at: Optional[Tuple[int,int]] = None):
        th = THEMES[self.skin]
        fur, fur_dark, ear_in, belly = th["fur"], th["fur_dark"], th["ear_in"], th["belly"]
        blush, line, accent = th["blush"], th["line"], th["accent"]

        x, y = int(self.pos[0]), int(self.pos[1])
        r = PET_RADIUS
        overlay = img.copy()
        cv2.ellipse(overlay, (x, y + r + 10), (int(r*0.9), int(r*0.32)), 0, 0, 360, (30,30,30), -1, cv2.LINE_AA)
        img[:] = cv2.addWeighted(overlay, 0.28, img, 0.72, 0)
        tail_ang = math.sin(self.wag_phase) * 28  
        tail_len = int(r*0.9)
        tx = int(x - r*0.85)
        ty = int(y + r*0.05)
        for i in range(3):
            a = tail_ang * (0.6 + 0.2*i)
            ax, ay = int(tx - i*tail_len*0.26), int(ty + (i-1)*4)
            cv2.ellipse(img, (ax, ay), (int(r*0.35*(1-i*0.12)), int(r*0.22*(1-i*0.1))),
                        a, 0, 360, fur_dark, -1, cv2.LINE_AA)
            cv2.ellipse(img, (ax, ay), (int(r*0.35*(1-i*0.12)), int(r*0.22*(1-i*0.1))),
                        a, 0, 360, line, 1, cv2.LINE_AA)


        cv2.circle(img, (x, y), r, fur, -1, cv2.LINE_AA)
        cv2.circle(img, (x, y), int(r*0.92), fur_dark, -1, cv2.LINE_AA)
        cv2.circle(img, (x-8, y-10), int(r*0.85), fur, -1, cv2.LINE_AA)
        cv2.circle(img, (x-12, y-18), int(r*0.42), accent, -1, cv2.LINE_AA)
        overlay = img.copy()
        cv2.circle(overlay, (x-12, y-18), int(r*0.42), accent, -1, cv2.LINE_AA)
        img[:] = cv2.addWeighted(overlay, 0.12, img, 0.88, 0)
        cv2.ellipse(img, (x, y + int(r*0.18)), (int(r*0.66), int(r*0.54)), 0, 0, 360, belly, -1, cv2.LINE_AA)


        def ear(px, py, flip):
            outer = np.array([
                (px, py),
                (px - 22*flip, py + 30),
                (px + 20*flip, py + 34)
            ], np.int32)
            cv2.fillConvexPoly(img, outer, fur_dark, lineType=cv2.LINE_AA)
            inner = np.array([
                (px, py+6),
                (px - 14*flip, py + 28),
                (px + 12*flip, py + 26)
            ], np.int32)
            cv2.fillConvexPoly(img, inner, ear_in, lineType=cv2.LINE_AA)
            cv2.polylines(img, [outer], True, line, 2, cv2.LINE_AA)
        ear(x-24, y-40, -1)
        ear(x+24, y-40,  1)

        cv2.ellipse(img, (x, y+10), (int(r*0.55), int(r*0.40)), 0, 0, 360, belly, -1, cv2.LINE_AA)
        cv2.ellipse(img, (x, y+8), (8,5), 0, 0, 360, (40,40,60), -1, cv2.LINE_AA)


        lx, ly = x, y-6
        dirx, diry = 0.0, -0.25
        if look_at is not None:
            vx, vy = (look_at[0]-x, look_at[1]-y)
            d = max(1.0, math.hypot(vx, vy))
            dirx, diry = (vx/d, vy/d)
        eye_offset = int(r*0.35)
        pupil_r = 7

        cv2.circle(img, (x-eye_offset, y-6), 13, (250,250,250), -1, cv2.LINE_AA)
        cv2.circle(img, (x+eye_offset, y-6), 13, (250,250,250), -1, cv2.LINE_AA)
        prx = int(dirx*6); pry = int(diry*6)
        cv2.circle(img, (x-eye_offset+prx, y-6+pry), pupil_r, (20,20,40), -1, cv2.LINE_AA)
        cv2.circle(img, (x+eye_offset+prx, y-6+pry), pupil_r, (20,20,40), -1, cv2.LINE_AA)
        if (t % 2.6) < 0.09:
            cv2.line(img, (x-eye_offset-9, y-6), (x-eye_offset+9, y-6), line, 3, cv2.LINE_AA)
            cv2.line(img, (x+eye_offset-9, y-6), (x+eye_offset+9, y-6), line, 3, cv2.LINE_AA)


        arc = int(np.interp(self.happiness, [0,100], [8, 18]))
        cv2.ellipse(img, (x, y+20), (18, 12), 0, 200-arc, 340+arc, (50,50,70), 3, cv2.LINE_AA)


        for s in (-1, 1):
            cv2.line(img, (x+int(16*s), y+14), (x+int((16+22)*s), y+10), line, 2, cv2.LINE_AA)
            cv2.line(img, (x+int(16*s), y+18), (x+int((16+24)*s), y+18), line, 2, cv2.LINE_AA)
            cv2.line(img, (x+int(16*s), y+22), (x+int((16+22)*s), y+26), line, 2, cv2.LINE_AA)


        b_alpha = max(0.0, min(1.0, (self.happiness-30)/70))
        if b_alpha > 0:
            overlay = img.copy()
            cv2.circle(overlay, (x-22, y+20), 10, blush, -1, cv2.LINE_AA)
            cv2.circle(overlay, (x+22, y+20), 10, blush, -1, cv2.LINE_AA)
            img[:] = cv2.addWeighted(overlay, 0.25*b_alpha, img, 1-0.25*b_alpha, 0)

 
        for px in (-24, 24):
            cv2.ellipse(img, (x+px, y+r-8), (12,10), 0, 0, 360, belly, -1, cv2.LINE_AA)
            cv2.circle(img, (x+px-6, y+r-12), 3, (230,200,200), -1, cv2.LINE_AA)
            cv2.circle(img, (x+px,   y+r-13), 3, (230,200,200), -1, cv2.LINE_AA)
            cv2.circle(img, (x+px+6, y+r-12), 3, (230,200,200), -1, cv2.LINE_AA)

        
        cv2.circle(img, (x, y), r, line, 2, cv2.LINE_AA)

def pinch_info(hand_lms, w, h):
    lm = hand_lms.landmark
    ix, iy = int(lm[8].x*w), int(lm[8].y*h)
    tx, ty = int(lm[4].x*w), int(lm[4].y*h)
    d = math.hypot(ix-tx, iy-ty)
    return d, ((ix+tx)//2, (iy+ty)//2)

def speed_from_pts(pts: deque, dt: float) -> float:
    if len(pts) < 2: return 0.0
    (x1,y1) = pts[-1]; (x0,y0) = pts[-2]
    return math.hypot(x1-x0, y1-y0) / max(1e-3, dt)

def add_hearts(hearts: List[Heart], center: Tuple[int,int], n=6):
    x, y = center
    for _ in range(n):
        ang = random.uniform(-math.pi, 0)
        spd = random.uniform(120, 260)
        vx, vy = spd*math.cos(ang), spd*math.sin(ang) - 40
        hearts.append(Heart(pos=np.array([x,y], float), vel=np.array([vx,vy], float)))

def draw_hearts(img, hearts: List[Heart], dt: float):
    for h in list(hearts):
        h.life -= dt; h.pos += h.vel * dt; h.vel[1] -= 240*dt
        if h.life <= 0: hearts.remove(h)
    for h in hearts:
        a = max(0.0, min(1.0, h.life / HEART_LIFE))
        col = tuple(int(a*c + (1-a)*60) for c in h.color)
        cv2.circle(img, (int(h.pos[0]), int(h.pos[1])), 6, col, -1, cv2.LINE_AA)


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Camera not found."); return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    cv2.namedWindow(TITLE, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(TITLE, 1280, 720)

    pet = Pet(pos=np.array([640.0, 420.0], float))
    treats: List[Treat] = []
    toys: List[Toy] = []
    hearts: List[Heart] = []

    hands_state = {"Left": HandState(label="Left"), "Right": HandState(label="Right")}
    debug = False
    prev = time.time()

    with mp_hands.Hands(static_image_mode=False,
                        max_num_hands=2,
                        min_detection_confidence=0.6,
                        min_tracking_confidence=0.6) as hands:
        while True:
            ok, frame = cap.read()
            if not ok: break
            frame = cv2.flip(frame, 1); h, w = frame.shape[:2]

            now = time.time()
            dt = max(1e-3, min(SLOW_DT_MAX, now - prev)); prev = now
            res = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            for hs in hands_state.values(): hs.pinch = False
            pinch_thresh = max(22, int(PINCH_FRAC*h))

            if res.multi_hand_landmarks:
                for lms in res.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, lms, mp_hands.HAND_CONNECTIONS,
                                              mp_drawing.DrawingSpec(color=(180,240,180), thickness=1, circle_radius=2),
                                              mp_drawing.DrawingSpec(color=(130,190,255), thickness=1))
               
                if res.multi_handedness and len(res.multi_hand_landmarks)==len(res.multi_handedness):
                    for lms, hd in zip(res.multi_hand_landmarks, res.multi_handedness):
                        hs = hands_state[hd.classification[0].label]
                        d, pt = pinch_info(lms, w, h)
                        hs.pinch = d < pinch_thresh; hs.pinch_pt = pt; hs.prev_pts.append(pt)
                else:
                    lmss = res.multi_hand_landmarks
                    xs = [lm.landmark[0].x for lm in lmss]
                    order = np.argsort(xs)
                    labels = ["Left","Right"] if len(lmss)==2 else ["Right"]
                    mapping = {int(order[i]): labels[i] for i in range(len(labels))}
                    for i,lms in enumerate(lmss):
                        hs = hands_state[mapping.get(i,"Right")]
                        d, pt = pinch_info(lms, w, h)
                        hs.pinch = d < pinch_thresh; hs.pinch_pt = pt; hs.prev_pts.append(pt)

            for hs in hands_state.values():
                spd = speed_from_pts(hs.prev_pts, dt)
                if hs.pinch and not hs.was_pinch:
                    hs.pinch_start_t = now
                if (not hs.pinch) and hs.was_pinch:
                    hold = now - hs.pinch_start_t
                    pos = np.array(hs.pinch_pt, dtype=float)
                    if hold >= PINCH_HOLD_TREAT_S and spd < PINCH_V_THROW_PX_S:
                        treats.append(Treat(pos=pos)); add_hearts(hearts, (int(pos[0]), int(pos[1])), n=4)
                    else:
                        if len(hs.prev_pts)>=2:
                            (x1,y1) = hs.prev_pts[-1]; (x0,y0) = hs.prev_pts[-2]
                            vel = np.array([(x1-x0)/max(dt,1e-3), (y1-y0)/max(dt,1e-3)], float)*0.85
                        else:
                            vel = np.array([0.0,0.0], float)
                        toys.append(Toy(pos=pos.copy(), vel=vel))
                hs.was_pinch = hs.pinch

        
            for hs in hands_state.values():
                if not hs.pinch and hs.prev_pts:
                    xh,yh = hs.prev_pts[-1]
                    if math.hypot(xh-pet.pos[0], yh-pet.pos[1]) < NEAR_PET_PX and len(hs.prev_pts)>=2:
                        x1,_ = hs.prev_pts[-1]; x0,_ = hs.prev_pts[-2]
                        vx = (x1-x0)/max(dt,1e-3)
                        if abs(vx) > 400:
                            pet.happiness = min(100.0, pet.happiness + HAPPY_PET_BONUS)
                            add_hearts(hearts, (int(pet.pos[0]), int(pet.pos[1]-PET_RADIUS-8)), n=3)

     
            for toy in toys:
                toy.vel *= (1.0 - TOY_DRAG*dt); toy.pos += toy.vel*dt; toy.life -= dt
            toys[:] = [t for t in toys if t.life > 0]
            for ttr in treats: ttr.life -= dt
            treats[:] = [t for t in treats if t.life > 0]

          
            pet.want_target(treats, toys)
            pet.update(dt, w, h)

            if treats:
                i = int(np.argmin([np.linalg.norm(t.pos - pet.pos) for t in treats]))
                if np.linalg.norm(treats[i].pos - pet.pos) <= PET_RADIUS*1.2:
                    pet.hunger = max(0.0, pet.hunger - TREAT_FEED)
                    pet.happiness = min(100.0, pet.happiness + HAPPY_TREAT_BONUS)
                    add_hearts(hearts, (int(pet.pos[0]), int(pet.pos[1]-PET_RADIUS)), n=6)
                    treats.pop(i)
            if toys:
                i = int(np.argmin([np.linalg.norm(t.pos - pet.pos) for t in toys]))
                if np.linalg.norm(toys[i].pos - pet.pos) <= PET_RADIUS*1.25:
                    pet.happiness = min(100.0, pet.happiness + TOY_HAPPY_BONUS)
                    add_hearts(hearts, (int(pet.pos[0]), int(pet.pos[1]-PET_RADIUS)), n=5)
                    toys.pop(i)

      
            canvas = frame

            
            for toy in toys:
                p = (int(toy.pos[0]), int(toy.pos[1]))
                cv2.circle(canvas, p, 10, (90,190,255), -1, cv2.LINE_AA)
                cv2.circle(canvas, p, 10, (255,255,255), 1, cv2.LINE_AA)
            
            for tr in treats:
                p = (int(tr.pos[0]), int(tr.pos[1]))
                cv2.circle(canvas, p, 8, (60,220,120), -1, cv2.LINE_AA)
                cv2.circle(canvas, p, 8, (255,255,255), 1, cv2.LINE_AA)

           
            focus = None; best = 1e9
            for hs in hands_state.values():
                if hs.prev_pts:
                    d = math.hypot(hs.prev_pts[-1][0]-pet.pos[0], hs.prev_pts[-1][1]-pet.pos[1])
                    if d < best: best = d; focus = hs.prev_pts[-1]

            pet.draw(canvas, now, look_at=focus)
            draw_hearts(canvas, hearts, dt)

            
            theme_name = THEMES[pet.skin]["name"]
            cv2.rectangle(canvas, (10,10), (360,98), (25,25,25), -1)
            cv2.putText(canvas, f"AIR PET  |  Skin: {theme_name}", (18,32), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2, cv2.LINE_AA)
            
            cv2.putText(canvas, "Happiness", (18,56), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (230,230,230), 1, cv2.LINE_AA)
            cv2.rectangle(canvas, (120,44), (330,58), (60,60,60), -1)
            hw = int(np.clip(pet.happiness,0,100)*2.1)
            cv2.rectangle(canvas, (120,44), (120+hw,58), (100,220,255), -1)
            
            cv2.putText(canvas, "Hunger", (18,84), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (230,230,230), 1, cv2.LINE_AA)
            cv2.rectangle(canvas, (120,70), (330,84), (60,60,60), -1)
            hg = int(np.clip(pet.hunger,0,100)*2.1)
            cv2.rectangle(canvas, (120,70), (120+hg,84), (255,160,120), -1)

            cv2.putText(canvas, "Pinch hold=Treat  |  Pinch quick=Throw toy  |  1..5 skins  |  Q quit  R reset  C clear  D debug",
                        (18, h-24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230,230,230), 1, cv2.LINE_AA)

            if debug:
                for hs in hands_state.values():
                    if hs.prev_pts:
                        cv2.circle(canvas, hs.prev_pts[-1], 6, (255,255,0), 1, cv2.LINE_AA)

            cv2.imshow(TITLE, canvas)

            # INPUT
            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'):
                break
            elif k == ord('r'):
                pet.happiness, pet.hunger = 65.0, 35.0
                treats.clear(); toys.clear(); hearts.clear()
            elif k == ord('c'):
                treats.clear(); toys.clear()
            elif k == ord('d'):
                debug = not debug
            elif k in [ord('1'),ord('2'),ord('3'),ord('4'),ord('5')]:
                pet.skin = int(chr(k)) - 1

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
