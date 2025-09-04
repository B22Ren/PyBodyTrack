import cv2
import os
import sys
import time
import subprocess
from dataclasses import dataclass
from typing import Tuple, List
import math

WINDOW = "Air Lab — Home"

APPS: List[Tuple[str, str]] = [
    ("Air Keyboard",        "air_keyboard.py"),
    ("Air Shooting",     "Arrowshooting.py"),
    ("Air Painting 3D",     "painting.py"),      
    ("Air Energy Ball",     "energy_ball.py"),
    ("Air Music Conductor", "Dirijor.py"),           
    ("Air Pet (Cute)",      "pet_interaction.py"),
    ("Air Portal",          "portals.py")

]

@dataclass
class Card:
    label: str
    script: str
    x: int; y: int; w: int; h: int
    exists: bool

    def contains(self, px, py) -> bool:
        return self.x <= px <= self.x + self.w and self.y <= py <= self.y + self.h

def abs_path(rel: str) -> str:
    base = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(base, rel))

def build_cards(w=1280, h=720) -> List[Card]:
    cols = 3                                   
    rows = max(1, math.ceil(len(APPS) / cols)) 
    pad_x, pad_y = 36, 36
    top_h = 120                                
    y0 = top_h + pad_y

  
    avail_h = h - y0 - pad_y
   
    card_w = (w - pad_x*(cols+1)) // cols
    card_h = max(110, (avail_h - pad_y*(rows-1)) // rows)  

    cards: List[Card] = []
    i = 0
    for r in range(rows):
        for c in range(cols):
            if i >= len(APPS):
                break
            label, rel = APPS[i]
            p = abs_path(rel)
            exists = os.path.isfile(p)
            x = pad_x + c*(card_w + pad_x)
            y = y0 + r*(card_h + pad_y)
            cards.append(Card(label, p, x, y, card_w, card_h, exists))
            i += 1
    return cards


def draw_card(img, card: Card, hover=False):
    x,y,w,h = card.x, card.y, card.w, card.h
    base = (40, 60, 100) if card.exists else (50, 50, 50)
    hi   = (90, 160, 255) if hover and card.exists else base
    # bg
    overlay = img.copy()
    cv2.rectangle(overlay, (x, y), (x+w, y+h), hi, -1, cv2.LINE_AA)
    img[:] = cv2.addWeighted(overlay, 0.35, img, 0.65, 0)

    # border
    cv2.rectangle(img, (x, y), (x+w, y+h), (220,220,230) if card.exists else (110,110,110), 2, cv2.LINE_AA)

    # title
    fs = 0.75 if len(card.label) <= 14 else 0.6
    (tw, th), _ = cv2.getTextSize(card.label, cv2.FONT_HERSHEY_SIMPLEX, fs, 2)
    tx = x + (w - tw)//2
    ty = y + h//2 + th//2
    color = (255,255,255) if card.exists else (160,160,160)
    cv2.putText(img, card.label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, fs, color, 2, cv2.LINE_AA)

    # small footer
    if not card.exists:
        msg = "file not found"
        (tw, th), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.putText(img, msg, (x+(w-tw)//2, y+h-14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (130,130,130), 1, cv2.LINE_AA)

def top_bar(img, w):
    cv2.rectangle(img, (0,0), (w,120), (25,25,35), -1)
    cv2.putText(img, "AIR LAB — pick an experience", (24, 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (245,245,250), 2, cv2.LINE_AA)
    cv2.putText(img, "Click a card   •   Press 1..9   •   Q = Quit   •   After an app, press Q to come back here",
                (24, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (220,230,240), 1, cv2.LINE_AA)

def launch(script_path: str):
    try:
        cmd = [sys.executable, script_path]
        print(f"[Home] Launching: {cmd}")
        subprocess.run(cmd, check=False)
    except Exception as e:
        print(f"[Home] Launch error: {e}")
        err = f"Error running {os.path.basename(script_path)}"
        toast(err)

def toast(msg, ms=1400, w=1280, h=720):
    end = time.time() + ms/1000.0
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    while time.time() < end:
        img = 255 * (0.08 + 0.92*0) * (0*0)  
        blank = cv2.getWindowImageRect(WINDOW)  

        box = (int(w*0.5)-220, int(h*0.5)-40, 440, 80)
        overlay = (box[0], box[1], box[0]+box[2], box[1]+box[3])
        frame = np.zeros((h, w, 3), dtype='uint8')
        import numpy as np 
        cv2.rectangle(frame, (overlay[0], overlay[1]), (overlay[2], overlay[3]), (30,30,30), -1, cv2.LINE_AA)
        cv2.rectangle(frame, (overlay[0], overlay[1]), (overlay[2], overlay[3]), (220,220,220), 1, cv2.LINE_AA)
        (tw, th), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.putText(frame, msg, (int(w*0.5)-tw//2, int(h*0.5)+th//2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240,240,240), 1, cv2.LINE_AA)
        cv2.imshow(WINDOW, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

mouse_x = mouse_y = -1
mouse_clicked = False
def on_mouse(event, x, y, flags, param):
    global mouse_x, mouse_y, mouse_clicked
    mouse_x, mouse_y = x, y
    if event == cv2.EVENT_LBUTTONDOWN:
        mouse_clicked = True

def main():
    import numpy as np

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    w, h = 1280, 720
    cv2.resizeWindow(WINDOW, w, h)
    cv2.setMouseCallback(WINDOW, on_mouse)

    cards = build_cards(w, h)
    last_launch_time = 0.0

    while True:
        frame = np.zeros((h, w, 3), dtype='uint8')
        for i in range(h):
            a = i / h
            frame[i,:] = (int(20+80*a), int(30+70*a), int(60+110*a))

        top_bar(frame, w)

        hovered_idx = -1
        for i, card in enumerate(cards):
            hover = card.contains(mouse_x, mouse_y)
            if hover: hovered_idx = i
            draw_card(frame, card, hover=hover)

    
            cx, cy = card.x + 24, card.y + 28
            cv2.circle(frame, (cx, cy), 14, (245,245,245), 2, cv2.LINE_AA)
            cv2.putText(frame, str(i+1), (cx-6, cy+6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (245,245,245), 2, cv2.LINE_AA)

        cv2.imshow(WINDOW, frame)

        k = cv2.waitKey(16) & 0xFF
        now = time.time()

   
        if ord('1') <= k <= ord('9'):
            idx = k - ord('1')
            if 0 <= idx < len(cards) and cards[idx].exists and (now - last_launch_time) > 0.25:
                last_launch_time = now
                launch(cards[idx].script)


        global mouse_clicked
        if mouse_clicked:
            mouse_clicked = False
            if hovered_idx != -1:
                card = cards[hovered_idx]
                if card.exists and (now - last_launch_time) > 0.25:
                    last_launch_time = now
                    launch(card.script)

        if k in (ord('q'), 27):
            break

        if int(now) % 2 == 0: 
            cards = build_cards(w, h)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
