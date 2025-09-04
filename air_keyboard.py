import cv2
import time
import numpy as np
import webbrowser
from dataclasses import dataclass
from typing import Tuple, List

import mediapipe as mp
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

WINDOW_TITLE = "Air Keyboard (Google / YouTube / Instagram)"

# --- styling ---
KEY_ALPHA = 0.35
TARGET_ALPHA = 0.45
HOVER_ALPHA_BOOST = 0.25
SELECTED_BORDER = (0, 170, 255)
AUTO_HIDE = True        
AUTO_HIDE_ZONE = 0.55  

@dataclass
class Button:
    label: str
    x: int
    y: int
    w: int
    h: int
    kind: str = "key"

    def rect(self) -> Tuple[int, int, int, int]:
        return (self.x, self.y, self.w, self.h)

    def contains(self, px: int, py: int) -> bool:
        return self.x <= px <= self.x + self.w and self.y <= py <= self.y + self.h

def draw_transparent_rect(dst, x, y, w, h, color, alpha):
    overlay = dst.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1, lineType=cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, dst, 1 - alpha, 0, dst)

def draw_button(img, btn: Button, hovered=False, selected=False):
    x, y, w, h = btn.rect()
    base = (40, 40, 40) if btn.kind == "key" else (35, 60, 90)
    alpha = TARGET_ALPHA if btn.kind == "target" else KEY_ALPHA
    if hovered:
        alpha = min(1.0, alpha + HOVER_ALPHA_BOOST)

    draw_transparent_rect(img, x, y, w, h, base, alpha)
    border_col = SELECTED_BORDER if selected else (220, 220, 220)
    cv2.rectangle(img, (x, y), (x + w, y + h), border_col, 1, lineType=cv2.LINE_AA)

    base_scale = max(0.45, min(0.95, h / 55.0))
    if len(btn.label) <= 2:
        font_scale = base_scale
    elif len(btn.label) <= 4:
        font_scale = base_scale * 0.88
    else:
        font_scale = base_scale * 0.80

    (tw, th), _ = cv2.getTextSize(btn.label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
    tx = x + (w - tw) // 2
    ty = y + (h + th) // 2
    cv2.putText(img, btn.label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA)

def distance(p1, p2):
    return np.hypot(p1[0] - p2[0], p1[1] - p2[1])

# ----layout--------
def build_targets(w: int, h: int) -> List[Button]:
    margin = int(0.03 * w)
    t_h = max(34, int(0.055 * h))
    t_w = max(96, int(0.105 * w))
    gap = max(8, int(0.015 * w))
    y = int(0.02 * h)
    x = margin
    return [
        Button("Google", x, y, t_w, t_h, kind="target"),
        Button("YouTube", x + (t_w + gap), y, t_w, t_h, kind="target"),
        Button("Instagram", x + 2 * (t_w + gap), y, int(t_w * 1.1), t_h, kind="target"),
    ]

def build_keyboard(w: int, h: int) -> List[Button]:
    margin_x = int(0.03 * w)
    gap = max(5, int(0.008 * w))

    kb_top = int(0.68 * h)  
    avail_w = w - 2 * margin_x
    avail_h = max(int(0.20 * h), h - kb_top - int(0.02 * h)) 

    rows_total, gaps_total = 4, 3
    key_h = max(22, min(52, (avail_h - gaps_total * gap) // rows_total))
    max_cols = 10
    key_w = max(22, min(64, (avail_w - (max_cols - 1) * gap) // max_cols))

    start_x = margin_x
    y = kb_top

    def add_row(chars: List[str], start_x_row: int, y_row: int, buttons: List[Button]):
        x = start_x_row
        for ch in chars:
            buttons.append(Button(ch, x, y_row, key_w, key_h))
            x += key_w + gap

    buttons: List[Button] = []
    add_row(list("QWERTYUIOP"), start_x, y, buttons)
    y += key_h + gap
    add_row(list("ASDFGHJKL"), start_x + key_w // 2, y, buttons)
    y += key_h + gap
    add_row(list("ZXCVBNM"), start_x + key_w, y, buttons)

    y_special = y + key_h + gap
    buttons.append(Button("BACK", start_x, y, key_w, key_h))
    space_w = key_w * 5 + gap * 4  # a bit narrower
    buttons.append(Button("SPACE", start_x, y_special, space_w, key_h))
    clr_x = start_x + space_w + gap
    clr_w = key_w * 2 + gap
    buttons.append(Button("CLR", clr_x, y_special, clr_w, key_h))
    enter_x = clr_x + clr_w + gap
    enter_w = key_w * 2 + gap
    buttons.append(Button("ENTER", enter_x, y_special, enter_w, key_h))
    return buttons

def open_target(target: str, query: str):
    q = query.strip()
    if not q:
        return
    if target == "Google":
        webbrowser.open(f"https://www.google.com/search?q={q}")
    elif target == "YouTube":
        webbrowser.open(f"https://www.youtube.com/results?search_query={q}")
    elif target == "Instagram":
        webbrowser.open(f"https://www.instagram.com/explore/tags/{q.replace(' ','')}/")
    else:
        webbrowser.open(f"https://www.google.com/search?q={q}")

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Camera not found.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_TITLE, 1280, 720)

    selected_target = "Google"
    typed_text = ""
    last_click_time = 0.0
    click_cooldown = 0.35

    hovered_key_idx = -1
    hovered_target_idx = -1

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

           
            clean_for_detection = frame.copy()  
            rgb = cv2.cvtColor(clean_for_detection, cv2.COLOR_BGR2RGB)
            res = hands.process(rgb)

           
            pinch_now = False
            index_tip = None
            thumb_tip = None

            if res.multi_hand_landmarks:
                for handLms in res.multi_hand_landmarks:
                    lm = handLms.landmark
                    ix, iy = int(lm[8].x * w), int(lm[8].y * h)
                    tx, ty = int(lm[4].x * w), int(lm[4].y * h)
                    index_tip = (ix, iy)
                    thumb_tip = (tx, ty)
                    d = distance(index_tip, thumb_tip)
                    pinch_threshold = max(22, int(0.045 * h))
                    pinch_now = d < pinch_threshold

            
            target_buttons = build_targets(w, h)
            keyboard_buttons = build_keyboard(w, h)

            margin = int(0.03 * w)
            query_bar_top = int(0.12 * h)
            query_bar_bot = int(0.20 * h)
            bar_h = query_bar_bot - query_bar_top

            
            show_keyboard = True
            if AUTO_HIDE:
                if index_tip is None or (index_tip[1] / h) < AUTO_HIDE_ZONE:
                    show_keyboard = False

            
            draw_transparent_rect(frame, margin - 20, int(0.095 * h),
                                  (w - margin + 20) - (margin - 20),
                                  int(0.23 * h) - int(0.095 * h),
                                  (25, 25, 25), 0.22)

            
            draw_transparent_rect(frame, margin + 60, query_bar_top,
                                  (w - margin) - (margin + 60),
                                  bar_h, (80, 80, 80), 0.33)
            cv2.rectangle(frame, (margin + 60, query_bar_top),
                          (w - margin, query_bar_bot), (200, 200, 200), 1, cv2.LINE_AA)
            cv2.putText(frame, "Query:", (margin - 10, int(0.11 * h)),
                        cv2.FONT_HERSHEY_SIMPLEX, max(0.5, h/900), (240, 240, 240), 1, cv2.LINE_AA)

            
            hovered_target_idx = -1
            for btn in target_buttons:
                draw_button(frame, btn, hovered=False, selected=(btn.label == selected_target))

           
            hovered_key_idx = -1
            if show_keyboard:
                for btn in keyboard_buttons:
                    draw_button(frame, btn, hovered=False)

           
            fscale = max(0.6, min(1.1, bar_h / 45.0))
            shown_text = typed_text[-80:]
            cv2.putText(frame, shown_text, (margin + 70, query_bar_bot - int(bar_h * 0.28)),
                        cv2.FONT_HERSHEY_SIMPLEX, fscale, (255, 255, 255), 2, cv2.LINE_AA)

            
            if res.multi_hand_landmarks:
                for handLms in res.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, handLms, mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(120,120,120), thickness=1, circle_radius=2),
                        mp_drawing.DrawingSpec(color=(80,80,80), thickness=1)
                    )

            
            if index_tip:
               
                for i, btn in enumerate(target_buttons):
                    if btn.contains(*index_tip):
                        hovered_target_idx = i
                        draw_button(frame, btn, hovered=True, selected=(btn.label == selected_target))
                    elif btn.label == selected_target:
                        draw_button(frame, btn, hovered=False, selected=True)

                
                if show_keyboard:
                    for i, btn in enumerate(keyboard_buttons):
                        if btn.contains(*index_tip):
                            hovered_key_idx = i
                            draw_button(frame, btn, hovered=True)

            
            if index_tip and thumb_tip:
                cv2.line(frame, index_tip, thumb_tip, (0, 255, 255), 2, cv2.LINE_AA)

            
            now = time.time()
            if pinch_now and (now - last_click_time) > 0.35:
                if hovered_target_idx != -1:
                    selected_target = target_buttons[hovered_target_idx].label
                    last_click_time = now
                elif hovered_key_idx != -1 and show_keyboard:
                    key = keyboard_buttons[hovered_key_idx].label
                    if key == "BACK":
                        typed_text = typed_text[:-1]
                    elif key == "SPACE":
                        typed_text += " "
                    elif key == "CLR":
                        typed_text = ""
                    elif key == "ENTER":
                        open_target(selected_target, typed_text)
                    else:
                        typed_text += key
                    last_click_time = now

            hud_scale = max(0.5, h/1000)
            cv2.putText(frame, f"Target: {selected_target}", (int(0.35*w), int(0.09*h)),
                        cv2.FONT_HERSHEY_SIMPLEX, hud_scale, (255,255,255), 2, cv2.LINE_AA)
            cv2.putText(frame, "Pinch index+thumb to click | Press 'q' to quit",
                        (int(0.03*w), h-20), cv2.FONT_HERSHEY_SIMPLEX, hud_scale, (230,230,230), 1, cv2.LINE_AA)

            cv2.imshow(WINDOW_TITLE, frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
