import cv2
import time
import math
import numpy as np
import threading
import io
import wave
import sys
import os


BACKEND = "none"
sa = None
winsound = None
playsound = None

try:
    import simpleaudio as sa
    BACKEND = "simpleaudio"
except Exception:
    try:
        import winsound
        BACKEND = "winsound"
    except Exception:
        try:
            from playsound import playsound
            BACKEND = "playsound"
        except Exception:
            BACKEND = "none"

# Audio helpers
def make_click(sample_rate=44100, freq=2000, ms=40, volume=0.8):
    t = np.linspace(0, ms / 1000.0, int(sample_rate * ms / 1000.0), False)
    wave_env = np.sin(2 * np.pi * freq * t) * np.hanning(len(t))
    audio = (wave_env * volume * 32767).astype(np.int16)
    return audio, sample_rate

def _wav_bytes_from_i16(buf_i16: np.ndarray, sr: int) -> bytes:
    bio = io.BytesIO()
    with wave.open(bio, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(buf_i16.tobytes())
    return bio.getvalue()

def _play_with_playsound(buf_i16: np.ndarray, sr: int):
    import tempfile
    path = None
    try:
        fd, path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        with wave.open(path, "wb") as wf:
            wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(sr)
            wf.writeframes(buf_i16.tobytes())
        playsound(path)
    finally:
        if path:
            try: os.remove(path)
            except: pass

def play_click(buf_i16: np.ndarray, sample_rate: int):
    if BACKEND == "simpleaudio":
        sa.play_buffer(buf_i16, 1, 2, sample_rate)
        return

    if BACKEND == "winsound" and winsound is not None:
        data = _wav_bytes_from_i16(buf_i16, sample_rate)
        threading.Thread(
            target=winsound.PlaySound,
            args=(data, winsound.SND_MEMORY),
            daemon=True
        ).start()
        return

    if BACKEND == "playsound" and playsound is not None:
        threading.Thread(target=_play_with_playsound, args=(buf_i16, sample_rate), daemon=True).start()
        return

    #  simple Beep
    if winsound is not None:
        winsound.Beep(880, 60)

def clamp(v, a, b): return max(a, min(b, v))
def lerp(a, b, t): return a + (b - a) * t

# Hand tracking 
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands


def pinch_distance(landmarks, idx_a, idx_b):
    ax, ay = landmarks[idx_a].x, landmarks[idx_a].y
    bx, by = landmarks[idx_b].x, landmarks[idx_b].y
    return math.hypot(ax - bx, ay - by)

def is_pinch(landmarks, thresh=0.06):
    
    return pinch_distance(landmarks, 4, 8) < thresh

def handedness_label(handedness_obj):
    return handedness_obj.classification[0].label 

def get_hand_center_y(landmarks):
  
    y = (landmarks[0].y + landmarks[5].y) / 2.0
    return clamp(y, 0.0, 1.0)

def y_to_bpm(y_norm):     
    return int(lerp(180, 40, clamp(y_norm, 0, 1)))

def y_to_volume(y_norm):  
    return float(lerp(1.0, 0.0, clamp(y_norm, 0, 1)))

# ---------- Main ----------
def main():
    print(f"[Audio] Backend: {BACKEND}")
    if BACKEND == "none":
        print("No audio backend available. Install 'simpleaudio' or 'playsound'.")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Camera not found.")
        return

  
    normal_click, sr = make_click(freq=1650, ms=32, volume=0.8)
    accent_click, _ = make_click(freq=2200, ms=44, volume=1.0)


    playing = False
    bpm = 100
    volume = 0.7
    time_sig_beats = 4
    beat_counter = 0
    next_tick = time.time() + 1e9 

    right_pinch_prev = False
    left_pinch_prev = False
    pinch_cooldown = 0.25
    last_right_toggle = 0.0
    last_left_toggle = 0.0


    bpm_smoothed = bpm
    vol_smoothed = volume

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        model_complexity=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    ) as hands:

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)  
            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = hands.process(rgb)

            right_y = left_y = None
            right_is_pinch = left_is_pinch = False

            if res.multi_hand_landmarks and res.multi_handedness:
                for landmarks, handed in zip(res.multi_hand_landmarks, res.multi_handedness):
                    label = handedness_label(handed)
                    mp_drawing.draw_landmarks(
                        frame, landmarks, mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(200, 255, 200), thickness=2, circle_radius=2),
                        mp_drawing.DrawingSpec(color=(120, 200, 255), thickness=2, circle_radius=2)
                    )
                    if label == "Right":
                        right_y = get_hand_center_y(landmarks.landmark)
                        right_is_pinch = is_pinch(landmarks.landmark)
                    else:
                        left_y = get_hand_center_y(landmarks.landmark)
                        left_is_pinch = is_pinch(landmarks.landmark)

         
            if right_y is not None:
                target_bpm = y_to_bpm(right_y)
                bpm_smoothed = 0.85 * bpm_smoothed + 0.15 * target_bpm
                bpm = int(bpm_smoothed)

            if left_y is not None:
                target_vol = y_to_volume(left_y)
                vol_smoothed = 0.8 * vol_smoothed + 0.2 * target_vol
                volume = clamp(vol_smoothed, 0.0, 1.0)

            now = time.time()
            if right_is_pinch and not right_pinch_prev and (now - last_right_toggle) > pinch_cooldown:
                playing = not playing
                last_right_toggle = now
                if playing:
                    interval = 60.0 / max(40, bpm)
                    next_tick = now + 0.25  # small lead-in
                    beat_counter = 0
                else:
                    next_tick = now + 1e9
            right_pinch_prev = right_is_pinch

            if left_is_pinch and not left_pinch_prev and (now - last_left_toggle) > pinch_cooldown:
                time_sig_beats = 3 if time_sig_beats == 4 else 4
                last_left_toggle = now
            left_pinch_prev = left_is_pinch

            if playing:
                interval = 60.0 / max(40, bpm)
                if now >= next_tick:
                    is_accent = (beat_counter % time_sig_beats == 0)
                    click = accent_click if is_accent else normal_click

                    click_scaled = (click.astype(np.float32) * volume).clip(-32767, 32767).astype(np.int16)
                    play_click(click_scaled, sr)

                    beat_counter += 1
                    next_tick += interval

            # HUD
            cv2.rectangle(frame, (10, 10), (360, 160), (16, 34, 70), -1)
            cv2.putText(frame, f"AIR MUSIC CONDUCTOR  |  Audio: {BACKEND}", (18, 34),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 245, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, f"State: {'PLAYING' if playing else 'STOPPED'}", (18, 62),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 180) if playing else (180, 180, 180), 2, cv2.LINE_AA)
            cv2.putText(frame, f"BPM (Right hand): {bpm:3d}", (18, 88),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 230, 160), 2, cv2.LINE_AA)
            cv2.putText(frame, f"Volume (Left hand): {int(volume*100):3d}%", (18, 114),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 220, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, f"Time Sig: {time_sig_beats}/4 (Left pinch toggles)", (18, 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (170, 210, 255), 1, cv2.LINE_AA)

            cv2.putText(frame, "Right pinch=Play/Stop | Left pinch=3/4↔4/4 | T=test sound | Q=quit",
                        (18, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 240, 255), 2, cv2.LINE_AA)

            cv2.imshow("Air Music Conductor (Python)", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key in (ord('t'), ord('T')):  # test sound immediately
                test = (normal_click.astype(np.float32) * 0.9).astype(np.int16)
                play_click(test, sr)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
