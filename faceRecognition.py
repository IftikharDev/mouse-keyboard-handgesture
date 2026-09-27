"""
AI Virtual Mouse & Keyboard Control via Hand Gestures
=====================================================
Control your computer hands-free using your webcam:
  - 🖱 VIRTUAL MOUSE:
      • Smooth cursor movement with index fingertip
      • Left Click: Quick pinch (Thumb + Index tip)
      • Drag & Drop: Hold pinch (Thumb + Index tip) > 0.3s
      • Right Click: Pinch Middle fingertip + Thumb tip
      • Scroll: Two fingers up (Index + Middle) moving up / down
  - ⌨ VIRTUAL KEYBOARD:
      • Transparent on-screen QWERTY keyboard overlay
      • Dwell-to-type: Hover index fingertip for 0.4s
      • Pinch-to-type: Pinch thumb & index on a key for instant typing
      • Special keys: Backspace, Enter, Space, CapsLock, Clear
      • Injects real keypresses into any active window
  - 🔄 MODE SWITCH:
      • Hover or pinch the on-screen mode button
      • Press 'm' for Mouse, 'k' for Keyboard, 'Tab' to toggle, 'q' to quit
"""

import os
import sys
import time
import math
import urllib.request
import cv2
import numpy as np

# Input simulation via pynput
try:
    from pynput.mouse import Controller as MouseController, Button
    from pynput.keyboard import Controller as KeyboardController, Key
except ImportError:
    print("Error: pynput is required. Install via: pip install pynput")
    sys.exit(1)

# MediaPipe modern Tasks Vision API
try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision
except ImportError:
    print("Error: mediapipe is required. Install via: pip install mediapipe")
    sys.exit(1)

# Detect Screen Resolution (with Xlib / Tk fallback)
def get_screen_resolution():
    try:
        import Xlib.display
        display = Xlib.display.Display()
        screen = display.screen()
        return screen.width_in_pixels, screen.height_in_pixels
    except Exception:
        pass
    try:
        import subprocess
        out = subprocess.check_output(['xrandr']).decode('utf-8')
        for line in out.splitlines():
            if '*' in line:
                res = line.split()[0]
                w, h = res.split('x')
                return int(w), int(h)
    except Exception:
        pass
    return 1920, 1080

SCREEN_WIDTH, SCREEN_HEIGHT = get_screen_resolution()
print(f"[*] Detected Screen Resolution: {SCREEN_WIDTH}x{SCREEN_HEIGHT}")

# Ensure MediaPipe Hand Landmarker model exists
MODEL_FILENAME = "hand_landmarker.task"
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), MODEL_FILENAME)
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

def ensure_model_exists():
    if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 1000000:
        print(f"[*] Downloading Hand Landmarker model (~7.5MB)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print(f"[✓] Model downloaded successfully to {MODEL_PATH}")

ensure_model_exists()

# Initialize Hand Landmarker
base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.6,
    min_hand_presence_confidence=0.6,
    min_tracking_confidence=0.6
)
landmarker = vision.HandLandmarker.create_from_options(options)

# Initialize Controllers
mouse = MouseController()
keyboard = KeyboardController()

# Configuration Constants
CAM_W, CAM_H = 640, 480
MARGIN_X, MARGIN_Y = 80, 60          # Active mouse interaction box in camera
SMOOTHING = 3.5                      # Exponential mouse smoothing factor
CLICK_DIST = 34                      # Pinch distance threshold in pixels
RELEASE_DIST = 42                    # Pinch release threshold in pixels
PINCH_HOLD_DRAG_TIME = 0.32          # Seconds before pinch triggers drag
DWELL_TIME = 0.42                    # Hover time to auto-type a key (seconds)

# Hand Connections for Skeleton Rendering
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index
    (5, 9), (9, 10), (10, 11), (11, 12),   # Middle
    (9, 13), (13, 14), (14, 15), (15, 16), # Ring
    (13, 17), (17, 18), (18, 19), (19, 20),# Pinky
    (0, 17)                                # Palm base
]

# Keyboard Layout Definition
KEY_ROWS = [
    ['Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P', 'BKSP'],
    ['A', 'S', 'D', 'F', 'G', 'H', 'J', 'K', 'L', ';', 'ENTER'],
    ['Z', 'X', 'C', 'V', 'B', 'N', 'M', ',', '.', 'SPACE'],
    ['CAPS', 'CLEAR', 'EXIT KB']
]

class KeyButton:
    def __init__(self, text, x, y, w, h):
        self.text = text
        self.x = x
        self.y = y
        self.w = w
        self.h = h

    def contains(self, px, py):
        return self.x <= px <= self.x + self.w and self.y <= py <= self.y + self.h

def build_keyboard_buttons(frame_w, frame_h):
    buttons = []
    start_y = int(frame_h * 0.42)
    row_height = 42
    spacing = 6
    pad_x = 14

    for r_idx, row in enumerate(KEY_ROWS):
        cur_y = start_y + r_idx * (row_height + spacing)
        # Determine total key weights for this row
        weights = []
        for k in row:
            if k in ('BKSP', 'ENTER'):
                weights.append(1.5)
            elif k in ('SPACE', 'CLEAR', 'EXIT KB'):
                weights.append(2.0)
            elif k == 'CAPS':
                weights.append(1.4)
            else:
                weights.append(1.0)

        total_weight = sum(weights)
        avail_width = frame_w - (2 * pad_x) - (len(row) - 1) * spacing
        unit_w = avail_width / total_weight

        cur_x = pad_x
        for k, w_mult in zip(row, weights):
            kw = int(unit_w * w_mult)
            buttons.append(KeyButton(k, int(cur_x), int(cur_y), kw, row_height))
            cur_x += kw + spacing

    return buttons

KEYBOARD_BUTTONS = build_keyboard_buttons(CAM_W, CAM_H)

# Application State
mode = "MOUSE"                       # "MOUSE" or "KEYBOARD"
prev_mouse_x, prev_mouse_y = SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2
pinch_start_time = None
is_dragging = False
last_right_click_time = 0
prev_scroll_y = None
caps_lock = False

# Keyboard typing state
hovered_key = None
hover_start_time = None
key_triggered = False
feedback_text = ""
feedback_timer = 0
typed_history = ""

# FPS calculation
fps_prev_time = time.time()
fps = 0.0

# Open Camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)

if not cap.isOpened():
    print("Error: Could not access camera /dev/video0. Please check permissions or device.")
    sys.exit(1)

print("[✓] Camera started successfully! Press 'q' or ESC in the window to quit.")

def draw_hud(frame, current_mode, status_msg, fps_val):
    # Top status bar
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 48), (20, 20, 25), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    # Mode Indicator Badge
    mode_color = (0, 255, 128) if current_mode == "MOUSE" else (255, 160, 0)
    cv2.rectangle(frame, (8, 8), (145, 40), mode_color, 2, cv2.LINE_AA)
    cv2.putText(frame, f"MODE: {current_mode}", (14, 30), cv2.FONT_HERSHEY_DUPLEX, 0.55, mode_color, 1, cv2.LINE_AA)

    # Top Toggle Button (Interactive)
    toggle_text = "⌨ OPEN KB" if current_mode == "MOUSE" else "🖱 MOUSE"
    btn_x1, btn_y1, btn_x2, btn_y2 = w - 160, 8, w - 8, 40
    cv2.rectangle(frame, (btn_x1, btn_y1), (btn_x2, btn_y2), (70, 70, 85), -1)
    cv2.rectangle(frame, (btn_x1, btn_y1), (btn_x2, btn_y2), (0, 200, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, toggle_text, (btn_x1 + 10, btn_y1 + 22), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # Action Status and FPS
    cv2.putText(frame, status_msg, (155, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(frame, f"FPS: {int(fps_val)}", (w - 240, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1, cv2.LINE_AA)

def draw_skeleton(frame, landmarks, w, h):
    # Draw bone connections
    for start_idx, end_idx in HAND_CONNECTIONS:
        pt1 = (int(landmarks[start_idx].x * w), int(landmarks[start_idx].y * h))
        pt2 = (int(landmarks[end_idx].x * w), int(landmarks[end_idx].y * h))
        cv2.line(frame, pt1, pt2, (40, 200, 240), 2, cv2.LINE_AA)

    # Draw landmark joints
    for idx, lm in enumerate(landmarks):
        px, py = int(lm.x * w), int(lm.y * h)
        if idx in (4, 8, 12, 16, 20):  # Fingertips
            cv2.circle(frame, (px, py), 6, (0, 255, 0), -1, cv2.LINE_AA)
            cv2.circle(frame, (px, py), 8, (255, 255, 255), 1, cv2.LINE_AA)
        else:
            cv2.circle(frame, (px, py), 3, (0, 140, 255), -1, cv2.LINE_AA)

def draw_keyboard(frame, buttons, active_btn, hover_progress, caps, preview_str, flash_btn=None):
    h, w = frame.shape[:2]

    # Semi-transparent background for keyboard zone
    overlay = frame.copy()
    start_y = int(h * 0.35)
    cv2.rectangle(overlay, (4, start_y), (w - 4, h - 4), (15, 18, 22), -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)

    # Preview Bar
    preview_y = start_y + 26
    cv2.rectangle(frame, (14, start_y + 4), (w - 14, preview_y), (35, 40, 48), -1)
    cv2.rectangle(frame, (14, start_y + 4), (w - 14, preview_y), (80, 90, 110), 1, cv2.LINE_AA)
    disp_text = f"Typed: {preview_str[-38:]}|"
    cv2.putText(frame, disp_text, (20, preview_y - 6), cv2.FONT_HERSHEY_DUPLEX, 0.45, (0, 255, 200), 1, cv2.LINE_AA)

    for btn in buttons:
        is_hovered = (active_btn is not None and active_btn.text == btn.text)
        is_flashed = (flash_btn is not None and flash_btn == btn.text)

        # Base key styling
        if is_flashed:
            bg_col = (0, 255, 120)
            border_col = (255, 255, 255)
            text_col = (0, 0, 0)
        elif is_hovered:
            bg_col = (60, 90, 140)
            border_col = (0, 230, 255)
            text_col = (255, 255, 255)
        else:
            bg_col = (30, 35, 42)
            border_col = (70, 80, 95)
            text_col = (230, 230, 230)

        # Draw Key Box
        cv2.rectangle(frame, (btn.x, btn.y), (btn.x + btn.w, btn.y + btn.h), bg_col, -1)
        cv2.rectangle(frame, (btn.x, btn.y), (btn.x + btn.w, btn.y + btn.h), border_col, 1 if not is_hovered else 2, cv2.LINE_AA)

        # Draw Dwell Progress Bar if hovered
        if is_hovered and hover_progress > 0.0:
            fill_w = int(btn.w * hover_progress)
            cv2.rectangle(frame, (btn.x, btn.y + btn.h - 4), (btn.x + fill_w, btn.y + btn.h), (0, 255, 160), -1)

        # Determine label text (apply caps to single letters)
        label = btn.text
        if len(label) == 1 and label.isalpha():
            label = label if caps else label.lower()

        # Center text inside key
        scale = 0.55 if len(label) <= 2 else 0.42
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, scale, 1)
        tx = btn.x + (btn.w - tw) // 2
        ty = btn.y + (btn.h + th) // 2
        cv2.putText(frame, label, (tx, ty), cv2.FONT_HERSHEY_DUPLEX, scale, text_col, 1, cv2.LINE_AA)

def send_key_action(key_text):
    global caps_lock, typed_history, mode
    if key_text == 'SPACE':
        keyboard.tap(Key.space)
        typed_history += " "
    elif key_text == 'BKSP':
        keyboard.tap(Key.backspace)
        typed_history = typed_history[:-1]
    elif key_text == 'ENTER':
        keyboard.tap(Key.enter)
        typed_history += "\n"
    elif key_text == 'CAPS':
        caps_lock = not caps_lock
    elif key_text == 'CLEAR':
        typed_history = ""
    elif key_text == 'EXIT KB':
        mode = "MOUSE"
    else:
        # Standard character
        char = key_text if caps_lock else key_text.lower()
        keyboard.tap(char)
        typed_history += char

# Main Processing Loop
while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        print("Camera frame not available. Exiting...")
        break

    # Calculate FPS
    now_time = time.time()
    fps = 0.9 * fps + 0.1 * (1.0 / max(now_time - fps_prev_time, 0.0001))
    fps_prev_time = now_time

    # Mirror horizontally for intuitive user interaction
    frame = cv2.flip(frame, 1)
    frame_h, frame_w = frame.shape[:2]

    # Convert to RGB for MediaPipe HandLandmarker
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    detection_result = landmarker.detect(mp_image)

    status_message = "Ready"
    flash_key = None

    if feedback_timer > 0:
        feedback_timer -= 1
        if feedback_timer == 0:
            feedback_text = ""

    # Check top mode toggle button click/hover with index finger
    toggle_btn_x1, toggle_btn_y1 = frame_w - 160, 8
    toggle_btn_x2, toggle_btn_y2 = frame_w - 8, 40

    if detection_result and detection_result.hand_landmarks:
        hand_landmarks = detection_result.hand_landmarks[0]
        draw_skeleton(frame, hand_landmarks, frame_w, frame_h)

        # Extract fingertip pixel coordinates
        # 4: Thumb, 8: Index, 12: Middle, 16: Ring, 20: Pinky
        thumb_tip = (int(hand_landmarks[4].x * frame_w), int(hand_landmarks[4].y * frame_h))
        index_tip = (int(hand_landmarks[8].x * frame_w), int(hand_landmarks[8].y * frame_h))
        middle_tip = (int(hand_landmarks[12].x * frame_w), int(hand_landmarks[12].y * frame_h))
        ring_tip = (int(hand_landmarks[16].x * frame_w), int(hand_landmarks[16].y * frame_h))
        pinky_tip = (int(hand_landmarks[20].x * frame_w), int(hand_landmarks[20].y * frame_h))

        # Check finger extension (tip higher than PIP joint in image coordinates)
        index_up = hand_landmarks[8].y < hand_landmarks[6].y
        middle_up = hand_landmarks[12].y < hand_landmarks[10].y
        ring_up = hand_landmarks[16].y < hand_landmarks[14].y
        pinky_up = hand_landmarks[20].y < hand_landmarks[18].y

        # Distances
        dist_thumb_index = math.hypot(thumb_tip[0] - index_tip[0], thumb_tip[1] - index_tip[1])
        dist_thumb_middle = math.hypot(thumb_tip[0] - middle_tip[0], thumb_tip[1] - middle_tip[1])

        # Check if index finger hits the top toggle button
        if toggle_btn_x1 <= index_tip[0] <= toggle_btn_x2 and toggle_btn_y1 <= index_tip[1] <= toggle_btn_y2:
            cv2.circle(frame, index_tip, 12, (0, 255, 255), 2)
            if dist_thumb_index < CLICK_DIST:
                mode = "KEYBOARD" if mode == "MOUSE" else "MOUSE"
                time.sleep(0.25)  # debounce switch

        # ==========================================================
        # 1. KEYBOARD MODE
        # ==========================================================
        if mode == "KEYBOARD":
            status_message = "Hover/Pinch key to type"

            # Check which button index finger tip is hovering over
            current_hover = None
            for btn in KEYBOARD_BUTTONS:
                if btn.contains(index_tip[0], index_tip[1]):
                    current_hover = btn
                    break

            if current_hover is not None:
                if hovered_key is None or hovered_key.text != current_hover.text:
                    hovered_key = current_hover
                    hover_start_time = time.time()
                    key_triggered = False

                elapsed = time.time() - hover_start_time
                progress = min(1.0, elapsed / DWELL_TIME)

                # Two trigger methods: 1) Dwell time reached OR 2) Quick pinch on key
                is_pinch = (dist_thumb_index < CLICK_DIST)
                if (progress >= 1.0 or is_pinch) and not key_triggered:
                    send_key_action(current_hover.text)
                    key_triggered = True
                    flash_key = current_hover.text
                    status_message = f"Typed: {current_hover.text}"
                    cv2.circle(frame, index_tip, 15, (0, 255, 120), -1)

                draw_keyboard(frame, KEYBOARD_BUTTONS, hovered_key, progress, caps_lock, typed_history, flash_key)
            else:
                hovered_key = None
                hover_start_time = None
                key_triggered = False
                draw_keyboard(frame, KEYBOARD_BUTTONS, None, 0.0, caps_lock, typed_history, None)

            # Draw cursor on index fingertip
            cv2.circle(frame, index_tip, 9, (0, 255, 255), -1, cv2.LINE_AA)

        # ==========================================================
        # 2. MOUSE MODE
        # ==========================================================
        else:
            # Draw active interaction bounding box
            cv2.rectangle(frame, (MARGIN_X, MARGIN_Y), (frame_w - MARGIN_X, frame_h - MARGIN_Y), (60, 65, 80), 1, cv2.LINE_AA)
            cv2.putText(frame, "Touch zone", (MARGIN_X + 6, MARGIN_Y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 130, 150), 1)

            # Map camera index coordinates to full screen resolution
            raw_screen_x = np.interp(index_tip[0], (MARGIN_X, frame_w - MARGIN_X), (0, SCREEN_WIDTH))
            raw_screen_y = np.interp(index_tip[1], (MARGIN_Y, frame_h - MARGIN_Y), (0, SCREEN_HEIGHT))

            # Apply smooth exponential moving average to eliminate jitter
            smooth_mouse_x = prev_mouse_x + (raw_screen_x - prev_mouse_x) / SMOOTHING
            smooth_mouse_y = prev_mouse_y + (raw_screen_y - prev_mouse_y) / SMOOTHING
            prev_mouse_x, prev_mouse_y = smooth_mouse_x, smooth_mouse_y

            # Clamp coordinates to monitor bounds
            target_x = int(max(0, min(SCREEN_WIDTH - 1, smooth_mouse_x)))
            target_y = int(max(0, min(SCREEN_HEIGHT - 1, smooth_mouse_y)))

            # Scroll Gesture: Index + Middle up, other fingers down
            if index_up and middle_up and not ring_up and not pinky_up and dist_thumb_index > 50:
                mid_y = (index_tip[1] + middle_tip[1]) // 2
                cv2.circle(frame, index_tip, 8, (255, 180, 0), -1)
                cv2.circle(frame, middle_tip, 8, (255, 180, 0), -1)

                if prev_scroll_y is not None:
                    delta_scroll = mid_y - prev_scroll_y
                    if delta_scroll < -10:
                        mouse.scroll(0, 2)
                        status_message = "Scrolling UP ↑"
                        prev_scroll_y = mid_y
                    elif delta_scroll > 10:
                        mouse.scroll(0, -2)
                        status_message = "Scrolling DOWN ↓"
                        prev_scroll_y = mid_y
                else:
                    prev_scroll_y = mid_y
                    status_message = "Scroll Mode"

            else:
                prev_scroll_y = None

                # Cursor Move
                try:
                    mouse.position = (target_x, target_y)
                except Exception:
                    pass

                # Draw pointer indicator at index fingertip
                cv2.circle(frame, index_tip, 8, (0, 255, 255), -1, cv2.LINE_AA)

                # Right Click: Middle + Thumb pinch
                if dist_thumb_middle < CLICK_DIST and not is_dragging:
                    mid_pt = ((thumb_tip[0] + middle_tip[0]) // 2, (thumb_tip[1] + middle_tip[1]) // 2)
                    cv2.circle(frame, mid_pt, 12, (255, 0, 255), -1, cv2.LINE_AA)
                    if time.time() - last_right_click_time > 0.45:
                        mouse.click(Button.right)
                        last_right_click_time = time.time()
                        status_message = "Right Click [✓]"

                # Left Click / Drag: Thumb + Index pinch
                elif dist_thumb_index < CLICK_DIST:
                    mid_pt = ((thumb_tip[0] + index_tip[0]) // 2, (thumb_tip[1] + index_tip[1]) // 2)
                    cv2.circle(frame, mid_pt, 12, (0, 255, 0), -1, cv2.LINE_AA)

                    if pinch_start_time is None:
                        pinch_start_time = time.time()

                    pinch_duration = time.time() - pinch_start_time
                    if pinch_duration >= PINCH_HOLD_DRAG_TIME:
                        if not is_dragging:
                            mouse.press(Button.left)
                            is_dragging = True
                        status_message = "Dragging [HOLD]"
                    else:
                        status_message = "Pinch"

                # Pinch Released
                elif dist_thumb_index >= RELEASE_DIST:
                    if pinch_start_time is not None:
                        pinch_duration = time.time() - pinch_start_time
                        if is_dragging:
                            mouse.release(Button.left)
                            is_dragging = False
                            status_message = "Drop [✓]"
                        elif pinch_duration < PINCH_HOLD_DRAG_TIME:
                            mouse.click(Button.left)
                            status_message = "Left Click [✓]"
                        pinch_start_time = None
                    else:
                        status_message = "Pointer Moving"

    else:
        # No hand detected
        status_message = "Hand not in view"
        prev_scroll_y = None
        if is_dragging:
            mouse.release(Button.left)
            is_dragging = False
        pinch_start_time = None
        if mode == "KEYBOARD":
            draw_keyboard(frame, KEYBOARD_BUTTONS, None, 0.0, caps_lock, typed_history, None)

    # Render HUD overlay on top of frame
    draw_hud(frame, mode, status_message, fps)

    # Display window
    cv2.imshow("AI Virtual Mouse & Keyboard - Hand Gestures", frame)

    # Keyboard hotkey listeners
    key_code = cv2.waitKey(1) & 0xFF
    if key_code in (ord('q'), 27):  # 'q' or ESC
        break
    elif key_code == ord('m'):
        mode = "MOUSE"
    elif key_code == ord('k'):
        mode = "KEYBOARD"
    elif key_code == 9:             # Tab key
        mode = "KEYBOARD" if mode == "MOUSE" else "MOUSE"

# Clean up
if is_dragging:
    try:
        mouse.release(Button.left)
    except Exception:
        pass

cap.release()
cv2.destroyAllWindows()
print("[*] Camera and resources released cleanly. Goodbye!")
