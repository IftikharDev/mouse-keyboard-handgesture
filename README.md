# AI Virtual Mouse & Keyboard via Hand Gestures 🖐️🖱️⌨️

Control your computer hands-free using your webcam with AI hand tracking. No physical mouse or keyboard needed!

---

## 🚀 Features

### 🖱️ Virtual Mouse Mode
- **Smooth Cursor Movement**: Point and move with your index fingertip. Built-in exponential smoothing eliminates jitter.
- **Left Click**: Quick pinch between your Thumb and Index fingertip.
- **Drag & Drop**: Hold pinch (Thumb + Index) for more than 0.3 seconds to drag, release to drop.
- **Right Click**: Pinch between your Thumb and Middle fingertip.
- **Scrolling**: Extend both Index and Middle fingers together and move up/down to scroll.

### ⌨️ Virtual Keyboard Mode
- **On-Screen Transparent Overlay**: Semi-transparent QWERTY keyboard rendered directly on your camera feed.
- **Dual Typing Trigger**:
  - **Dwell-to-Type**: Hover index fingertip over any key for 0.4s to automatically type it.
  - **Pinch-to-Type**: Pinch Thumb + Index on a key to type immediately.
- **Real-Time Preview Bar**: See the characters you've typed directly on screen.
- **System Injection**: Automatically types real keystrokes into any active focused window (browser, text editor, terminal, etc.).
- **Special Keys**: `SPACE`, `BKSP` (Backspace), `ENTER`, `CAPS` (Caps Lock), `CLEAR`.

### 🔄 Seamless Mode Switching
- Touch the on-screen **`[ ⌨ OPEN KB ]`** / **`[ 🖱 MOUSE ]`** button with your index finger.
- Keyboard shortcuts:
  - Press `Tab`: Toggle between Mouse and Keyboard mode.
  - Press `m`: Switch to Mouse Mode.
  - Press `k`: Switch to Keyboard Mode.
  - Press `q` or `ESC`: Quit application.

---

## 🛠️ Setup & Installation

### 1. Set up Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run the Application
```bash
python faceRecognition.py
```
*(The MediaPipe hand tracking model will automatically download on first run if not already present).*

---

## 🖐️ Gesture Reference Card

| Action | Hand Gesture |
| :--- | :--- |
| **Move Cursor** | Point with Index finger inside the active touch zone |
| **Left Click** | Quick pinch (Thumb + Index fingertip) |
| **Drag & Drop** | Hold pinch > 0.3s to grab, release to drop |
| **Right Click** | Pinch Thumb + Middle fingertip |
| **Scroll Up / Down** | Two fingers up (Index + Middle) moving up or down |
| **Type a Key** | Hover index fingertip over key for 0.4s OR pinch on key |
| **Toggle Mode** | Touch the top toggle button or press `Tab` |
