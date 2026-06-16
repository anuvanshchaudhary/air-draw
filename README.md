# Air Canvas 🎨✨

[![Live Demo](https://img.shields.io/badge/Live-Demo-brightgreen.svg)](https://anuvanshchaudhary.github.io/air-draw/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-green.svg)](https://opencv.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10+-teal.svg)](https://mediapipe.dev/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Air Canvas** is a premium, gesture-controlled drawing application powered by computer vision. It allows you to paint in the air using your webcam, featuring a side-by-side split layout, glowing neon brush effects, geometric shape snapping, and interactive status indicators.

Originally developed as a local Python desktop application, the project has been updated to include a **browser-native Web Demo** designed for web portfolios, allowing recruiters and viewers to try the interactive tracking experience live without cloning any code!

---

## 🛠️ Project Tech Stacks

### 1. Interactive Web Demo (Client-Side)
Developed to showcase the project online. It runs 100% locally in the visitor's browser for lag-free performance.
* **Frontend**: HTML5 Canvas, Vanilla CSS (with glassmorphic styling), and JavaScript.
* **Hand Tracking**: Powered by the **MediaPipe Hands JavaScript SDK** loaded via CDN.

### 2. Original Desktop Application (Python)
The core desktop application written in Python. It relies on the following key dependencies:
* **`opencv-python` (>= 4.8.0)**: Used for camera capture, mirrored frame processing, desktop GUI windowing (`cv2.imshow`), and primitive canvas drawing.
* **`mediapipe` (>= 0.10.30)**: Google's machine learning Tasks API used to track hand landmarks in a local video stream.
* **`numpy` (>= 1.24.0)**: Handles multi-dimensional array operations, masking, and high-performance pixel blending.

---

## 🌟 Key Features

### 1. Interactive Hand Gestures
Control your canvas naturally using hand tracking:
* **Draw Mode ☝️** (*Index finger up*): Paint lines onto the canvas. Features **Stroke Smoothing / Stabilization** using a rolling average buffer on the index fingertip position to eliminate tracking jitter.
* **Hover Mode ✌️** (*Index + Middle up*): Navigate the UI without drawing, select colors, or adjust brush sizes.
* **Pinch-to-Resize 🤏**: Pinch your thumb and index finger together to dynamically resize the brush from `2px` up to `30px` based on pinch distance.
* **Erase Mode 🖐️** (*Index + Middle + Ring up*): Erase parts of your drawing with a thick brush.
* **Fist Hold (Clear) ✊** (*Fist held for 2 seconds*): Clear the entire canvas. A **Dual-Sided Clear HUD** shows a circular progress indicator around your wrist (left screen) and a larger progress indicator in the center of the drawing canvas (right screen).

### 2. Premium 50-50 Split Screen Layout
* **Left Screen (Webcam Feed)**: Displays your mirrored camera input overlayed with tracked hand landmarks, top bar controls, gesture mode indicators, and FPS logs.
* **Right Screen (Drawing Canvas)**: Shows your artwork in full size on a dark drawing canvas.

### 3. Magical Neon Glow & Sparkle Effects
* **Neon Glow Paint**: Strokes are rendered as multilayered neon light tubes featuring a thick outer glow, medium inner glow, and a high-contrast core line.
* **Starry Sparkles**: Beautiful 4-pointed and 8-pointed starry sparkle particles are drawn along your paths. Sparkles are computed using deterministic coordinate hashing, keeping them completely stable and non-flickering.
* **Smart Eraser**: Eraser strokes are kept clean and black, leaving zero glow artifacts behind.

### 4. Interactive Top Bar Menu
* **Vibrant 9-Color Palette**: Paint with Coral Red, Orange, Sunny Yellow, Emerald Green, Cyan, Royal Blue, Purple, Hot Pink, or White. Swatches display a glowing double-ring selection border when active.
* **Bottom Glow Border**: A dynamic border runs along the bottom of the top bar, glowing in the active brush color.
* **Undo & Redo Actions**: Visual buttons to navigate your stroke history.
* **Manual Brush Size Adjustment**: Interactive `[-]` and `[+]` buttons to change size.
* **Live Brush Indicator**: A widget displaying a live preview circle of your brush size and its numeric value (e.g. `12px`).
* **Hover Scale Effects**: Color swatches scale up and buttons reveal a white halo outline when your finger hovers over them.

### 5. Floating HUD & Recognition Status Cards
* **Gesture Mode HUD**: A dark glass pill in the bottom-left corner with a color-coded status dot (Green for draw, Orange for hover, Purple for erase).
* **Performance FPS Monitor**: An elegant status pill in the top-right that color-codes your frame rate (Green for >=20 FPS, Orange for 10-19 FPS, Red for <10 FPS).
* **Doodle Recognition Card ⭐**: Renders a top-center floating glass card displaying the recognized doodle name and emoji (e.g., "Detected: Star ⭐", "Detected: Heart ❤️", "Detected: Triangle 🔺", "Detected: Line 📏") for 2.0 seconds after pen-up.
* **Glassmorphic Save Notification**: A centered card showing a success checkmark when your drawing is saved.

### 6. Automated Shape & Line Snapping
* Analyzes your stroke path when a drawing gesture ends.
* Automatically snaps hand-drawn circles, rectangles, and near-straight open strokes to perfect circles, rectangles, and straight lines if confidence is high.

### 7. Dual-Format Exports
Press **S** on your keyboard (or click the Save button in the Web demo) to export your drawing. It will save two images in the `saves/` directory:
1. **Drawing Canvas**: Your artwork on a solid black background.
2. **Composite Image**: Your drawing blended over a clean copy of your webcam frame (without the top bar HUD and landmarks).

---

## ⌨️ Keyboard Controls (Desktop Version)

| Key | Action |
| --- | --- |
| `S` | Export drawing and composite image |
| `Ctrl + Z` | Undo last stroke |
| `Ctrl + Y` | Redo last undone stroke |
| `Q` | Exit application |

---

## 🚀 Getting Started

### 🌐 Live Web Demo (GitHub Pages)

You can try the live demo instantly:
👉 **[Open Air Canvas Web Demo](https://anuvanshchaudhary.github.io/air-draw/)**

*(To run the web version locally, simply open the `index.html` file in any modern web browser).*

---

### 💻 Running the Python Desktop Application

#### Prerequisites
Ensure you have Python 3.8+ installed on your system.

#### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

#### 2. Download the Landmark Model
The hand tracking module requires the Google MediaPipe Hand Landmarker model. It will check for `hand_landmarker.task` in the project directory. If it is missing, download it from the following URL and place it in the project folder:
[hand_landmarker.task](https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task)

#### 3. Run the Application
```bash
python main.py
```
