# 🎵 Hand Gesture Volume Control
The project allows you to control your system volume using hand gestures captured by a webcam by moving your thumb and index finger closer and farther apart to increase and decrease volume in real time

## ✨ Features
✅ Real-time hand tracking using MediaPipe
✅ System volume control through the PyCaw
 library.
✅ Smooth volume interpolation for natural adjustments.
✅ Press q to quit the application safely.

## 🛠️ Requirements
1. Make sure you install Python 3.7+ or under 3.10
2. Install the required dependencies:
   pip install opencv-python mediapipe numpy comtypes pycaw

## 🚀 How to Run
1. Clone or download this repository.
2. Save the provided script (e.g., hand_volume_control.py).
3. Run the script:
python hand_volume_control.py

5. Ensure your webcam is connected and enabled.
6. Show your hand in front of the camera, and:
  Move thumb and index finger closer ➡️ Volume decreases.
  Move them apart ➡️ Volume increases.

## 📸 Controls

| Key / Gesture     | Action           |
| ----------------- | ---------------- |
| Thumb & Index Gap | Adjust Volume    |
| **q** (keyboard)  | Quit Application |

## ⚠️ Notes

Works on Windows only (due to PyCaw and Windows API usage).
Ensure camera permissions are enabled for Python/OpenCV.
For best results, use in a well-lit environment.

## 📂 Project Structure
hand-gesture-volume/
│
├── hand_volume_control.py   # Main script
└── README.md                # Project documentation

## 🖥️ Tech Stack
OpenCV – Video capture and image processing.
MediaPipe – Real-time hand tracking.
NumPy – Mathematical operations.
PyCaw – System volume control.
Windows API – Simulates volume key presses.

## 🔧 Future Improvements
Add cross-platform support (Linux/Mac).
Add gesture-based mute/unmute.
Display on-screen volume indicator.
--------------------------------------------------------------------------------
## 👤 Author
Created by Kesaobaka Lekaote 💻


   






