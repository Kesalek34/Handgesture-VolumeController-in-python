import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # suppress TensorFlow logs

import cv2
import numpy as np
import mediapipe as mp
from ctypes import cast, POINTER, windll
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume


mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils


try:
    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume = cast(interface, POINTER(IAudioEndpointVolume))
    min_vol, max_vol, _ = volume.GetVolumeRange()
except Exception as e:
    print(f"Error initializing audio: {e}")
    exit()

VK_VOLUME_UP = 0xAF
VK_VOLUME_DOWN = 0xAE
KEYEVENTF_EXTENDEDKEY = 0x0001
KEYEVENTF_KEYUP = 0x0002

def press_volume_key(key, steps=1):
    """Simulate pressing volume up/down keys"""
    for _ in range(steps):
        windll.user32.keybd_event(key, 0, KEYEVENTF_EXTENDEDKEY, 0)
        windll.user32.keybd_event(key, 0, KEYEVENTF_EXTENDEDKEY | KEYEVENTF_KEYUP, 0)

try:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera.")
except Exception as e:
    print(f"Error initializing camera: {e}")
    exit()

prev_vol_percent = int(np.interp(volume.GetMasterVolumeLevel(), [min_vol, max_vol], [0, 100]))

try:
    while True:
        success, img = cap.read()
        if not success:
            print("Failed to read from camera.")
            break

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        vol_percent = prev_vol_percent

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                h, w, _ = img.shape
                thumb = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                index = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                thumb_x, thumb_y = int(thumb.x * w), int(thumb.y * h)
                index_x, index_y = int(index.x * w), int(index.y * h)

                
                cv2.circle(img, (thumb_x, thumb_y), 10, (255, 0, 0), cv2.FILLED)
                cv2.circle(img, (index_x, index_y), 10, (255, 0, 0), cv2.FILLED)
                cv2.line(img, (thumb_x, thumb_y), (index_x, index_y), (255, 0, 0), 3)

                distance = np.hypot(index_x - thumb_x, index_y - thumb_y)

                
                target_vol = int(np.interp(distance, [30, 300], [0, 100]))
                step_factor = np.interp(distance, [30, 300], [1, 5])
                vol_percent = prev_vol_percent + int((target_vol - prev_vol_percent) * step_factor / 5)
                vol_percent = max(0, min(100, vol_percent))

                vol = np.interp(vol_percent, [0, 100], [min_vol, max_vol])
                volume.SetMasterVolumeLevel(vol, None)

                diff = vol_percent - prev_vol_percent
                if diff > 0:
                    press_volume_key(VK_VOLUME_UP, steps=diff//2 or 1)
                elif diff < 0:
                    press_volume_key(VK_VOLUME_DOWN, steps=abs(diff)//2 or 1)

                prev_vol_percent = vol_percent

        cv2.imshow("🎵 Volume Control", img)

        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            print("Exiting...")
            break

except Exception as e:
    print(f"An error occurred: {e}")

finally:
    cap.release()
    cv2.destroyAllWindows()
