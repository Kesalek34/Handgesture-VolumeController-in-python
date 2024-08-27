import cv2
import numpy as np
import mediapipe as mp
from ctypes import cast,POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities,IAudioEndpointVolume

#Mediapipe hands setup
mp_hands = mp.solutions.hands #Initialize MediaPipe hands module
hands =mp_hands.Hands(min_detection_confidence =0.7 ,min_tracking_confidence =0.7) #Setup the hands model
mp_draw = mp.solutions.drawing_utils #utility for drwaing hand landmarks

#Pycaw setup for volume control
devices = AudioUtilities.GetSpeakers()  #Get the default audio output device(speakers)
interface = devices.Activate(
    IAudioEndpointVolume._iid_,CLSCTX_ALL, None) #Activate the audio endpoint for volume control
volume = cast(interface, POINTER(IAudioEndpointVolume)) #Cast the interface to the IAudioEndpointVolume type

vol_range = volume.GetVolumeRange() #Retrieve the volume rang(min and max values)
min_vol = vol_range[0]  #Mininum volume level
max_vol = vol_range[1] #Minuim volume level

#Start capturing vidoe from the webcam
cap = cv2.VideoCapture(0) #Open the default webcam

while True:
    success, img = cap.read() #Capture a frame from the webcam
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #convert the image to RGB(required for mediaPipe)
    results = hands.process(img_rgb) #Process the RGB image to detect hands

    if results.multi_hand_landmarks: #if hands are detected in frame
        for hand_landmarks in results.multi_hand_landmarks: #Iterate through each detected hand
            #Extract landmarks for thumb and index finger
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP] #Thumb tip landmark
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP] #Index finger tip landmark

            #Convert landmark positions to pixel coordinates
            h, w, c = img.shape #Get shape height, width and channe of the image
            thumb_x, thumb_y = int(thumb_tip.x* w), int(thumb_tip.y*h) #Convert thumb coordinates to pixels
            index_x, index_y = int(index_tip.x *w), int(index_tip.y *h) #Convert index finger coordinates to pixels

            #Calculate the distance between thumb and index finger
            distance = np.hypot(index_x - thumb_x, index_y - thumb_y) #Euclidean distance

            #Map the distance between thumb and index finger
            vol = np.interp(distance, [30,300],[min_vol, max_vol]) #Map the distance to the volume control range 
            volume.SetMasterVolumeLevel(vol,None) #Set system volume to the calculated level

            #Draw landmarks and connections on the image 
            mp_draw.draw_landmarks(img,hand_landmarks, mp_hands.HAND_CONNECTIONS) #Draw hand landmarks
            cv2.circle(img, (thumb_x, thumb_y),10,(255,0,0), cv2.FILLED) #Draw a circle on the thumb tip
            cv2.circle(img, (index_x, index_y),10,(255,0,0), cv2.FILLED) #Draw a circle on the index finger tip 
            cv2.line(img, (thumb_x, thumb_y), (index_x, index_y), (255,0,0), 3) #Draw a line between thumb and index finger

    #Display the image with the drawings
    cv2.imshow("Volume Control", img) #Show the image in a window titled "Volume Control"

    if cv2.waitKey(1) & 0xFF == ord('q'): #Wait for the 'q' key to exit the loop
        break

#Release the resources
cap.release() #Release the webcam
cv2.destroyAllWindows() #Close all OpenCv windows


