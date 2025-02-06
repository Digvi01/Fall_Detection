import cv2
import mediapipe as mp
import numpy as np
import time
import pygame
import threading
import streamlit as st
from PIL import Image

# Initialize pygame mixer for sound
pygame.mixer.init()
ALERT_SOUND = "alert.mp3"

def play_alert_sound():
    """Plays an alert sound in a separate thread."""
    try:
        pygame.mixer.music.load(ALERT_SOUND)
        pygame.mixer.music.play()
    except Exception as e:
        print("Sound error:", e)

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose()

# Parameters
GROUND_SMOOTHING = 0.9
FALL_THRESHOLD = 0.35
ANGLE_THRESHOLD = 35
FALL_DURATION = 1.5
STILL_FALL_DURATION = 5
NO_MOVEMENT_DURATION = 3
SITTING_ANGLE_THRESHOLD = 60

# Tracking variables
previous_ground_y = None
fall_start_time = None
still_fall_start_time = None
no_movement_start_time = None
fall_detected = False
previous_torso_y = None

# Streamlit app
st.title("Fall Detection System")
st.write("This application detects falls using MediaPipe Pose Estimation.")

# Start video capture
cap = cv2.VideoCapture(0)
frame_placeholder = st.empty()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        st.error("Failed to capture video. Please check your camera.")
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(rgb_frame)
    h, w, _ = frame.shape

    ground_candidates = []
    keypoints_on_ground = 0
    total_keypoints = 0
    torso_angle = None

    if result.pose_landmarks:
        landmarks = result.pose_landmarks.landmark

        for i in [mp_pose.PoseLandmark.LEFT_ANKLE.value, mp_pose.PoseLandmark.RIGHT_ANKLE.value]:
            ground_candidates.append(int(landmarks[i].y * h))

        if ground_candidates:
            estimated_ground_y = int(np.median(ground_candidates))
            if previous_ground_y is None:
                previous_ground_y = estimated_ground_y
            else:
                previous_ground_y = int(GROUND_SMOOTHING * previous_ground_y + (1 - GROUND_SMOOTHING) * estimated_ground_y)

        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        
        dx = (right_shoulder.x - left_shoulder.x) * w
        dy = (right_shoulder.y - left_shoulder.y) * h
        if dx != 0:
            torso_angle = np.degrees(np.arctan(abs(dy / dx)))

        for i in range(len(landmarks)):
            x, y = int(landmarks[i].x * w), int(landmarks[i].y * h)
            total_keypoints += 1
            if y >= previous_ground_y:
                keypoints_on_ground += 1
                cv2.circle(frame, (x, y), 6, (0, 0, 255), -1)
            else:
                cv2.circle(frame, (x, y), 6, (0, 255, 0), -1)

        is_lying_down = torso_angle is not None and torso_angle < ANGLE_THRESHOLD
        is_sitting = torso_angle is not None and torso_angle > SITTING_ANGLE_THRESHOLD

        if keypoints_on_ground / total_keypoints > FALL_THRESHOLD and is_lying_down and not is_sitting:
            if not fall_detected:
                fall_start_time = time.time()
                fall_detected = True
            elif time.time() - fall_start_time >= FALL_DURATION:
                cv2.putText(frame, "FALL DETECTED!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                threading.Thread(target=play_alert_sound, daemon=True).start()

        elif fall_detected and is_lying_down:
            if still_fall_start_time is None:
                still_fall_start_time = time.time()
            elif time.time() - still_fall_start_time >= STILL_FALL_DURATION:
                cv2.putText(frame, "PERSON STILL LYING DOWN!", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                threading.Thread(target=play_alert_sound, daemon=True).start()

            if no_movement_start_time is None:
                no_movement_start_time = time.time()
            elif time.time() - no_movement_start_time >= NO_MOVEMENT_DURATION:
                cv2.putText(frame, "NO MOVEMENT DETECTED!", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                threading.Thread(target=play_alert_sound, daemon=True).start()
        else:
            fall_detected = False
            fall_start_time = None
            still_fall_start_time = None
            no_movement_start_time = None

    cv2.line(frame, (0, previous_ground_y), (w, previous_ground_y), (255, 255, 0), 2)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)

    if st.button("Stop"):
        break

cap.release()
cv2.destroyAllWindows()
