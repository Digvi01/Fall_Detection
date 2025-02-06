import cv2
import mediapipe as mp
import numpy as np
import time
import playsound
import threading
import streamlit as st
from PIL import Image

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose()

# Parameters
GROUND_SMOOTHING = 0.9  # Adjust ground level gradually for better accuracy
FALL_THRESHOLD = 0.35  # % of keypoints touching ground
ANGLE_THRESHOLD = 35  # If torso angle <35° (close to horizontal), it's a fall
FALL_DURATION = 1.5  # Seconds before confirming fall
STILL_FALL_DURATION = 5  # Time lying down before alert triggers
NO_MOVEMENT_DURATION = 3  # Additional seconds before alert if no movement
ALERT_SOUND = "alert.mp3"  # Alert sound
SITTING_ANGLE_THRESHOLD = 60  # Torso angle above this is considered sitting

# Tracking variables
previous_ground_y = None
fall_start_time = None
still_fall_start_time = None
no_movement_start_time = None
fall_detected = False
previous_torso_y = None

def play_alert_sound():
    """Plays an alert sound in a separate thread."""
    try:
        playsound.playsound(ALERT_SOUND)
    except Exception as e:
        print("Sound error:", e)

# Streamlit app
st.title("Fall Detection System")
st.write("This application detects falls using MediaPipe Pose Estimation.")

# Start video capture
cap = cv2.VideoCapture(0)

# Placeholder for displaying the video feed
frame_placeholder = st.empty()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        st.error("Failed to capture video. Please check your camera.")
        break

    # Convert frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Perform pose detection
    result = pose.process(rgb_frame)
    h, w, _ = frame.shape

    ground_candidates = []
    keypoints_on_ground = 0
    total_keypoints = 0
    torso_angle = None
    torso_speed = 0

    if result.pose_landmarks:
        landmarks = result.pose_landmarks.landmark

        # Identify ground level using ankle points
        for i in [mp_pose.PoseLandmark.LEFT_ANKLE.value, mp_pose.PoseLandmark.RIGHT_ANKLE.value]:
            ground_candidates.append(int(landmarks[i].y * h))

        # Smooth ground estimation
        if ground_candidates:
            estimated_ground_y = int(np.median(ground_candidates))
            if previous_ground_y is None:
                previous_ground_y = estimated_ground_y
            else:
                previous_ground_y = int(GROUND_SMOOTHING * previous_ground_y + (1 - GROUND_SMOOTHING) * estimated_ground_y)

        # Get torso keypoints
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
        right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]

        # Calculate torso angle
        dx = (right_shoulder.x - left_shoulder.x) * w
        dy = (right_shoulder.y - left_shoulder.y) * h
        if dx != 0:
            torso_angle = np.degrees(np.arctan(abs(dy / dx)))

        # Count keypoints near the ground
        for i in range(len(landmarks)):
            x, y = int(landmarks[i].x * w), int(landmarks[i].y * h)
            total_keypoints += 1
            if y >= previous_ground_y:
                keypoints_on_ground += 1
                cv2.circle(frame, (x, y), 6, (0, 0, 255), -1)  # Red for ground contact
            else:
                cv2.circle(frame, (x, y), 6, (0, 255, 0), -1)  # Green for normal keypoints

        # Check if the person is lying down (fall position)
        is_lying_down = torso_angle is not None and torso_angle < ANGLE_THRESHOLD
        is_sitting = torso_angle is not None and torso_angle > SITTING_ANGLE_THRESHOLD

        # Detect fall based on motion and position (excluding sitting)
        if keypoints_on_ground / total_keypoints > FALL_THRESHOLD and is_lying_down and not is_sitting:
            if not fall_detected:
                fall_start_time = time.time()
                fall_detected = True
            elif time.time() - fall_start_time >= FALL_DURATION:
                cv2.putText(frame, "FALL DETECTED!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                threading.Thread(target=play_alert_sound, daemon=True).start()

        # If a person has already fallen and remains still
        elif fall_detected and is_lying_down:
            if still_fall_start_time is None:
                still_fall_start_time = time.time()
            elif time.time() - still_fall_start_time >= STILL_FALL_DURATION:
                cv2.putText(frame, "PERSON STILL LYING DOWN!", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                threading.Thread(target=play_alert_sound, daemon=True).start()
            
            # Detect no movement after falling
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

    # Draw ground reference
    cv2.line(frame, (0, previous_ground_y), (w, previous_ground_y), (255, 255, 0), 2)

    # Convert frame to RGB for Streamlit
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Display the frame in Streamlit
    frame_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)

    # Exit on 'q' (not applicable in Streamlit, so we use a stop button)
    if st.button("Stop"):
        break

cap.release()
cv2.destroyAllWindows()
