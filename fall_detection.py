import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import time
import playsound
import threading

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Parameters
GROUND_SMOOTHING = 0.9
FALL_THRESHOLD = 0.35
ANGLE_THRESHOLD = 35
FALL_DURATION = 1.5
STILL_FALL_DURATION = 5
NO_MOVEMENT_DURATION = 3
ALERT_SOUND = "alert.mp3"
SITTING_ANGLE_THRESHOLD = 60

# Tracking variables
previous_ground_y = None
fall_start_time = None
still_fall_start_time = None
no_movement_start_time = None
fall_detected = False

# Function to play alert sound
def play_alert_sound():
    try:
        playsound.playsound(ALERT_SOUND)
    except Exception as e:
        st.error(f"Sound error: {e}")

# Streamlit app
def main():
    st.title("Real-Time Fall Detection using Pose Estimation")
    run_detection = st.checkbox("Start Fall Detection")
    
    cap = cv2.VideoCapture(0)
    stframe = st.empty()

    while run_detection and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(rgb_frame)
        h, w, _ = frame.shape

        global previous_ground_y, fall_start_time, still_fall_start_time, no_movement_start_time, fall_detected
        ground_candidates = []
        keypoints_on_ground = 0
        total_keypoints = 0
        torso_angle = None

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

            # Calculate torso angle
            left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            dx = (right_shoulder.x - left_shoulder.x) * w
            dy = (right_shoulder.y - left_shoulder.y) * h
            if dx != 0:
                torso_angle = np.degrees(np.arctan(abs(dy / dx)))

            # Count keypoints near the ground
            for i in range(len(landmarks)):
                y = int(landmarks[i].y * h)
                total_keypoints += 1
                if y >= previous_ground_y:
                    keypoints_on_ground += 1

            # Check if the person is lying down (fall position)
            is_lying_down = torso_angle is not None and torso_angle < ANGLE_THRESHOLD
            is_sitting = torso_angle is not None and torso_angle > SITTING_ANGLE_THRESHOLD

            # Detect fall
            if keypoints_on_ground / total_keypoints > FALL_THRESHOLD and is_lying_down and not is_sitting:
                if not fall_detected:
                    fall_start_time = time.time()
                    fall_detected = True
                elif time.time() - fall_start_time >= FALL_DURATION:
                    st.error("FALL DETECTED!")
                    threading.Thread(target=play_alert_sound, daemon=True).start()

            # Check for still lying down
            elif fall_detected and is_lying_down:
                if still_fall_start_time is None:
                    still_fall_start_time = time.time()
                elif time.time() - still_fall_start_time >= STILL_FALL_DURATION:
                    st.warning("PERSON STILL LYING DOWN!")
                    threading.Thread(target=play_alert_sound, daemon=True).start()
            else:
                fall_detected = False
                fall_start_time = None
                still_fall_start_time = None
                no_movement_start_time = None

        stframe.image(frame, channels="BGR")
        time.sleep(0.03)  # Reduce CPU usage
    
    cap.release()

if __name__ == "__main__":
    main()
