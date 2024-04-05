import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import numpy as np


import Checker

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

# Change this line to the path of your video file
video_path = "/path/to/your/video.mp4"

cap = cv2.VideoCapture(video_path)
left_angles = []
right_angles = []
incorrect_points_left = []
incorrect_points_right = []
frame_count = 0
with mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as holistic:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      break

    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = holistic.process(image)

    if results.left_hand_landmarks:
        left_wrist_angle, is_left_posture_correct = Checker.check_wrist_posture(results, hand='left')
        left_angles.append(left_wrist_angle)
        if not is_left_posture_correct:
            incorrect_points_left.append((frame_count, left_wrist_angle))
    else:
        left_angles.append(None)
        
    if results.right_hand_landmarks:
        right_wrist_angle, is_right_posture_correct = Checker.check_wrist_posture(results, hand='right')
        right_angles.append(right_wrist_angle)
        if not is_right_posture_correct:
            incorrect_points_right.append((frame_count, right_wrist_angle))
    else:
        right_angles.append(None)
        
    frame_count += 1

cap.release()

# Plotting
plt.figure(figsize=(10, 5))
plt.plot(left_angles, label='Left wrist angle')
plt.plot(right_angles, label='Right wrist angle')

incorrect_points_left = np.array(incorrect_points_left)
if incorrect_points_left.size > 0:
    plt.scatter(incorrect_points_left[:, 0], incorrect_points_left[:, 1], color='red')

incorrect_points_right = np.array(incorrect_points_right)
if incorrect_points_right.size > 0:
    plt.scatter(incorrect_points_right[:, 0], incorrect_points_right[:, 1], color='blue')

plt.legend()
plt.show()