import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import numpy as np


import Checker

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

# Change this line to the path of your video file
video_path = "/Users/fanbalance/Documents/UofT/2023-2024/Winter/ESC102/Hand_Posture_Detection/src/resources/test_1_left.mov"

cap = cv2.VideoCapture(video_path)
left_angles = []
right_angles = []
incorrect_points_left = []
incorrect_points_right = []
frame_counter = 0
frame_skip = 29  # Change this to the number of frames you want to skip
with mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as holistic:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      break
  
    # Skip frames
    frame_counter += 1
    if frame_counter % frame_skip == 0:

        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = holistic.process(image)

        if results.left_hand_landmarks:
            left_wrist_angle, is_left_posture_correct = Checker.check_wrist_posture(results, hand='left')
            left_angles.append(left_wrist_angle)
            if not is_left_posture_correct:
                incorrect_points_left.append((frame_counter, left_wrist_angle))
        else:
            left_angles.append(None)
            
        if results.right_hand_landmarks:
            right_wrist_angle, is_right_posture_correct = Checker.check_wrist_posture(results, hand='right')
            right_angles.append(right_wrist_angle)
            if not is_right_posture_correct:
                incorrect_points_right.append((frame_counter, right_wrist_angle))
        else:
            right_angles.append(None)
            
    frame_counter += 1

cap.release()

# Plotting
plt.figure(figsize=(10, 5))
plt.plot(left_angles, label='Left wrist angle')
plt.plot(right_angles, label='Right wrist angle')

# incorrect_points_left = np.array(incorrect_points_left)
# if incorrect_points_left.size > 0:
#     plt.scatter(incorrect_points_left[:, 0], incorrect_points_left[:, 1], color='red')

# incorrect_points_right = np.array(incorrect_points_right)
# if incorrect_points_right.size > 0:
#     plt.scatter(incorrect_points_right[:, 0], incorrect_points_right[:, 1], color='blue')

plt.legend()
plt.show()