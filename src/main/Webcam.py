import cv2
import mediapipe as mp

import Checker
import Illustrater


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

# initial values
is_left_wrist_posture_correct = True
is_right_wrist_posture_correct = True

cap = cv2.VideoCapture(0)
frame_counter = 0
frame_skip = 23  # Change this to the number of frames you want to skip
with mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as holistic:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue
    


    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = holistic.process(image)
    
    
    # Skip frames
    frame_counter += 1
    if frame_counter % frame_skip == 0:
      
      # check hand posture and get feedback
      if results.left_hand_landmarks:
          left_wrist_angle, is_left_wrist_posture_correct = Checker.check_wrist_posture(results, hand='left')
      if results.right_hand_landmarks:
          right_wrist_angle, is_right_wrist_posture_correct = Checker.check_wrist_posture(results, hand='right')
      # Checker.check_DIP_posture(results)


    # draw elbow, wrist, fingertip landmarks
    annotated_image = image.copy()
    Illustrater.draw_elbow_wrist_fingertip_landmarks(annotated_image, results, is_left_wrist_posture_correct, is_right_wrist_posture_correct)
    # Illustrater.draw_hand_landmarks(annotated_image, results)


    # display annotated image, click esc to exit
    cv2.imshow("Annotated Image", annotated_image)
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()