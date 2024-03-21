import cv2
import mediapipe as mp

import Checker


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

cap = cv2.VideoCapture(0)
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
    
    # check hand posture
    if results.left_hand_landmarks:
        Checker.check_wrist_posture(results, hand='left')
    if results.right_hand_landmarks:
        Checker.check_wrist_posture(results, hand='right')
    Checker.check_DIP_posture(results)
            
        
    # draw landmarks
    annotated_image = image.copy()
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            annotated_image,
            results.left_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            annotated_image,
            results.right_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())


    # display annotated image, click esc to exit
    cv2.imshow("Annotated Image", annotated_image)
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()