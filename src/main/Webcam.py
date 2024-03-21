import cv2
import mediapipe as mp

import sys
sys.path.append(".")
from src.constants import Constants
from src.utils import Utilities


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
    
        # calculate angles and check hand posture
    # left hand
    if results.left_hand_landmarks:
        # check wrist posture
        if results.pose_world_landmarks:
            pose_landmarks_world = results.pose_world_landmarks.landmark
    
            left_wrist_landmarks_world = []    
            for joint_index in Constants.LEFT_WRIST_JOINTS_IDX:
                left_wrist_landmarks_world.append(Utilities.get_pose_coordinates_world(pose_landmarks_world, joint_index))  
            left_wrist_angle = Utilities.get_angle_world(left_wrist_landmarks_world)
            if Utilities.check_wrist(left_wrist_angle):
                print('Fix left wrist posture.\n')
                    
        # check DIP buckling
        left_hand_landmarks = results.left_hand_landmarks.landmark
        
        # index finger
        left_index_DIP_landmarks = []
        for joint_index in Constants.INDEX_DIP_JOINTS_IDX:
            left_index_DIP_landmarks.append(Utilities.get_hand_coordinates(left_hand_landmarks, joint_index))
        left_index_DIP_angle = Utilities.get_angle(left_index_DIP_landmarks)  
        if Utilities.check_DIP_buckling(left_index_DIP_angle):
            print('Fix left index finger posture.\n')
            
        # middle finger
        left_middle_DIP_landmarks = []
        for joint_index in Constants.MIDDLE_DIP_JOINTS_IDX:
            left_middle_DIP_landmarks.append(Utilities.get_hand_coordinates(left_hand_landmarks, joint_index))
        left_middle_DIP_angle = Utilities.get_angle(left_middle_DIP_landmarks)
        if Utilities.check_DIP_buckling(left_middle_DIP_angle):
            print('Fix left middle finger posture.\n')
            
        # ring finger
        left_ring_DIP_landmarks = []
        for joint_index in Constants.RING_DIP_JOINTS_IDX:
            left_ring_DIP_landmarks.append(Utilities.get_hand_coordinates(left_hand_landmarks, joint_index))
        left_ring_DIP_angle = Utilities.get_angle(left_ring_DIP_landmarks)
        if Utilities.check_DIP_buckling(left_ring_DIP_angle):
            print('Fix left ring finger posture.\n')
            
        # pinky finger
        left_pinky_DIP_landmarks = []
        for joint_index in Constants.PINKY_DIP_JOINTS_IDX:
            left_pinky_DIP_landmarks.append(Utilities.get_hand_coordinates(left_hand_landmarks, joint_index))
        left_pinky_DIP_angle = Utilities.get_angle(left_pinky_DIP_landmarks)
        if Utilities.check_DIP_buckling(left_pinky_DIP_angle):
            print('Fix left pinky finger posture.\n')
            
    # right hand
    if results.right_hand_landmarks:
        # check wrist posture
        if results.pose_world_landmarks:
            pose_landmarks_world = results.pose_world_landmarks.landmark
    
            right_wrist_landmarks_world = []    
            for joint_index in Constants.IDX_RIGHT_WRIST_JOINTS:
                right_wrist_landmarks_world.append(Utilities.get_pose_coordinates_world(pose_landmarks_world, joint_index))  
            right_wrist_angle = Utilities.get_angle_world(right_wrist_landmarks_world)
            if Utilities.check_wrist(right_wrist_angle):
                print('Fix right wrist posture.\n')
                    
        # check DIP buckling
        right_hand_landmarks = results.right_hand_landmarks.landmark
        
        # index finger
        right_index_DIP_landmarks = []
        for joint_index in Constants.INDEX_DIP_JOINTS_IDX:
            right_index_DIP_landmarks.append(Utilities.get_hand_coordinates(right_hand_landmarks, joint_index))
        right_index_DIP_angle = Utilities.get_angle(right_index_DIP_landmarks)  
        if Utilities.check_DIP_buckling(right_index_DIP_angle):
            print('Fix right index finger posture.\n')
            
        # middle finger
        right_middle_DIP_landmarks = []
        for joint_index in Constants.MIDDLE_DIP_JOINTS_IDX:
            right_middle_DIP_landmarks.append(Utilities.get_hand_coordinates(right_hand_landmarks, joint_index))
        right_middle_DIP_angle = Utilities.get_angle(right_middle_DIP_landmarks)
        if Utilities.check_DIP_buckling(right_middle_DIP_angle):
            print('Fix right middle finger posture.\n')
            
        # ring finger
        right_ring_DIP_landmarks = []
        for joint_index in Constants.RING_DIP_JOINTS_IDX:
            right_ring_DIP_landmarks.append(Utilities.get_hand_coordinates(right_hand_landmarks, joint_index))
        right_ring_DIP_angle = Utilities.get_angle(right_ring_DIP_landmarks)
        if Utilities.check_DIP_buckling(right_ring_DIP_angle):
            print('Fix right ring finger posture.\n')
            
        # pinky finger
        right_pinky_DIP_landmarks = []
        for joint_index in Constants.PINKY_DIP_JOINTS_IDX:
            right_pinky_DIP_landmarks.append(Utilities.get_hand_coordinates(right_hand_landmarks, joint_index))
        right_pinky_DIP_angle = Utilities.get_angle(right_pinky_DIP_landmarks)
        if Utilities.check_DIP_buckling(right_pinky_DIP_angle):
            print('Fix right pinky finger posture.\n')
            
        
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