import cv2
import mediapipe as mp
import numpy as np

import sys
sys.path.append(".")
from src.constants import Constants
from src.utils import Utilities

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic


image_file = "/Users/fanbalance/Documents/UofT/2023-2024/Winter/ESC102/Hand_Posture_Detection/src/resources/static_test/static_test_side_view1.jpg"

with mp_holistic.Holistic(
        static_image_mode=True,
        model_complexity=2,
        enable_segmentation=True,
        refine_face_landmarks=True) as holistic:


    # process image
    image = cv2.imread(image_file)
    image_height, image_width, _ = image.shape
    results = holistic.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    
    # calculate angles and check hand posture
    # left hand
    if results.left_hand_landmarks:
        # check wrist posture
        if results.pose_world_landmarks:
            pose_landmarks_world = results.pose_world_landmarks.landmark
    
            left_wrist_landmarks_world = []    
            for joint_index in Constants.IDX_LEFT_WRIST_JOINTS:
                left_wrist_landmarks_world.append(Utilities.get_pose_coordinates_world(pose_landmarks_world, joint_index))  
            left_wrist_angle = Utilities.get_angle_world(left_wrist_landmarks_world)
            if Utilities.check_wrist(left_wrist_landmarks_world, left_wrist_angle):
                print('Fix left wrist posture.\n')
                    
        # check DIP buckling
        left_hand_landmarks = results.left_hand_landmarks.landmark
        
        # index finger
        left_index_DIP_landmarks = []
        for joint_index in Constants.INDEX_DIP_JOINTS_IDX:
            left_index_DIP_landmarks.append(Utilities.get_hand_coordinates(left_hand_landmarks, joint_index))
        left_index_DIP_angle = Utilities.get_angle(left_index_DIP_landmarks)  
        if Utilities.check_finger_buckling(left_index_DIP_angle):
            print('Fix left index finger posture.\n')
            
        # middle finger
        left_middle_DIP_landmarks = []
        for joint_index in Constants.MIDDLE_DIP_JOINTS_IDX:
            left_middle_DIP_landmarks.append(Utilities.get_hand_coordinates(left_hand_landmarks, joint_index))
        left_middle_DIP_angle = Utilities.get_angle(left_middle_DIP_landmarks)
        if Utilities.check_finger_buckling(left_middle_DIP_angle):
            print('Fix left middle finger posture.\n')
            
        # ring finger
        left_ring_DIP_landmarks = []
        for joint_index in Constants.RING_DIP_JOINTS_IDX:
            left_ring_DIP_landmarks.append(Utilities.get_hand_coordinates(left_hand_landmarks, joint_index))
        left_ring_DIP_angle = Utilities.get_angle(left_ring_DIP_landmarks)
        if Utilities.check_finger_buckling(left_ring_DIP_angle):
            print('Fix left ring finger posture.\n')
            
        # pinky finger
        left_pinky_DIP_landmarks = []
        for joint_index in Constants.PINKY_DIP_JOINTS_IDX:
            left_pinky_DIP_landmarks.append(Utilities.get_hand_coordinates(left_hand_landmarks, joint_index))
        left_pinky_DIP_angle = Utilities.get_angle(left_pinky_DIP_landmarks)
        if Utilities.check_finger_buckling(left_pinky_DIP_angle):
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
            if Utilities.check_wrist(right_wrist_landmarks_world, right_wrist_angle):
                print('Fix right wrist posture.\n')
                    
        # check DIP buckling
        right_hand_landmarks = results.right_hand_landmarks.landmark
        
        # index finger
        right_index_DIP_landmarks = []
        for joint_index in Constants.INDEX_DIP_JOINTS_IDX:
            right_index_DIP_landmarks.append(Utilities.get_hand_coordinates(right_hand_landmarks, joint_index))
        right_index_DIP_angle = Utilities.get_angle(right_index_DIP_landmarks)  
        if Utilities.check_finger_buckling(right_index_DIP_angle):
            print('Fix right index finger posture.\n')
            
        # middle finger
        right_middle_DIP_landmarks = []
        for joint_index in Constants.MIDDLE_DIP_JOINTS_IDX:
            right_middle_DIP_landmarks.append(Utilities.get_hand_coordinates(right_hand_landmarks, joint_index))
        right_middle_DIP_angle = Utilities.get_angle(right_middle_DIP_landmarks)
        if Utilities.check_finger_buckling(right_middle_DIP_angle):
            print('Fix right middle finger posture.\n')
            
        # ring finger
        right_ring_DIP_landmarks = []
        for joint_index in Constants.RING_DIP_JOINTS_IDX:
            right_ring_DIP_landmarks.append(Utilities.get_hand_coordinates(right_hand_landmarks, joint_index))
        right_ring_DIP_angle = Utilities.get_angle(right_ring_DIP_landmarks)
        if Utilities.check_finger_buckling(right_ring_DIP_angle):
            print('Fix right ring finger posture.\n')
            
        # pinky finger
        right_pinky_DIP_landmarks = []
        for joint_index in Constants.PINKY_DIP_JOINTS_IDX:
            right_pinky_DIP_landmarks.append(Utilities.get_hand_coordinates(right_hand_landmarks, joint_index))
        right_pinky_DIP_angle = Utilities.get_angle(right_pinky_DIP_landmarks)
        if Utilities.check_finger_buckling(right_pinky_DIP_landmarks):
            print('Fix right pinky finger posture.\n')
            
        
            
            
        
        
        
        
        
        
        

            
    
                



