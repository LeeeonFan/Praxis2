import cv2
import mediapipe as mp
import numpy as np

import sys
sys.path.append(".")
from src.utils import Constants
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
    if results.pose_world_landmarks:
        world_landmarks = results.pose_world_landmarks.landmark
        
        # left wrist
        if results.left_hand_landmarks:
            left_wrist_landmarks_world = []
            
            for joint_index in Constants.IDX_LEFT_WRIST:
                left_wrist_landmarks_world.append(Utilities.get_world_coordinates(world_landmarks, joint_index))  
                  
            left_wrist_angle = Utilities.get_angle(left_wrist_landmarks_world)
            
            if Utilities.check_wrist_extension(left_wrist_landmarks_world, left_wrist_angle):
                print("Left wrist is dipping down.\n")
        
            
            # if is_extension(left_hand_angle, THRESHOLD):
            #     print("Left wrist is dipping down.\n")
                



