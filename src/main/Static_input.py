import cv2
import mediapipe as mp
import numpy as np

import os
os.chdir("./src/main")
import sys
sys.path.append("..")
from utils import utilities

# mp_drawing = mp.solutions.drawing_utils
# mp_drawing_styles = mp.solutions.drawing_styles
# mp_holistic = mp.solutions.holistic

# left_hand_point_indecies = [mp_holistic.PoseLandmark.LEFT_ELBOW, mp_holistic.PoseLandmark.LEFT_WRIST, mp_holistic.PoseLandmark.LEFT_INDEX]
# right_hand_point_indecies = [mp_holistic.PoseLandmark.RIGHT_ELBOW, mp_holistic.PoseLandmark.RIGHT_WRIST, mp_holistic.PoseLandmark.RIGHT_INDEX] 

# left_hand_world_landmarks = []
# right_hand_world_landmarks = []


# IMAGE_FILE = "/Users/fanbalance/Documents/UofT/2023-2024/Winter/ESC102/Mediapipe/src/resources/Piano_Top_View.jpeg"

# with mp_holistic.Holistic(
#         static_image_mode=True,
#         model_complexity=2,
#         enable_segmentation=True,
#         refine_face_landmarks=True) as holistic:


#     # process image
#     image = cv2.imread(IMAGE_FILE)
#     image_height, image_width, _ = image.shape
#     results = holistic.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    
#     # calculate angles
#     if results.pose_world_landmarks:
#         world_landmarks = results.pose_world_landmarks.landmark
        
#         if results.left_hand_landmarks:
#             for point_index in left_hand_point_indecies:
#                 denormalized_world_coordinates = utilities.denormalize(utilities.get_world_coordinates(world_landmarks, point_index))
#                 left_hand_world_landmarks.append(denormalized_world_coordinates)
                
#             left_hand_angle = utilities.calculate_angle(left_hand_world_landmarks[0], left_hand_world_landmarks[1], left_hand_world_landmarks[2])
#             left_hand_angle = utilities.rad_to_deg(left_hand_angle)
            
#             if left_hand_angle > 90:
#                 print("Right wrist is dipping down.\n")
                

#         if results.right_hand_landmarks:
#             for point_index in right_hand_point_indecies:
#                 denormalized_world_coordinates = utilities.denormalize(utilities.get_world_coordinates(world_landmarks, point_index))
#                 right_hand_world_landmarks.append(denormalized_world_coordinates)
                
#             right_hand_angle = utilities.calculate_angle(right_hand_world_landmarks[0], right_hand_world_landmarks[1], right_hand_world_landmarks[2])
#             right_hand_angle = utilities.rad_to_deg(right_hand_angle)
            
#             if right_hand_angle > 90:
#                 print("Right hand is dipping down.\n")
