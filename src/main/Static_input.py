import cv2
import mediapipe as mp
import numpy as np
from src.utils import utilities

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

left_hand_point_indecies = [mp_holistic.PoseLandmark.LEFT_ELBOW, mp_holistic.PoseLandmark.LEFT_WRIST, mp_holistic.PoseLandmark.LEFT_INDEX]
right_hand_point_indecies = [mp_holistic.PoseLandmark.RIGHT_ELBOW, mp_holistic.PoseLandmark.RIGHT_WRIST, mp_holistic.PoseLandmark.RIGHT_INDEX] 

IMAGE_FILE = ""

with mp_holistic.Holistic(
        static_image_mode=True,
        model_complexity=2,
        enable_segmentation=True,
        refine_face_landmarks=True) as holistic:

    image = cv2.imread(IMAGE_FILE)
    image_height, image_width, _ = image.shape
    results = holistic.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
    
    if results.pose_world_landmarks:
        if results.left_hand_landmarks:
            for point_index in left_hand_point_indecies:
                left_elbow = results.pose_world_landmarks.landmark[point_index]
                
                
                
        if results.right_hand_landmarks:
            for point_index in right_hand_point_indecies:
                
        
        