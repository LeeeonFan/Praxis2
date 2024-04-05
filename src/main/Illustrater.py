import cv2
import mediapipe as mp

import sys
sys.path.append(".")
from src.constants import Constants

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

def set_landmarks_styles():
    return 
def set_connections_styles():
    return 

def draw_hand_landmarks(annotated_image, results):
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            annotated_image,
            results.left_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            Constants.get_normal_hand_landmarks_style(),
            Constants.get_normal_hand_connections_style())
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            annotated_image,
            results.right_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            Constants.get_normal_hand_landmarks_style(),
            Constants.get_normal_hand_connections_style())
        



        
        
def draw_elbow_wrist_fingertip_landmarks(image, results,left_result=True,right_result=True):
    mp_holistic = mp.solutions.holistic

    image_height, image_width, _ = image.shape
    
    if left_result:
        color_left = Constants.BLUE_BGR
    else:
        color_left = Constants.RED_BGR
    
    if right_result:
        color_right = Constants.BLUE_BGR
    else:
        color_right = Constants.RED_BGR

    if results.pose_landmarks:
        # Draw the elbow-wrist-fingertip landmarks and connections for the left hand
        if results.left_hand_landmarks:
            elbow = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_ELBOW]
            wrist = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_WRIST]
            fingertip = results.left_hand_landmarks.landmark[mp_holistic.HandLandmark.INDEX_FINGER_TIP]
            cv2.line(image, (int(elbow.x * image_width), int(elbow.y * image_height)), (int(wrist.x * image_width), int(wrist.y * image_height)), color_left, 2)
            cv2.line(image, (int(wrist.x * image_width), int(wrist.y * image_height)), (int(fingertip.x * image_width), int(fingertip.y * image_height)), color_left, 2)

        # Draw the elbow-wrist-fingertip landmarks and connections for the right hand
        if results.right_hand_landmarks:
            elbow = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_ELBOW]
            wrist = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_WRIST]
            fingertip = results.right_hand_landmarks.landmark[mp_holistic.HandLandmark.INDEX_FINGER_TIP]
            cv2.line(image, (int(elbow.x * image_width), int(elbow.y * image_height)), (int(wrist.x * image_width), int(wrist.y * image_height)), color_right, 2)
            cv2.line(image, (int(wrist.x * image_width), int(wrist.y * image_height)), (int(fingertip.x * image_width), int(fingertip.y * image_height)), color_right, 2)
        
    