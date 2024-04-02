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

def draw_landmarks_test(annotated_image, results):
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
        
def draw_landmarks(annotated_image, results):
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

def draw_hand_landmarks():
    return

def highlight_incorrect_hand_posture():
    return