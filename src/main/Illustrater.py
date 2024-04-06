import cv2
import mediapipe as mp
import copy

import sys
sys.path.append(".")
from src.constants import Constants
from src.utils import Utilities

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

# initial value 
default_left_hand_results = {
    'index': True,
    'middle': True,
    'ring': True,
    'pinky': True,
}
default_right_hand_results = {
    'index': True,
    'middle': True,
    'ring': True,
    'pinky': True,
}
    
def set_landmark_style(left_hand_results=default_left_hand_results, right_hand_results=default_right_hand_results):
    adjusted_hand_landmark_style = copy.deepcopy(Constants.NORMAL_HAND_LANDMARK_STYLE)
    
    if not left_hand_results['index']:
        adjusted_hand_landmark_style[mp_holistic.HandLandmark.INDEX_FINGER_TIP] = mp_drawing.DrawingSpec(
            color=Constants.MP_RED, thickness=mp_drawing_styles._THICKNESS_FINGER, circle_radius=Constants.RADIUS)
    if not left_hand_results['middle']:
        adjusted_hand_landmark_style[mp_holistic.HandLandmark.MIDDLE_FINGER_TIP] = mp_drawing.DrawingSpec(
            color=Constants.MP_RED, thickness=mp_drawing_styles._THICKNESS_FINGER, circle_radius=Constants.RADIUS)
    if not left_hand_results['ring']:
        adjusted_hand_landmark_style[mp_holistic.HandLandmark.RING_FINGER_TIP] = mp_drawing.DrawingSpec(
            color=Constants.MP_RED, thickness=mp_drawing_styles._THICKNESS_FINGER, circle_radius=Constants.RADIUS)
    if not left_hand_results['pinky']:
        adjusted_hand_landmark_style[mp_holistic.HandLandmark.PINKY_TIP] = mp_drawing.DrawingSpec(
            color=Constants.MP_RED, thickness=mp_drawing_styles._THICKNESS_FINGER, circle_radius=Constants.RADIUS)
        
    return adjusted_hand_landmark_style


def set_connection_style(left_hand_results=default_left_hand_results, right_hand_results=default_right_hand_results):
    adjusted_hand_connection_style = copy.deepcopy(Constants.NORMAL_HAND_CONNECTION_STYLE)
    
    if not left_hand_results['index']:
        adjusted_hand_connection_style[mp_holistic.HandLandmark.INDEX_FINGER_TIP] = mp_drawing.DrawingSpec(
            color=Constants.MP_RED, thickness=mp_drawing_styles._THICKNESS_FINGER)
    if not left_hand_results['middle']:
        adjusted_hand_connection_style[mp_holistic.HandLandmark.MIDDLE_FINGER_TIP] = mp_drawing.DrawingSpec(
            color=Constants.MP_RED, thickness=mp_drawing_styles._THICKNESS_FINGER)
    if not left_hand_results['ring']:
        adjusted_hand_connection_style[mp_holistic.HandLandmark.RING_FINGER_TIP] = mp_drawing.DrawingSpec(
            color=Constants.MP_RED, thickness=mp_drawing_styles._THICKNESS_FINGER)
    if not left_hand_results['pinky']:
        adjusted_hand_connection_style[mp_holistic.HandLandmark.PINKY_TIP] = mp_drawing.DrawingSpec(
            color=Constants.MP_RED, thickness=mp_drawing_styles._THICKNESS_FINGER)
    
    return adjusted_hand_connection_style


def draw_hand_landmarks_test(annotated_image, results, left_hand_results=default_left_hand_results, right_hand_results=default_right_hand_results):
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            annotated_image,
            results.left_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            Utilities.get_hand_landmarks_style(set_connection_style(left_hand_results, right_hand_results)),
            Utilities.get_hand_connections_style(set_connection_style(left_hand_results, right_hand_results)))
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            annotated_image,
            results.right_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            Utilities.get_hand_landmarks_style(set_connection_style(left_hand_results, right_hand_results)),
            Utilities.get_hand_connections_style(set_connection_style(left_hand_results, right_hand_results)))
        
        
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
        
    