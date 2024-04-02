from typing import Mapping, Tuple

import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands_connections = mp.solutions.hands_connections
mp_holistic = mp.solutions.holistic

# Left hand joints
LEFT_WRIST_JOINTS_IDX = [mp_holistic.PoseLandmark.LEFT_ELBOW, mp_holistic.PoseLandmark.LEFT_WRIST, mp_holistic.PoseLandmark.LEFT_INDEX]

# Right hand joints
RIGHT_WRIST_JOINTS_IDX = [mp_holistic.PoseLandmark.RIGHT_ELBOW, mp_holistic.PoseLandmark.RIGHT_WRIST, mp_holistic.PoseLandmark.RIGHT_INDEX] 

# DIP joints
INDEX_DIP_JOINTS_IDX = [mp_holistic.HandLandmark.INDEX_FINGER_PIP, mp_holistic.HandLandmark.INDEX_FINGER_DIP, mp_holistic.HandLandmark.INDEX_FINGER_TIP]
MIDDLE_DIP_JOINTS_IDX = [mp_holistic.HandLandmark.MIDDLE_FINGER_PIP, mp_holistic.HandLandmark.MIDDLE_FINGER_DIP, mp_holistic.HandLandmark.MIDDLE_FINGER_TIP]
RING_DIP_JOINTS_IDX = [mp_holistic.HandLandmark.RING_FINGER_PIP, mp_holistic.HandLandmark.RING_FINGER_DIP, mp_holistic.HandLandmark.RING_FINGER_TIP]
PINKY_DIP_JOINTS_IDX = [mp_holistic.HandLandmark.PINKY_PIP, mp_holistic.HandLandmark.PINKY_DIP, mp_holistic.HandLandmark.PINKY_TIP]

# Wrist posture (unit: degree, meter)
MAX_WRIST_EXTENSION_ANGLE = 5
MAX_WRIST_FLEXION_ANGLE = 26.6
EPSILON_Z = 0.1

# Finger posture
MAX_DIP_ANGLE = 30

# Drawing styles for hand landmarks

NORMAL_HAND_LANDMARK_STYLE = {
    mp_drawing_styles._PALM_LANDMARKS:
        mp_drawing.DrawingSpec(
            color=mp_drawing_styles._WHITE, thickness=mp_drawing_styles._THICKNESS_DOT, circle_radius=mp_drawing_styles._RADIUS),
    mp_drawing_styles._THUMP_LANDMARKS:
        mp_drawing.DrawingSpec(
            color=mp_drawing_styles._WHITE, thickness=mp_drawing_styles._THICKNESS_DOT, circle_radius=mp_drawing_styles._RADIUS),
    mp_drawing_styles._INDEX_FINGER_LANDMARKS:
        mp_drawing.DrawingSpec(
            color=mp_drawing_styles._WHITE, thickness=mp_drawing_styles._THICKNESS_DOT, circle_radius=mp_drawing_styles._RADIUS),
    mp_drawing_styles._MIDDLE_FINGER_LANDMARKS:
        mp_drawing.DrawingSpec(
            color=mp_drawing_styles._WHITE, thickness=mp_drawing_styles._THICKNESS_DOT, circle_radius=mp_drawing_styles._RADIUS),
    mp_drawing_styles._RING_FINGER_LANDMARKS:
        mp_drawing.DrawingSpec(
            color=mp_drawing_styles._WHITE, thickness=mp_drawing_styles._THICKNESS_DOT, circle_radius=mp_drawing_styles._RADIUS),
    mp_drawing_styles._PINKY_FINGER_LANDMARKS:
        mp_drawing.DrawingSpec(
            color=mp_drawing_styles._WHITE, thickness=mp_drawing_styles._THICKNESS_DOT, circle_radius=mp_drawing_styles._RADIUS),
}


# Drawing styles for hand connections

NORMAL_HAND_CONNECTION_STYLE = {
    mp_hands_connections.HAND_PALM_CONNECTIONS:
        mp_drawing.DrawingSpec(color=mp_drawing_styles._WHITE, thickness=mp_drawing_styles._THICKNESS_WRIST_MCP),
    mp_hands_connections.HAND_THUMB_CONNECTIONS:
        mp_drawing.DrawingSpec(color=mp_drawing_styles._WHITE, thickness=mp_drawing_styles._THICKNESS_FINGER),
    mp_hands_connections.HAND_INDEX_FINGER_CONNECTIONS:
        mp_drawing.DrawingSpec(color=mp_drawing_styles._WHITE, thickness=mp_drawing_styles._THICKNESS_FINGER),
    mp_hands_connections.HAND_MIDDLE_FINGER_CONNECTIONS:
        mp_drawing.DrawingSpec(color=mp_drawing_styles._WHITE, thickness=mp_drawing_styles._THICKNESS_FINGER),
    mp_hands_connections.HAND_RING_FINGER_CONNECTIONS:
        mp_drawing.DrawingSpec(color=mp_drawing_styles._WHITE, thickness=mp_drawing_styles._THICKNESS_FINGER),
    mp_hands_connections.HAND_PINKY_FINGER_CONNECTIONS:
        mp_drawing.DrawingSpec(color=mp_drawing_styles._WHITE, thickness=mp_drawing_styles._THICKNESS_FINGER)
}

def get_normal_hand_landmarks_style() -> Mapping[int, mp.solutions.drawing_utils.DrawingSpec]:
  hand_landmark_style = {}
  for k, v in NORMAL_HAND_LANDMARK_STYLE.items():
    for landmark in k:
      hand_landmark_style[landmark] = v
  return hand_landmark_style

def get_normal_hand_connections_style() -> Mapping[Tuple[int, int], mp.solutions.drawing_utils.DrawingSpec]:
  hand_connection_style = {}
  for k, v in NORMAL_HAND_CONNECTION_STYLE.items():
    for connection in k:
      hand_connection_style[connection] = v
  return hand_connection_style

