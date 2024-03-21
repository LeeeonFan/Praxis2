import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

# Left hand joints
LEFT_WRIST_JOINTS_IDX = [mp_holistic.PoseLandmark.LEFT_ELBOW, mp_holistic.PoseLandmark.LEFT_WRIST, mp_holistic.PoseLandmark.LEFT_INDEX]

# Right hand joints
IDX_RIGHT_WRIST_JOINTS = [mp_holistic.PoseLandmark.RIGHT_ELBOW, mp_holistic.PoseLandmark.RIGHT_WRIST, mp_holistic.PoseLandmark.RIGHT_INDEX] 

# DIP joints
INDEX_DIP_JOINTS_IDX = [mp.holistic.HandLandmark.INDEX_FINGER_PIP, mp.holistic.HandLandmark.INDEX_FINGER_DIP, mp.holistic.HandLandmark.INDEX_FINGER_TIP]
MIDDLE_DIP_JOINTS_IDX = [mp.holistic.HandLandmark.MIDDLE_FINGER_PIP, mp.holistic.HandLandmark.MIDDLE_FINGER_DIP, mp.holistic.HandLandmark.MIDDLE_FINGER_TIP]
RING_DIP_JOINTS_IDX = [mp.holistic.HandLandmark.RING_FINGER_PIP, mp.holistic.HandLandmark.RING_FINGER_DIP, mp.holistic.HandLandmark.RING_FINGER_TIP]
PINKY_DIP_JOINTS_IDX = [mp.holistic.HandLandmark.PINKY_PIP, mp.holistic.HandLandmark.PINKY_DIP, mp.holistic.HandLandmark.PINKY_TIP]

# Wrist posture
MAX_WRIST_EXTENSION_ANGLE = 0
MAX_WRIST_FLEXION_ANGLE = 0
EPSILON_Z = 0

# Finger posture
MAX_DIP_ANGLE = 0