import cv2
import mediapipe as mp

import sys
sys.path.append(".")
from src.constants import Constants
from src.utils import Utilities

def check_wrist_posture(results, hand='left'):
    
    # Define the wrist joints
    wrist_joints = {
        'left': Constants.LEFT_WRIST_JOINTS_IDX,
        'right': Constants.RIGHT_WRIST_JOINTS_IDX
    }

    # Check wrist posture
    if results.pose_world_landmarks:
        pose_landmarks_world = results.pose_world_landmarks.landmark
        wrist_landmarks_world = [Utilities.get_pose_coordinates_world(pose_landmarks_world, joint_index) for joint_index in wrist_joints[hand]]
        wrist_angle = Utilities.get_angle_world(wrist_landmarks_world)
        if Utilities.is_wrist_posture_correct(wrist_angle):
            print(f'Fix {hand} wrist posture.\n')

def check_DIP_posture(results):
    # Define the finger joints
    finger_DIP_joints = {
        'index': Constants.INDEX_DIP_JOINTS_IDX,
        'middle': Constants.MIDDLE_DIP_JOINTS_IDX,
        'ring': Constants.RING_DIP_JOINTS_IDX,
        'pinky': Constants.PINKY_DIP_JOINTS_IDX,
        # Add other fingers here
    }

    # Check left hand
    if results.left_hand_landmarks:
        left_hand_landmarks = results.left_hand_landmarks.landmark
        for finger, joints in finger_DIP_joints.items():
            finger_DIP_landmarks = [Utilities.get_hand_coordinates(left_hand_landmarks, joint_index) for joint_index in joints]
            finger_DIP_angle = Utilities.get_angle(finger_DIP_landmarks)
            if Utilities.is_DIP_bent(finger_DIP_angle):
                print(f'Fix left {finger} finger posture.\n')

    # Check right hand
    if results.right_hand_landmarks:
        right_hand_landmarks = results.right_hand_landmarks.landmark
        for finger, joints in finger_DIP_joints.items():
            finger_DIP_landmarks = [Utilities.get_hand_coordinates(right_hand_landmarks, joint_index) for joint_index in joints]
            finger_DIP_angle = Utilities.get_angle(finger_DIP_landmarks)
            if Utilities.is_DIP_bent(finger_DIP_angle):
                print(f'Fix right {finger} finger posture.\n')