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
        is_posture_correct = Utilities.is_wrist_posture_correct(wrist_angle)
        print(f'{hand} wrist angle: {wrist_angle:.2f} degrees. {is_posture_correct}\n')
        # if not is_posture_correct:
        #     print(f'{hand} wrist posture is incorrect.\n')
        return wrist_angle, is_posture_correct
            

            

def check_DIP_posture(results):
    
    # Define the finger joints
    finger_DIP_joints = {
        'index': Constants.INDEX_DIP_JOINTS_IDX,
        'middle': Constants.MIDDLE_DIP_JOINTS_IDX,
        'ring': Constants.RING_DIP_JOINTS_IDX,
        'pinky': Constants.PINKY_DIP_JOINTS_IDX,
        # Add other fingers here
    }
    
    left_hand_results = {
        'index': True,
        'middle': True,
        'ring': True,
        'pinky': True,
    }
    right_hand_results = {
        'index': True,
        'middle': True,
        'ring': True,
        'pinky': True,
    }

    # Check left hand
    if results.left_hand_landmarks:
        left_hand_landmarks = results.left_hand_landmarks.landmark
        for finger, joints in finger_DIP_joints.items():
            finger_DIP_landmarks = [Utilities.get_hand_coordinates(left_hand_landmarks, joint_index) for joint_index in joints]
            finger_DIP_angle = Utilities.get_angle(finger_DIP_landmarks)
            is_left_DIP_correct = Utilities.is_DIP_bent(finger_DIP_angle)
            # print(f'left {finger} finger DIP angle: {finger_DIP_angle:.2f} degrees. {is_left_DIP_correct}\n')
            if not is_left_DIP_correct:
                left_hand_results[finger] = False

    # Check right hand
    if results.right_hand_landmarks:
        right_hand_landmarks = results.right_hand_landmarks.landmark
        for finger, joints in finger_DIP_joints.items():
            finger_DIP_landmarks = [Utilities.get_hand_coordinates(right_hand_landmarks, joint_index) for joint_index in joints]
            finger_DIP_angle = Utilities.get_angle(finger_DIP_landmarks)
            is_right_DIP_correct = Utilities.is_DIP_bent(finger_DIP_angle)
            # print(f'right {finger} finger DIP angle: {finger_DIP_angle:.2f} degrees. {is_right_DIP_correct}\n')
            if not is_right_DIP_correct:
                right_hand_results[finger] = False
                
    return left_hand_results, right_hand_results
                
                

                