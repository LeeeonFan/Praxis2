import numpy as np

import sys
sys.path.append('.')
from src.constants import Constants

def get_pose_coordinates_world(pose_landmarks_world, joint_index):
    return pose_landmarks_world[joint_index].x, pose_landmarks_world[joint_index].y, pose_landmarks_world[joint_index].z

def get_hand_coordinates(hand_landmarks, joint_index):
    return hand_landmarks[joint_index].x, hand_landmarks[joint_index].y, hand_landmarks[joint_index].z
    

def check_wrist(wrist_angle):
    return wrist_angle > min(Constants.MAX_WRIST_EXTENSION_ANGLE, Constants.MAX_WRIST_FLEXION_ANGLE)
    
def check_DIP_buckling(DIP_angle):
    return DIP_angle > Constants.MAX_DIP_ANGLE

# This function returns the angle in degrees. 
def get_angle(lanmarks):
    endpoint1_coordinates = lanmarks[0]
    vertex_coordinates = lanmarks[1]
    endpoint2_coordinates = lanmarks[2]
    
    vector1 = np.array([vertex_coordinates[0] - endpoint1_coordinates[0], vertex_coordinates[1] - endpoint1_coordinates[1]])
    vector2 = np.array([endpoint2_coordinates[0] - vertex_coordinates[0], endpoint2_coordinates[1] - vertex_coordinates[1]])
    
    # Normalize the vectors
    vector1 = vector1 / np.linalg.norm(vector1)
    vector2 = vector2 / np.linalg.norm(vector2)

    # Calculate the dot products
    dot_product = np.dot(vector1, vector2)

    # Calculate the angle in degrees
    radians = np.arccos(dot_product)

    # Convert the angle to degrees
    angle = np.rad2deg(radians)

    return angle

# This function returns the angle in degrees. 
def get_angle_world(lanmarks_world):
    endpoint1_coordinates = lanmarks_world[0]
    vertex_coordinates = lanmarks_world[1]
    endpoint2_coordinates = lanmarks_world[2]
    
    vector1 = np.array([vertex_coordinates[0] - endpoint1_coordinates[0], vertex_coordinates[1] - endpoint1_coordinates[1], vertex_coordinates[2] - endpoint1_coordinates[2]])
    vector2 = np.array([endpoint2_coordinates[0] - vertex_coordinates[0], endpoint2_coordinates[1] - vertex_coordinates[1], endpoint2_coordinates[2] - vertex_coordinates[2]])
    
    # Normalize the vectors
    vector1 = vector1 / np.linalg.norm(vector1)
    vector2 = vector2 / np.linalg.norm(vector2)

    # Calculate the dot products
    dot_product = np.dot(vector1, vector2)

    # Calculate the angle in degrees
    radians = np.arccos(dot_product)

    # Convert the angle to degrees
    angle = np.rad2deg(radians)

    return angle

def denormalize_world_coordinates(joint_coordinates, width, height):
    return joint_coordinates[0] * width, joint_coordinates[1] * height, joint_coordinates[2] * width