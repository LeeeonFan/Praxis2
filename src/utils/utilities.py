import numpy as np

import sys
sys.path.append('.')
from src.constants import Constants

def get_world_coordinates(world_landmarks, joint_index):
    return world_landmarks[joint_index].x, world_landmarks[joint_index].y, world_landmarks[joint_index].z
    
# This function returns the angle in degrees. 
def get_angle(world_lanmarks):
    endpoint1_coordinates = world_lanmarks[0]
    vertex_coordinates = world_lanmarks[1]
    endpoint2_coordinates = world_lanmarks[2]
    
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

def check_wrist_extension(wrist_landmarks_world, left_wrist_angle):
    wrist = wrist_landmarks_world[1]
    index = wrist_landmarks_world[2]
    
    extension = index[3] - wrist[3] > Constants.EPSILON 
    if left_wrist_angle > WRIST_EXTENSTION_THRESHOLD:
        return True
    else:
        return False


def rad_to_deg(radians):
    return np.rad2deg(radians)


def denormalize(joint_coordinates, width, height):
    return joint_coordinates[0] * width, joint_coordinates[1] * height, joint_coordinates[2] * width