import math 

def get_world_coordinates(world_landmarks, point_index):
    return world_landmarks[point_index].x, world_landmarks[point_index].y, world_landmarks[point_index].z

def denormalize(point, width, height):
    return int(point[0] * width), int(point[1] * height), int(point[2] * width)

def calculate_angle(coordinates1, coordinates2, coordinates3):
    x1, y1, z1 = coordinates1
    x2, y2, z2 = coordinates2
    x3, y3, z3 = coordinates3
    
    radians = math.atan2(z3 - z2, x3 - x2) - math.atan2(z1 - z2, x1 - x2)
    
    return abs(radians)


def rad_to_deg(radians):
    return radians * 180.0 / math.pi