def get_world_coordinates(world_landmarks, point_index):
    return world_landmarks[point_index].x, world_landmarks[point_index].y, world_landmarks[point_index].z

def denormalize(point, width, height):
    point[0] = int(point[0] * width)
    point[1] = int(point[1] * height)
    return point

