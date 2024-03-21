import cv2
import mediapipe as mp

import Checker

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic


image_file = "/Users/fanbalance/Documents/UofT/2023-2024/Winter/ESC102/Hand_Posture_Detection/src/resources/static_test/static_test_side_view1.jpg"

with mp_holistic.Holistic(
        static_image_mode=True,
        model_complexity=2,
        enable_segmentation=True,
        refine_face_landmarks=True) as holistic:


    # process image
    image = cv2.imread(image_file)
    image_height, image_width, _ = image.shape
    results = holistic.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # check hand posture
    if results.left_hand_landmarks:
        Checker.check_wrist_posture(results, hand='left')
    if results.right_hand_landmarks:
        Checker.check_wrist_posture(results, hand='right')
    Checker.check_DIP_posture(results)
    
        
    # draw landmarks
    annotated_image = image.copy()
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            annotated_image,
            results.left_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            annotated_image,
            results.right_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())

    # display annotated image
    cv2.imshow("Annotated Image", annotated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

