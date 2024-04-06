import cv2
import mediapipe as mp

import Checker
import Illustrater


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

    # check hand posture and get feedbackx
    if results.left_hand_landmarks:
        left_wrist_angle, is_left_wrist_posture_correct = Checker.check_wrist_posture(results, hand='left')
    if results.right_hand_landmarks:
        right_wrist_angle, is_right_wrist_posture_correct = Checker.check_wrist_posture(results, hand='right')
    left_hand_results, right_hand_results = Checker.check_DIP_posture(results)
    
        
    # draw landmarks
    annotated_image = image.copy()
    Illustrater.draw_elbow_wrist_fingertip_landmarks(annotated_image, results, is_left_wrist_posture_correct, is_right_wrist_posture_correct)
    # Illustrater.draw_hand_landmarks(annotated_image, results)
    Illustrater.draw_hand_landmarks_test(annotated_image, results, left_hand_results, right_hand_results)
    


    # display annotated image
    cv2.imshow("Annotated Image", annotated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

