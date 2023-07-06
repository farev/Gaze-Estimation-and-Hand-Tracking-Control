import cv2
import mediapipe as mp
import pyautogui

# Initialize Mediapipe
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Initialize PyAutoGUI
pyautogui.FAILSAFE = False

# Initialize mouse position
left_mouse_pressed = False
right_mouse_pressed = False

# Initialize gesture detection
pinch_threshold = 25
right_pinch_detected = False

# Initialize scroll gesture detection
fist = False
scroll_threshold = 5  # Adjust this value as needed
scroll_detected = False
scroll_direction = None
previous_wrist_y = None

# Webcam configuration
video_width, video_height = 640, 480

# Set up webcam
cap = cv2.VideoCapture(0) # 0 default webcam, 1 for external webacam
cap.set(3, video_width)
cap.set(4, video_height)

# Set up Mediapipe hands
with mp_hands.Hands(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    max_num_hands=1,
) as hands:
    while cap.isOpened():
        ret, frame = cap.read()

        # Flip the frame horizontally for a mirror effect
        frame = cv2.flip(frame, 1)

        # Convert the frame to RGB for Mediapipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame with Mediapipe
        results = hands.process(frame_rgb)

        # Draw hand landmarks on the frame
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2),
                )

                # Get the coordinates of the index and thumb fingertips
                index_fingertip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                middle_fingertip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
                thumb_fingertip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]


                # Convert the fingertip coordinates to pixel values
                index_x = int(index_fingertip.x * video_width)
                index_y = int(index_fingertip.y * video_height)
                middle_x = int(middle_fingertip.x * video_width)
                middle_y = int(middle_fingertip.y * video_height)
                thumb_x = int(thumb_fingertip.x * video_width)
                thumb_y = int(thumb_fingertip.y * video_height)
                wrist_y = int(wrist.y * video_height)


                # Calculate the distance between the fingertips
                index_distance = ((index_x - thumb_x) ** 2 + (index_y - thumb_y) ** 2) ** 0.5
                middle_distance = ((middle_x - thumb_x) ** 2 + (middle_y - thumb_y) ** 2) ** 0.5

                # Detect scroll gesture based on wrist position
                if index_distance <= pinch_threshold and middle_distance <= pinch_threshold:
                    fist = True
                    print('Fist')
                    if previous_wrist_y is not None:
                        wrist_movement = wrist_y - previous_wrist_y
                        if abs(wrist_movement) > scroll_threshold:
                            if not scroll_detected:
                                scroll_detected = True
                                scroll_direction = 'up' if wrist_movement < 0 else 'down'
                                if scroll_direction == 'up':
                                    pyautogui.scroll(100)
                                else:
                                    pyautogui.scroll(-100)
                    previous_wrist_y = wrist_y
                else:
                    fist = False
                    previous_wrist_y = None
                    scroll_detected = False

                # If the fingertips are close, trigger a mouse click
                # Left-click
                if index_distance < pinch_threshold and not fist:
                    pyautogui.mouseDown(button='left')

                else:
                    pyautogui.mouseUp(button='left')

                # Right-click
                if middle_distance < pinch_threshold and not fist:
                    if not right_mouse_pressed:
                        # Trigger a right-click
                        pyautogui.mouseUp(button='right')
                        right_mouse_pressed = True
                else:
                    if right_mouse_pressed:
                        # Release the right-click
                        pyautogui.mouseDown(button='left')
                        right_mouse_pressed = False

        else:
            pyautogui.mouseUp(button='left')
            #pyautogui.mouseUp(button='right')
            
        # Display the frame
        cv2.imshow("Hand Tracking", frame)

        # Quit the program if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()
