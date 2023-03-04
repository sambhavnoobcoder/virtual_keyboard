import cv2
import mediapipe as mp
import pyautogui

# Define the positions and keys of the virtual keyboard
positions = [(100, 100), (200, 100), (300, 100), (400, 100),(500, 100), (600, 100), (700, 100), (800, 100),(900,100),(1000,100),(100, 200), (200, 200), (300, 200), (400, 200),(500, 200), (600, 200), (700, 200), (800, 200),(900,200),(1000,200),(100, 300), (200, 300), (300, 300), (400, 300),(500, 300), (600, 300), (700, 300), (800, 300),(900,300),(100, 400), (200, 400), (300, 400), (400, 400),(500, 400), (600, 400), (700, 400)]#first row 10 second 9 third 8
# keys = ['a', 's', 'd', 'f','g','h','j','k']
keys =['1','2','3','4','5','6','7','8','9','0','q','w','e','r','t','y','u','i','o','p','a','s','d','f','g','h','j','k','l','z','x','c','v','b','n','m']

# Initialize the webcam and the MediaPipe hand tracking solution
cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands.Hands()

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    frame=cv2.flip(frame,1)

    # Convert the frame to RGB color space
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with the MediaPipe hand tracking solution
    results = mp_hands.process(rgb)

    # Draw the virtual keyboard on the frame
    for i, position in enumerate(positions):
        cv2.putText(frame, keys[i], position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.circle(frame, position, 10, (0, 255, 0), -1)
        # Draw a square boundary around the circle
        x, y = position
        w, h = 80, 80
        cv2.rectangle(frame, (x - w // 2, y - h // 2), (x + w // 2, y + h // 2), (255, 255, 0), 2)

    # Detect hand movements on the virtual keyboard
    # if results.multi_hand_landmarks:
    #     for hand_landmarks in results.multi_hand_landmarks:
    #         for i, position in enumerate(positions):
    #             # Calculate the distance between the center of the key and the hand landmarks
    #             dx = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP].x * frame.shape[1] - position[0]
    #             dy = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP].y * frame.shape[0] - position[1]
    #             distance = (dx ** 2 + dy ** 2) ** 0.5
    #
    #             # If the distance is within a threshold, simulate a key press
    #             if distance < 50:
    #                 pyautogui.press(keys[i])
    #
    #         # Draw the hand landmarks on the frame
    #         mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for i, position in enumerate(positions):
                # Calculate the distance between the center of the key and each fingertip landmark
                distances = []
                for finger_tip in [mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP,
                                   mp.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP,
                                   mp.solutions.hands.HandLandmark.RING_FINGER_TIP,
                                   mp.solutions.hands.HandLandmark.PINKY_TIP,
                                   mp.solutions.hands.HandLandmark.THUMB_TIP]:
                    dx = hand_landmarks.landmark[finger_tip].x * frame.shape[1] - position[0]
                    dy = hand_landmarks.landmark[finger_tip].y * frame.shape[0] - position[1]
                    distance = (dx ** 2 + dy ** 2) ** 0.5
                    distances.append(distance)

                # If the minimum distance is within a threshold, simulate a key press
                if min(distances) < 50:
                    pyautogui.press(keys[i])

            # Draw the hand landmarks on the frame
            mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)

    # Show the frame
    cv2.imshow('Virtual Keyboard', frame)

    # Exit the program if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
