import cv2
import mediapipe as mp
import pyautogui
import time
import math

class handDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=False, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands,
                                        self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]

    def findHands(self, img, draw=True):    # Finds all hands in a frame
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms,
                                               self.mpHands.HAND_CONNECTIONS)

        return img

    def findPosition(self, img, handNo=0, draw=True):   # Fetches the position of hands
        xList = []
        yList = []
        bbox = []
        self.lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                xList.append(cx)
                yList.append(cy)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

            xmin, xmax = min(xList), max(xList)
            ymin, ymax = min(yList), max(yList)
            bbox = xmin, ymin, xmax, ymax

            if draw:
                cv2.rectangle(img, (xmin - 20, ymin - 20), (xmax + 20, ymax + 20),
                              (0, 255, 0), 2)

        return self.lmList, bbox

    def fingersUp(self):    # Checks which fingers are up
        fingers = []
        # Thumb
        if self.lmList[self.tipIds[0]][1] > self.lmList[self.tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # Fingers
        for id in range(1, 5):

            if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        # totalFingers = fingers.count(1)

        return fingers

    def findDistance(self, p1, p2, img, draw=True,r=15, t=3):   # Finds distance between two fingers
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), t)
            cv2.circle(img, (x1, y1), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (cx, cy), r, (0, 0, 255), cv2.FILLED)
        length = math.hypot(x2 - x1, y2 - y1)

        return length, img, [x1, y1, x2, y2, cx, cy]




detector = handDetector(maxHands=1)                  # Detecting one hand at max
# Define the positions and keys of the virtual keyboard
positions = [(100, 100), (200, 100), (300, 100), (400, 100),(500, 100), (600, 100), (700, 100), (800, 100),(900,100),(1000,100),(100, 200), (200, 200), (300, 200), (400, 200),(500, 200), (600, 200), (700, 200), (800, 200),(900,200),(1000,200),(100, 300), (200, 300), (300, 300), (400, 300),(500, 300), (600, 300), (700, 300), (800, 300),(900,300),(1000,300),(100, 400), (200, 400), (300, 400), (400, 400),(500, 400), (600, 400), (700, 400),(800,400)]#first row 10 second 9 third 8
# keys = ['a', 's', 'd', 'f','g','h','j','k']
keys =['1','2','3','4','5','6','7','8','9','0','q','w','e','r','t','y','u','i','o','p','a','s','d','f','g','h','j','k','l','backspace','z','x','c','v','b','n','m','enter']

# Initialize the webcam and the MediaPipe hand tracking solution
cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands.Hands()

pressed_keys = []


while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    frame=cv2.flip(frame,1)
    frame = detector.findHands(frame)                       # Finding the hand
    lmlist, bbox = detector.findPosition(frame)           # Getting position of hand


    # Convert the frame to RGB color space
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with the MediaPipe hand tracking solution
    results = mp_hands.process(rgb)

    # Draw the virtual keyboard on the frame
    pressed_key_index=None
    for i, position in enumerate(positions):
        # box_color = (255, 255, 255)
        # if i in pressed_keys:
        #     box_color = (0, 255, 0)  # Change the color to green temporarily
        #
        cv2.putText(frame, keys[i], position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        # cv2.circle(frame, position, 10, (0, 255, 0), -1)
        # Draw a square boundary around the circle
        x, y = position
        w, h = 80, 80
        cv2.rectangle(frame, (x - w // 2, y - h // 2), (x + w // 2, y + h // 2), (255,0,255), 2)

        # if i == pressed_key_index:
        #     # if the key is pressed, draw it in a different color
        #     cv2.rectangle(frame, (position[0] - 40, position[1] - 40), (position[0] + 40, position[1] + 40),
        #                   (0, 255, 0), -1)
        # else:
        #     cv2.rectangle(frame, (position[0] - 40, position[1] - 40), (position[0] + 40, position[1] + 40),
        #                   (255, 0, 0), 2)
        # cv2.putText(frame, keys[i], position, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Detect hand movements on the virtual keyboard

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for i, position in enumerate(positions):
                # Calculate the distance between the center of the key and each fingertip landmark
                distances = []
                for finger_tip in [mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]:
                    dx = hand_landmarks.landmark[finger_tip].x * frame.shape[1] - position[0]
                    dy = hand_landmarks.landmark[finger_tip].y * frame.shape[0] - position[1]
                    distance = (dx ** 2 + dy ** 2) ** 0.5
                    distances.append(distance)

                # # If the minimum distance is within a threshold, simulate a key press
                # if min(distances) < 50:
                #     pyautogui.press(keys[i])

                if len(lmlist) != 0:
                    fingers = detector.fingersUp()  # Checking if fingers are upwards

                    if min(distances)<50:
                        if fingers[1] == 1 and fingers[2] == 1:
                            # pyautogui.press(keys[i])
                            # pressed_keys.append(i)
                            # # box_color = (0, 255, 0)  # Change the color of the box to green
                            #
                            # print("we were here")
                            # time.sleep(.5)

                            # if not key_pressed:
                            #     pressed_key_index = i
                            pyautogui.press(keys[i])
                            print("we pressed "+keys[i])
                            time.sleep(1)

                                # key_pressed = True
                                # key_pressed_time = time.time()
                            # else:
                                # if a key is already pressed, check if it has been held for more than 0.5 seconds
                                # if time.time() - key_pressed_time > 0.5:
                                #     pyautogui.press(keys[pressed_key_index])
                                #     key_pressed = False
                                # else:
                                #     draw the pressed key in a different color for visual feedback
                                    # cv2.rectangle(frame, (
                                    # positions[pressed_key_index][0] - 40, positions[pressed_key_index][1] - 40), (
                                    #               positions[pressed_key_index][0] + 40,
                                    #               positions[pressed_key_index][1] + 40), (0, 255, 0), -1)

            # Draw the hand landmarks on the frame
            mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)

    # Remove the highlighted color after the key is pressed and released
    # if len(pressed_keys) > 0:
    #     pressed_keys.pop()

    # Show the frame
    cv2.imshow('Virtual Keyboard', frame)
    #
    # Exit the program if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
