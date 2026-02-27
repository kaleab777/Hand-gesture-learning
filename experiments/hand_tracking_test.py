import cv2
import mediapipe as mp
import math
from collections import deque, Counter

history = deque(maxlen=10)     # last 10 frame predictions
MIN_VOTES = 6                  # must appear at least 6/10 frames

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


def dist(a, b):
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)


def print_screen(frame, text):
    cv2.putText(frame, text, (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Open the webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Flip the frame for natural interaction

    # Convert the BGR image to RGB(mediapipe needs RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb_frame)  # Process the frame and detect hands

    # Draw hand landmarks on the original frame
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )
            lm = hand_landmarks.landmark

        palm_width = dist(lm[5], lm[17])
        if palm_width < 1e-6:
            continue

        thumb_tip = lm[4]
        index_tip = lm[8]

        pinch_normalized = dist(lm[4], lm[8])/palm_width
        thumb_close_to_index = dist(lm[4], lm[6])/palm_width

        thumb_up = lm[4].y < lm[3].y < lm[2].y
        index_up = lm[8].y < lm[6].y
        middle_up = lm[12].y < lm[10].y
        ring_down = lm[16].y > lm[14].y
        pinky_down = lm[20].y > lm[18].y

        index_down = lm[8].y > lm[5].y
        middle_down = lm[12].y > lm[9].y
        ring_down2 = lm[16].y > lm[13].y
        pinky_down2 = lm[20].y > lm[17].y

        # print(f"Thumb-Index Distance: {thumb_close_to_index:.3f}")

        print(
            f"index_up: {index_up}, middle_up: {middle_up}, ring_down: {ring_down}, pinky_down: {pinky_down}")

        label = None
        if index_up and middle_up and ring_down and pinky_down:
            label = "Peace out"
        elif thumb_close_to_index < 0.35 and index_down and middle_down and ring_down2 and pinky_down2:
            label = "FIST BUMP!"
        elif pinch_normalized < 0.35 and thumb_up and not index_up and not middle_up and ring_down and pinky_down:
            label = "Pinching"
        elif thumb_up and (not index_up) and (not middle_up) and ring_down2 and pinky_down2:
            label = "Good!"

        history.append(label)

        counts = Counter([x for x in history if x is not None])
        stable_label = None

        if counts:
            best_label, best_count = counts.most_common(1)[0]
            if best_count >= MIN_VOTES:
                stable_label = best_label

        if stable_label:
            print_screen(frame, stable_label)

    cv2.imshow('Hand Tracking', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
