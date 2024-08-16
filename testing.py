import pickle
import cv2
import mediapipe as mp
import numpy as np
import time

model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels_dict = {
    0: {'character': 'A', 'capture_duration': 1.0},
    1: {'character': 'B', 'capture_duration': 1.0},
    2: {'character': ' ', 'capture_duration': 1.0},
    3: {'character': 'D', 'capture_duration': 1.0},
    4: {'character': 'E', 'capture_duration': 1.0},
    5: {'character': 'F', 'capture_duration': 1.0},
    6: {'character': 'G', 'capture_duration': 1.0},
    7: {'character': 'H', 'capture_duration': 1.0},
    8: {'character': 'I', 'capture_duration': 1.0},
    9: {'character': 'J', 'capture_duration': 1.0},
    10: {'character': 'K', 'capture_duration': 1.0},
    11: {'character': 'L', 'capture_duration': 1.0},
    12: {'character': 'M', 'capture_duration': 1.0},
    13: {'character': 'N', 'capture_duration': 1.0},
    14: {'character': 'O', 'capture_duration': 1.0},
    15: {'character': 'P', 'capture_duration': 1.0},
    16: {'character': 'Q', 'capture_duration': 1.0},
    17: {'character': 'R', 'capture_duration': 1.0},
    18: {'character': 'S', 'capture_duration': 1.0},
    19: {'character': 'T', 'capture_duration': 1.0},
    20: {'character': 'U', 'capture_duration': 1.0},
    21: {'character': 'V', 'capture_duration': 1.0},
    22: {'character': 'W', 'capture_duration': 1.0},
    23: {'character': 'X', 'capture_duration': 1.0},
    24: {'character': 'Y', 'capture_duration': 1.0},
    25: {'character': 'Z', 'capture_duration': 1.0},
    26: {'character': ' ', 'capture_duration': 1.0},
}



def print_letter(frame, predicted_character, x1, y1):
    if isinstance(predicted_character, dict):
        predicted_character = predicted_character.get('character', '')
    cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)


capture_image = False
captured_letters = ""
buffer_time = 1.5

while True:
    ret, frame = cap.read()

    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

        for hand_landmarks in results.multi_hand_landmarks:
            data_aux = []
            x_ = []
            y_ = []

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

                x_.append(x)
                y_.append(y)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10

            x2 = int(max(x_) * W) - 10
            y2 = int(max(y_) * H) - 10

            prediction = model.predict([np.asarray(data_aux)])

            predicted_character = labels_dict.get(int(prediction[0]))

            if predicted_character and not capture_image:
                capture_image = True
                start_time = time.time()
                capture_duration = predicted_character['capture_duration']

            if capture_image:
                if time.time() - start_time > capture_duration:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), thickness=cv2.FILLED)
                    captured_letters += predicted_character['character']
                    print(predicted_character['character'], end='', flush=True)
                    time.sleep(buffer_time)
                    capture_image = False

            print_letter(frame, predicted_character, x1, y1)

    cv2.imshow('frame', frame)
    cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()





