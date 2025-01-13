import cv2
import pickle
import mediapipe as mp
import numpy as np

# Load the trained model
model_path = ""
with open(model_path, 'rb') as f:
    model_dict = pickle.load(f)
    model = model_dict['model']

# Initialize Mediapipe Hands module
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

def preprocess_landmarks(hand_landmarks):
    #Extract and normalize landmarks from Mediapipe output.
    x_coords = [landmark.x for landmark in hand_landmarks.landmark]
    y_coords = [landmark.y for landmark in hand_landmarks.landmark]

    x_min, y_min = min(x_coords), min(y_coords)
    x_max, y_max = max(x_coords), max(y_coords)

    x_coords = [(x - x_min) / (x_max - x_min) if (x_max - x_min) != 0 else 0 for x in x_coords]
    y_coords = [(y - y_min) / (y_max - y_min) if (y_max - y_min) != 0 else 0 for y in y_coords]

    landmarks = []
    for x, y in zip(x_coords, y_coords):
        landmarks.append(x)
        landmarks.append(y)

    return landmarks

# Start the webcam feed
cap = cv2.VideoCapture(0)

print("Press 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame. Exiting...")
        break

    # Flip the frame horizontally for a mirror effect
    frame = cv2.flip(frame, 1)

    # Convert the frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with Mediapipe Hands
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks on the frame
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

            # Preprocess landmarks for prediction
            landmarks = preprocess_landmarks(hand_landmarks)

            if len(landmarks) == 42:  # Ensure correct number of landmarks
                # Predict gesture
                landmarks = np.array(landmarks).reshape(1, -1)
                prediction = model.predict(landmarks)[0]

                # Display prediction on the frame
                cv2.putText(
                    frame, f"Alphabet: {prediction}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA
                )

    # Display the frame
    cv2.imshow('Hand Gesture Recognition', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
hands.close()
