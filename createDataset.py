import os
import pickle
import mediapipe as mp
import cv2

# Initialize Mediapipe Hands module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Path to the data directory
DATA_DIR = r"C:\Users\ACER.DESKTOP-G5EVCFN\OneDrive\Desktop\FP2\Data"

# Initialize data and labels
data = []
labels = []

def process_images(data_dir):
    """Processes images in the given directory to extract hand landmarks."""
    for dir_ in os.listdir(data_dir):
        gesture_label = dir_  # Use folder name as the gesture label
        gesture_path = os.path.join(data_dir, dir_)

        for img_path in os.listdir(gesture_path):
            img_full_path = os.path.join(gesture_path, img_path)
            try:
                img = cv2.imread(img_full_path)
                if img is None:
                    print(f"Skipping invalid image: {img_full_path}")
                    continue

                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                results = hands.process(img_rgb)

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        landmarks = []

                        # Extract x and y coordinates of landmarks
                        x_coords = [landmark.x for landmark in hand_landmarks.landmark]
                        y_coords = [landmark.y for landmark in hand_landmarks.landmark]

                        # Normalize coordinates
                        x_min, y_min = min(x_coords), min(y_coords)
                        x_max, y_max = max(x_coords), max(y_coords)

                        x_coords = [(x - x_min) / (x_max - x_min) if (x_max - x_min) != 0 else 0 for x in x_coords]
                        y_coords = [(y - y_min) / (y_max - y_min) if (y_max - y_min) != 0 else 0 for y in y_coords]

                        # Flatten normalized coordinates
                        for x, y in zip(x_coords, y_coords):
                            landmarks.append(x)
                            landmarks.append(y)

                        # Only append data if the expected number of landmarks is present
                        if len(landmarks) == 42:
                            data.append(landmarks)
                            labels.append(gesture_label)

            except Exception as e:
                print(f"Error processing {img_full_path}: {e}")

# Process the dataset
process_images(DATA_DIR)

# Save the processed data to a pickle file
output_path = r"C:\Users\ACER.DESKTOP-G5EVCFN\OneDrive\Desktop\FP2\data.pickle"
with open(output_path, 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

print(f"Data processing complete. Saved to {output_path}")
