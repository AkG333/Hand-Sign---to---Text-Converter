import os
import cv2

# Define the directory to store data
DIR = r"C:\Users\ACER.DESKTOP-G5EVCFN\OneDrive\Desktop\FP2\Data2"

# Parameters
dataset_size = 100  # Number of images per class

# opening the camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

while True:
    # Asking for the feature/frame-meaning
    label = input("Enter the class label (or type 'exit' to quit): ").strip()
    if label.lower() == "exit":
        print("Exiting...")
        break

    # Create directory for the current class
    class_dir = os.path.join(DIR, label)
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    print(f"Ready to collect data for class '{label}'")

    # We Open the camera and wait for 'Y' to start
    while True:
        ret, frame = cap.read()#Here it extract a frame and returns a boolean value
        if not ret:
            print("Error: Failed to capture frame.")
            continue

        # Displaying instructions on the frame
        cv2.putText(frame, f'Class "{label}": Press "Y" to start or "Q" to exit.', (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.imshow('frame', frame)

        # Wait for user input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('y'):
            break
        elif key == ord('q'):
            print("Exiting...")
            cap.release()
            cv2.destroyAllWindows()
            exit()

    # Collect dataset for the class
    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            continue

        # Show the frame with progress
        cv2.putText(frame, f'Class "{label}": Image {counter + 1}/{dataset_size}', (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.imshow('frame', frame)

        # Save the frame as an image
        # here the first argument is the frame path and the second is the frame itself
        cv2.imwrite(os.path.join(class_dir, f'{counter}.jpg'), frame)
        counter += 1

        # Exiting the loop if 'Q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting...")
            cap.release()
            cv2.destroyAllWindows()
            exit()

    print(f"Finished collecting data for class '{label}'")
    w = input("Press Enter to proceed to the next class or type 'exit' to quit: ")
    if w.lower() == 'exit':
        break
    print("-" * 40)

cap.release()
cv2.destroyAllWindows()
