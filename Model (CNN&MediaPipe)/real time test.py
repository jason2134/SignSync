# %%
# Import necessary libraries
import cv2
import numpy as np
import os
import mediapipe as mp
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import random
import tensorflow as tf


# %%
# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)

# %%
# Initialize MediaPipe hands model and drawing utilities
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Define functions for MediaPipe detection and drawing
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # COLOR COVERSION RGB 2 BGR
    return image, results

def draw_styled_landmarks(image, results):
    # Check if hand landmarks were detected
    if results.multi_hand_landmarks and results.multi_handedness:
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            # Retrieve the handedness label for this hand (e.g., "Left" or "Right")
            handedness = results.multi_handedness[idx].classification[0].label
            if handedness == 'Left':
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                    mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                )
            else:  # Assume the other hand is "Right"
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                    mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                )



# %%
def extract_key_points(results):
    import numpy as np
    # Initialize key points for left and right hands as zeros (21 landmarks * 3 coordinates)
    left_hand = np.zeros(21 * 3)
    right_hand = np.zeros(21 * 3)
    
    # Check if any hand landmarks and handedness information are detected
    if results.multi_hand_landmarks and results.multi_handedness:
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            # Flatten the key points: each landmark provides x, y, and z coordinates
            keypoints = np.array([[res.x, res.y, res.z] for res in hand_landmarks.landmark]).flatten()
            # Retrieve handedness label ("Left" or "Right")
            handedness = results.multi_handedness[idx].classification[0].label
            
            # Save the keypoints based on the handedness of the detected hand
            if handedness == "Left":
                left_hand = keypoints
            elif handedness == "Right":
                right_hand = keypoints
    
    # Combine the key points for left and right hands into a single array
    return np.concatenate([left_hand, right_hand])


# %%
def normalize_keypoints(sequence):
    """
    sequence: np.array, shape (frames, 126)
              where each row is 2×21 landmarks flattened (x,y,z).
    Returns: np.array of same shape, but centred only.
    """
    # Reshape to (frames, 42 landmarks, 3 coords)
    seq = sequence.reshape(sequence.shape[0], 42, 3)
    normalized = []

    for frame in seq:
        # 1. Compute wrist midpoint
        wrist_left  = frame[0]      # left-hand wrist
        wrist_right = frame[21]     # right-hand wrist
        origin = (wrist_left + wrist_right) / 2.0

        # 2. Centre all keypoints around that midpoint
        centred = frame - origin

        # 3. Flatten back to length-126 vector
        normalized.append(centred.flatten())

    return np.array(normalized)



# Actions to detect
actions = np.array(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
                    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
                    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9','10',
                    'SPACE', 'BACKSPACE','BACKGROUND'])



# Update the preprocess_frame function
def preprocess_frame(results):
    # Extract keypoints
    keypoints = extract_key_points(results)
    # print(keypoints)
    # Reshape to (1, keypoints)
    keypoints = keypoints.reshape(1, -1)
    # Normalize keypoints
    keypoints = normalize_keypoints(keypoints)
    return keypoints[0]

# %%
colors = [
    (255, 0, 0),       # A
    (0, 255, 0),       # B
    (0, 0, 255),       # C
    (255, 255, 0),     # D
    (0, 255, 255),     # E
    (255, 0, 255),     # F
    (190, 125, 0),     # G
    (0, 190, 125),     # H
    (190, 0, 125),     # I
    (25, 185, 0),      # J
    (0, 25, 185),      # K
    (185, 0, 25),      # L
    (100, 0, 100),     # M
    (0, 100, 100),     # N
    (123, 123, 0),     # O
    (255, 165, 0),     # P
    (75, 0, 130),      # Q
    (255, 20, 147),    # R
    (0, 128, 0),       # S
    (128, 0, 128),     # T
    (0, 0, 128),       # U
    (128, 128, 0),     # V
    (0, 100, 200),     # W
    (28, 20, 50),      # X
    (85, 100, 128),    # Y
    (70, 75, 75),      # Z
    (200, 200, 200),   # 0
    (150, 150, 150),   # 1
    (100, 100, 100),   # 2
    (50, 50, 50),      # 3
    (25, 25, 25),      # 4
    (180, 50, 50),     # 5
    (50, 180, 50),     # 6
    (50, 50, 180),     # 7
    (180, 180, 50),    # 8
    (50, 180, 180),    # 9
    (50, 180, 10),     # 10
    (255, 192, 203),   # SPACE
    (169, 169, 169),    # BACKSPACE
    (169, 0, 169)      # BACKGROUND
]



# %%
def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    
    # Get the dimensions of the window
    frame_height, frame_width, _ = output_frame.shape
    
    # Set configuration for dynamic layout
    left_column_letters = 19  # Adjusted number of letters on the left side
    top_row_letters = 0       # No letters on the top side
    label_height = 30         # Vertical spacing for left and right
    label_width = 120         # Maximum width of rectangles for probability bars
    top_spacing = 40          # Adjustable spacing for top row
    top_letter_spacing = 5    # Adjustable spacing between top letters

    # Iterate over the results and position letters accordingly
    for num, prob in enumerate(res):
        if num < left_column_letters:
            # Left-side placement
            x_offset = 0
            y_position = top_spacing + num * label_height
            # Probability bars increasing to the right
            cv2.rectangle(output_frame, (x_offset, y_position), 
                          (x_offset + int(prob * 100), y_position + 20), 
                          colors[num % len(colors)], -1)
            cv2.putText(output_frame, actions[num], (x_offset, y_position + 15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        else:
            # Right-side placement
            x_offset = frame_width - label_width  # Adjust the starting point to be near the edge
            y_position = top_spacing + (num - left_column_letters) * label_height
            # Probability bars increasing to the left
            bar_start = x_offset + label_width - int(prob * 100)
            cv2.rectangle(output_frame, 
                          (bar_start, y_position),
                          (x_offset + label_width, y_position + 20), 
                          colors[num % len(colors)], -1)
            cv2.putText(output_frame, actions[num], 
                        (bar_start - 20, y_position + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    
    return output_frame

# %%
# New Detection variables
sequence = []
sentence = []
predictions = []
threshold = 0.9

# Load your trained model
model = tf.keras.models.load_model('new model.h5')  # Ensure 'best_model.keras' is your trained model

cap = cv2.VideoCapture(0)

# Access MediaPipe model
with mp_hands.Hands(
    static_image_mode=False, max_num_hands=2, min_detection_confidence=0.8, min_tracking_confidence=0.8
) as hands:
    while cap.isOpened():
        # Read the feed
        ret, frame = cap.read()

        if not ret:
            print("Ignoring empty camera frame.")
            break
        frame=cv2.flip(frame,1)
        # Make detections
        image, results = mediapipe_detection(frame, hands)
        # print(results)  # Uncomment to see the detection results

        # Draw landmarks
        draw_styled_landmarks(image, results)

        # Preprocess the frame
        keypoints = preprocess_frame(results)
        sequence.append(keypoints)
        sequence = sequence[-30:]  # Keep only the last 30 frames

        # Once we have 30 frames, make a prediction
        if len(sequence) == 30:
            res = model.predict(np.expand_dims(sequence, axis=0))[0]
            # print(actions[np.argmax(res)])
            predictions.append(np.argmax(res))
            # Visualize results if the prediction is consistent
            if np.unique(predictions[-10:])[0] == np.argmax(res):
                if res[np.argmax(res)] > threshold:
                    if len(sentence) > 0 and actions[np.argmax(res)] != 'BACKGROUND':
                        if actions[np.argmax(res)] != sentence[-1]:
                            sentence.append(actions[np.argmax(res)])
                    else:
                        if actions[np.argmax(res)] != 'BACKGROUND':
                            sentence.append(actions[np.argmax(res)])

                if len(sentence) > 5:
                    sentence = sentence[-5:]

            # Show probability visualization
            image = prob_viz(res, actions, image, colors)
        # Display the result
        cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
        cv2.putText(image, ' '.join(sentence), (3, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        # Show to the screen
        # image=cv2.flip(image,1)
        cv2.imshow('Frame', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()




