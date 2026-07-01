import cv2
import mediapipe as mp
import numpy as np
import json
import collections
import tensorflow as tf

#  Settings

MODEL_PATH      = "model.h5"
LABELS_PATH     = "labels.json"
FRAMES_COUNT    = 60
CONFIDENCE_MIN  = 0.90   # only show prediction if confidence >= 90%
SMOOTH_WINDOW   = 5      # smooth predictions over last 5 results


#  Load model and labels

print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)

with open(LABELS_PATH, "r") as f:
    label_map = json.load(f)

index_to_word = {v: k for k, v in label_map.items()}
print(f"  Words: {list(label_map.keys())}")
print("  Model ready!")


#  MediaPipe setup

mp_hands   = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands      = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)


#  Extract and normalize landmarks

def extract_landmarks(frame):
    """
    Returns normalized array of shape (126,) = 2 hand slots x 21 x 3
    Slots: [0:63]   = Hand 1 (first hand MediaPipe detects, zeros if none)
           [63:126] = Hand 2 (second hand MediaPipe detects, zeros if none)
    Each hand is normalized against its OWN wrist (landmark 0).
    A hand slot that's not detected is left as zeros (no fake normalization).

    Also returns the list of raw hand_landmarks (for drawing), or None.
    """
    rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    hand1 = np.zeros(63)
    hand2 = np.zeros(63)

    if result.multi_hand_landmarks:
        for i, hand_landmarks in enumerate(result.multi_hand_landmarks[:2]):
            coords = []
            for lm in hand_landmarks.landmark:
                coords.extend([lm.x, lm.y, lm.z])
            coords = np.array(coords)

            # Normalize this hand against its own wrist
            wrist = coords[:3].copy()
            coords[0::3] -= wrist[0]
            coords[1::3] -= wrist[1]
            coords[2::3] -= wrist[2]

            if i == 0:
                hand1 = coords
            else:
                hand2 = coords

        return np.concatenate([hand1, hand2]), result.multi_hand_landmarks
    else:
        return np.concatenate([hand1, hand2]), None


#  Draw prediction UI

def draw_ui(frame, word, confidence, buffer_size, hand_detected):
    h, w = frame.shape[:2]

    # Top bar background
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 90), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # Prediction word
    if word and confidence >= CONFIDENCE_MIN:
        cv2.putText(frame, word.upper(),
                    (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 0), 3)
        conf_text = f"{confidence * 100:.1f}%"
        cv2.putText(frame, conf_text,
                    (10, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "...",
                    (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (150, 150, 150), 2)

    # Buffer progress bar (how many frames collected)
    bar_w   = int((buffer_size / FRAMES_COUNT) * (w - 20))
    bar_color = (0, 200, 255) if buffer_size < FRAMES_COUNT else (0, 255, 0)
    cv2.rectangle(frame, (10, h - 20), (w - 10, h - 8), (50, 50, 50), -1)
    cv2.rectangle(frame, (10, h - 20), (10 + bar_w, h - 8), bar_color, -1)
    cv2.putText(frame, f"Buffer: {buffer_size}/{FRAMES_COUNT}",
                (10, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    # Hand detection indicator
    dot_color = (0, 255, 0) if hand_detected else (0, 0, 255)
    cv2.circle(frame, (w - 20, 20), 8, dot_color, -1)
    status = "Hand detected" if hand_detected else "No hand"
    cv2.putText(frame, status, (w - 130, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, dot_color, 1)

    # Instructions
    cv2.putText(frame, "Q = quit | C = clear",
                (w - 200, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

    return frame


#  Main real-time loop

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Cannot open camera")
    exit()

sequence         = []                              # rolling buffer of landmarks
recent_preds     = collections.deque(maxlen=SMOOTH_WINDOW)  # last N predictions
current_word     = None
current_conf     = 0.0

print("\nReal-time ASL recognition started")
print("Q = quit | C = clear buffer")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    # Extract landmarks
    landmarks, hands_lms = extract_landmarks(frame)
    hand_detected        = hands_lms is not None

    # Draw hand skeleton(s)
    if hands_lms:
        for hand_lms in hands_lms:
            mp_drawing.draw_landmarks(
                frame, hand_lms, mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=3),
                mp_drawing.DrawingSpec(color=(0, 180, 255), thickness=2)
            )

    # Add to sequence buffer
    sequence.append(landmarks)

    # Keep only last FRAMES_COUNT frames
    if len(sequence) > FRAMES_COUNT:
        sequence.pop(0)

    # Predict when buffer is full
    if len(sequence) == FRAMES_COUNT:
        input_data = np.expand_dims(sequence, axis=0)  # shape (1, FRAMES_COUNT, 126)
        predictions = model.predict(input_data, verbose=0)[0]
        print(f"Predictions: {predictions}")
        predicted_idx  = np.argmax(predictions)
        confidence     = predictions[predicted_idx]

        # Add to smoothing window
        recent_preds.append(predicted_idx)

        # Use most common prediction from recent window
        smoothed_idx  = collections.Counter(recent_preds).most_common(1)[0][0]
        current_word  = index_to_word[smoothed_idx]
        current_conf  = confidence

    # Draw UI
    frame = draw_ui(frame, current_word, current_conf,
                    len(sequence), hand_detected)

    cv2.imshow("ASL Real-Time Recognition", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        sequence.clear()
        recent_preds.clear()
        current_word = None
        current_conf = 0.0
        print("Buffer cleared")

cap.release()
cv2.destroyAllWindows()
print("Closed.")
