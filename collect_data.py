import cv2
import mediapipe as mp
import numpy as np
import os
import time

#  Settings

WORDS        = ["friend","forever","enemy"]
VIDEOS_COUNT = 40
FRAMES_COUNT = 60
DATASET_PATH = "dataset"


#  Create folders

for word in WORDS:
    os.makedirs(os.path.join(DATASET_PATH, word), exist_ok=True)


#  MediaPipe setup

mp_hands   = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands      = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)


#  Extract landmarks from a frame

def extract_landmarks(frame, draw=False):
    """
    Returns array of shape (126,) = 2 hand slots x 21 landmarks x 3 (x, y, z)
    Slots: [0:63]   = Hand 1 (first hand MediaPipe detects, zeros if none)
           [63:126] = Hand 2 (second hand MediaPipe detects, zeros if none)

    - If only one hand is visible, Hand 2 slot stays zeros.
    - If two hands are visible, both slots are filled.
    - Every frame is evaluated independently (a hand can appear/disappear
      mid-video with no issue).
    - No Left/Right distinction is used (matches the flip augmentation
      already applied during training, which treats hand identity as
      symmetric).

    If draw=True, also draws the detected hand landmarks on `frame` in place.
    """
    rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    hand1 = np.zeros(63)
    hand2 = np.zeros(63)

    if result.multi_hand_landmarks:
        for i, hand_landmarks in enumerate(result.multi_hand_landmarks[:2]):
            if draw:
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                )

            coords = []
            for lm in hand_landmarks.landmark:
                coords.extend([lm.x, lm.y, lm.z])
            coords = np.array(coords)

            if i == 0:
                hand1 = coords
            else:
                hand2 = coords

    return np.concatenate([hand1, hand2])


#  Draw UI on frame

def draw_info(frame, word, video_num, state, countdown=None):
    h, w = frame.shape[:2]

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 80), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    cv2.putText(frame, f"Word: {word.upper()}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
    cv2.putText(frame, f"Video: {video_num}/{VIDEOS_COUNT}",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    if state == "READY":
        msg   = f"Ready? Press SPACE to start" if countdown is None else f"Starting in {countdown}..."
        color = (0, 255, 255)
    elif state == "RECORDING":
        msg   = "RECORDING..."
        color = (0, 0, 255)
        cv2.circle(frame, (w - 30, 30), 12, (0, 0, 255), -1)
    else:
        msg   = state
        color = (0, 255, 0)

    cv2.putText(frame, msg, (10, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    return frame


#  Main loop

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Cannot open camera")
    exit()

print("Camera ready")
print("SPACE = start recording | Q = quit")

for word in WORDS:
    print(f"\n{'='*40}\n  Word: {word.upper()}\n{'='*40}")
    video_num = 0

    while video_num < VIDEOS_COUNT:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame = draw_info(frame, word, video_num + 1, "READY")
        cv2.imshow("ASL Data Collection", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            exit()

        if key == ord(' '):
            # Countdown 3 seconds
            for count in range(3, 0, -1):
                deadline = time.time() + 1.0
                while time.time() < deadline:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame = cv2.flip(frame, 1)
                    frame = draw_info(frame, word, video_num + 1, "READY", count)
                    cv2.imshow("ASL Data Collection", frame)
                    cv2.waitKey(1)

            # Record frames
            sequence = []
            for frame_num in range(FRAMES_COUNT):
                ret, frame = cap.read()
                if not ret:
                    break
                frame  = cv2.flip(frame, 1)
                landmarks = extract_landmarks(frame, draw=True)
                sequence.append(landmarks)

                progress = int((frame_num + 1) / FRAMES_COUNT * 200)
                cv2.rectangle(frame,
                              (10, frame.shape[0] - 60),
                              (10 + progress, frame.shape[0] - 45),
                              (0, 255, 0), -1)

                frame = draw_info(frame, word, video_num + 1, "RECORDING")
                cv2.imshow("ASL Data Collection", frame)
                cv2.waitKey(1)

            # Save sequence
            sequence  = np.array(sequence)
            save_path = os.path.join(DATASET_PATH, word, f"{video_num:03d}.npy")
            np.save(save_path, sequence)
            print(f"  Saved: {save_path} | shape: {sequence.shape}")
            video_num += 1

            # Short rest between videos
            if video_num < VIDEOS_COUNT:
                rest_end = time.time() + 1.5
                while time.time() < rest_end:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame = cv2.flip(frame, 1)
                    remaining = rest_end - time.time()
                    frame = draw_info(frame, word, video_num + 1,
                                      f"Saved! Next in {remaining:.1f}s")
                    cv2.imshow("ASL Data Collection", frame)
                    cv2.waitKey(1)

    print(f"Done: {word.upper()} ({VIDEOS_COUNT} videos)")

cap.release()
cv2.destroyAllWindows()
print(f"\nAll done! Data saved in '{DATASET_PATH}' folder")
