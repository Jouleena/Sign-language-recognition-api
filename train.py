import numpy as np
import os
import json
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical

# ─────────────────────────────────────────
#  Settings
# ─────────────────────────────────────────
DATASET_PATH = "dataset"
MODEL_PATH   = "model.h5"
LABELS_PATH  = "labels.json"
FRAMES_COUNT = 60
EPOCHS       = 70
BATCH_SIZE   = 10


#  Step 1 — Load data

print("Loading data...")

WORDS     = sorted(os.listdir(DATASET_PATH))
label_map = {word: idx for idx, word in enumerate(WORDS)}

with open(LABELS_PATH, "w") as f:
    json.dump(label_map, f)

print(f"  Words: {WORDS}")

X, y = [], []

for word in WORDS:
    word_path = os.path.join(DATASET_PATH, word)
    files     = [f for f in os.listdir(word_path) if f.endswith(".npy")]

    for file in files:
        sequence = np.load(os.path.join(word_path, file))

        if sequence.shape == (FRAMES_COUNT, 126):
            X.append(sequence)
            y.append(label_map[word])
        else:
            print(f"  WARNING: wrong shape {file} -> {sequence.shape}")

X = np.array(X)
y = np.array(y)

print(f"  Total samples before augmentation: {len(X)}")

# ─────────────────────────────────────────
#  Step 2 — Normalize landmarks (per hand, wrist-relative)
# ─────────────────────────────────────────
# Layout per frame: [0:63] = Hand 1, [63:126] = Hand 2
# Each hand is normalized against its OWN wrist (landmark 0).
# If a hand slot is all zeros (hand not present in that frame),
# it is left untouched — subtracting would just create fake
# non-zero "phantom hand" values out of nothing.
HAND_DIM = 63

for i in range(len(X)):
    for f in range(FRAMES_COUNT):
        for h_start in (0, HAND_DIM):
            h_end = h_start + HAND_DIM
            hand = X[i, f, h_start:h_end]

            if not np.any(hand):
                continue  # hand not detected in this frame -> keep zeros

            wrist = hand[:3].copy()
            X[i, f, h_start + 0:h_end:3] -= wrist[0]
            X[i, f, h_start + 1:h_end:3] -= wrist[1]
            X[i, f, h_start + 2:h_end:3] -= wrist[2]

print("  Normalization done")


# ─────────────────────────────────────────
#  Step 3 — Split data
# ─────────────────────────────────────────
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.10, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.22, random_state=42, stratify=y_trainval
)

print(f"\nData split:")
print(f"  Train:      {len(X_train)} samples (~70%)")
print(f"  Validation: {len(X_val)}  samples (~20%)")
print(f"  Test:       {len(X_test)}  samples (~10%)")


# ─────────────────────────────────────────
#  Step 3 — Augmentation
# ─────────────────────────────────────────
def augment(sequence):
    augmented = []

    # Gaussian Noise — simulates natural hand shakiness
    noise = np.random.normal(0, 0.005, sequence.shape)
    augmented.append(sequence + noise)

    # Hand Flip — DISABLED. If re-enabled, must be updated for the new
    # 126-dim two-hand layout: flip x for BOTH hand slots (0::3 within
    # [0:63] and within [63:126]) AND swap the two 63-value slots,
    # since a horizontal flip turns "Hand 1 on the left" into
    # "Hand 1 on the right" from the camera's point of view.
    # flipped           = sequence.copy()
    # flipped[:, 0::3]  = -flipped[:, 0::3]
    # augmented.append(flipped)

    # Time Scaling — simulates faster signing
    fast = sequence[::2]
    fast = np.resize(fast, sequence.shape)
    augmented.append(fast)

    return augmented

X_aug, y_aug = [], []
for i in range(len(X_train)):
    extras = augment(X_train[i])
    for ex in extras:
        X_aug.append(ex)
        y_aug.append(y_train[i])

X = np.concatenate([X_train, np.array(X_aug)])
y = np.concatenate([y_train, np.array(y_aug)])

np.save("X_augemented.npy", X)
np.save("Y_augemented.npy", y)

print(f"  Total samples after augmentation: {len(X)}")


num_classes = len(WORDS)
y_train_cat = to_categorical(y_train, num_classes)
y_val_cat   = to_categorical(y_val,   num_classes)
y_test_cat  = to_categorical(y_test,  num_classes)

# ─────────────────────────────────────────
#  Step 5 — Build model
# ─────────────────────────────────────────
print("\nBuilding model...")

model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(FRAMES_COUNT, 126)),
    Dropout(0.3),
    LSTM(64, return_sequences=False),
    Dropout(0.3),
    Dense(64, activation="relu"),
    Dropout(0.3),
    Dense(num_classes, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ─────────────────────────────────────────
#  Step 6 — Train
# ─────────────────────────────────────────
print("\nTraining started...")

callbacks = [
    EarlyStopping(monitor="val_loss", patience=10,
                  restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                      patience=5, verbose=1)
]

history = model.fit(
    X_train, y_train_cat,
    validation_data=(X_val, y_val_cat),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=1
)

# ─────────────────────────────────────────
#  Step 7 — Evaluate on validation set AND test set
# ─────────────────────────────────────────
print("\n" + "="*50)
print("FINAL RESULTS")
print("="*50)

# ---- Validation evaluation ----
val_loss, val_acc = model.evaluate(X_val, y_val_cat, verbose=0)
y_val_pred = np.argmax(model.predict(X_val), axis=1)
val_exact_acc = np.mean(y_val_pred == y_val) * 100

print(f"\n[VALIDATION SET]")
print(f"  Validation Accuracy: {val_exact_acc:.2f}%  ({np.sum(y_val_pred == y_val)}/{len(y_val)} correct)")
print(f"  Validation Loss:     {val_loss:.4f}")
print("\n  Detailed report (Validation):")
print(classification_report(y_val, y_val_pred, target_names=WORDS))

# ---- Test evaluation ----
test_loss, test_acc = model.evaluate(X_test, y_test_cat, verbose=0)
y_test_pred = np.argmax(model.predict(X_test), axis=1)
test_exact_acc = np.mean(y_test_pred == y_test) * 100

print(f"\n[TEST SET]")
print(f"  Test Accuracy: {test_exact_acc:.2f}%  ({np.sum(y_test_pred == y_test)}/{len(y_test)} correct)")
print(f"  Test Loss:     {test_loss:.4f}")
print("\n  Detailed report (Test):")
print(classification_report(y_test, y_test_pred, target_names=WORDS))

# ─────────────────────────────────────────
#  Step 8 — Save model
# ─────────────────────────────────────────
model.save(MODEL_PATH)
print(f"\nModel saved: {MODEL_PATH}")

# ─────────────────────────────────────────
#  Step 9 — Figure 1: Training & Validation Curves (Accuracy + Loss)
# ─────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(history.history["accuracy"],     label="Train")
axes[0].plot(history.history["val_accuracy"], label="Validation")
axes[0].set_title("Training & Validation Accuracy")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Accuracy")
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].plot(history.history["loss"],     label="Train")
axes[1].plot(history.history["val_loss"], label="Validation")
axes[1].set_title("Training & Validation Loss")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Loss")
axes[1].legend()
axes[1].grid(alpha=0.3)

fig.suptitle(f"Training Curves  |  Final Validation Accuracy: {val_exact_acc:.2f}%", fontsize=13)
plt.tight_layout()
plt.savefig("1_training_curves.png", dpi=150)
plt.show()
print("\nSaved: 1_training_curves.png")

# ─────────────────────────────────────────
#  Step 10 — Figure 2: Validation Confusion Matrix
# ─────────────────────────────────────────
cm_val = confusion_matrix(y_val, y_val_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm_val, annot=True, fmt="d", xticklabels=WORDS,
            yticklabels=WORDS, cmap="Blues")
plt.title(f"Validation Confusion Matrix — Accuracy: {val_exact_acc:.2f}% "
          f"({np.sum(y_val_pred == y_val)}/{len(y_val)})")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("2_validation_confusion_matrix.png", dpi=150)
plt.show()
print("Saved: 2_validation_confusion_matrix.png")

# ─────────────────────────────────────────
#  Step 11 — Figure 3: Test Confusion Matrix
# ─────────────────────────────────────────
cm_test = confusion_matrix(y_test, y_test_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm_test, annot=True, fmt="d", xticklabels=WORDS,
            yticklabels=WORDS, cmap="Greens")
plt.title(f"Test Confusion Matrix — Accuracy: {test_exact_acc:.2f}% "
          f"({np.sum(y_test_pred == y_test)}/{len(y_test)})")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("3_test_confusion_matrix.png", dpi=150)
plt.show()
print("Saved: 3_test_confusion_matrix.png")

print("\n" + "="*50)
print("SUMMARY")
print("="*50)
print(f"  Validation Accuracy: {val_exact_acc:.2f}%")
print(f"  Test Accuracy:       {test_exact_acc:.2f}%")
print("="*50)
