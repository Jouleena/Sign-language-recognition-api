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
EPOCHS       = 50
BATCH_SIZE   = 16


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

        if sequence.shape == (FRAMES_COUNT, 63):
            X.append(sequence)
            y.append(label_map[word])
        else:
            print(f"  WARNING: wrong shape {file} -> {sequence.shape}")

X = np.array(X)
y = np.array(y)

print(f"  Total samples before augmentation: {len(X)}")

# ─────────────────────────────────────────
#  Step 2 — Normalize landmarks
# ─────────────────────────────────────────
for i in range(len(X)):
    for f in range(FRAMES_COUNT):
        wrist           = X[i, f, :3].copy()
        X[i, f, 0::3] -= wrist[0]
        X[i, f, 1::3] -= wrist[1]
        X[i, f, 2::3] -= wrist[2]

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

    # # Hand Flip — simulates left hand
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
    LSTM(128, return_sequences=True, input_shape=(FRAMES_COUNT, 63)),
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
#  Step 7 — Evaluate on test set
# ─────────────────────────────────────────
print("\nFinal Results:")

test_loss, test_acc = model.evaluate(X_test, y_test_cat, verbose=0)
print(f"  Test Accuracy: {test_acc * 100:.2f}%")
print(f"  Test Loss:     {test_loss:.4f}")

y_pred = np.argmax(model.predict(X_test), axis=1)
print("\nDetailed report:")
print(classification_report(y_test, y_pred, target_names=WORDS))

# ─────────────────────────────────────────
#  Step 8 — Save model
# ─────────────────────────────────────────
model.save(MODEL_PATH)
print(f"\nModel saved: {MODEL_PATH}")

# ─────────────────────────────────────────
#  Step 9 — Plot results
# ─────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].plot(history.history["accuracy"],     label="Train")
axes[0].plot(history.history["val_accuracy"], label="Validation")
axes[0].set_title("Accuracy")
axes[0].set_xlabel("Epoch")
axes[0].legend()

axes[1].plot(history.history["loss"],     label="Train")
axes[1].plot(history.history["val_loss"], label="Validation")
axes[1].set_title("Loss")
axes[1].set_xlabel("Epoch")
axes[1].legend()

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", xticklabels=WORDS,
            yticklabels=WORDS, ax=axes[2], cmap="Blues")
axes[2].set_title("Confusion Matrix")
axes[2].set_xlabel("Predicted")
axes[2].set_ylabel("Actual")

plt.tight_layout()
plt.savefig("training_results.png", dpi=150)
plt.show()
print("Results saved: training_results.png")
