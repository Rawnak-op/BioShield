import numpy as np
import tensorflow as tf
import time
from collections import deque

from realtime_stream_sim import ecg_stream

# ---------------- CONFIG ----------------
FS = 500

WINDOW_SEC = 2.0
WINDOW_SIZE = int(FS * WINDOW_SEC)    # 1000 samples

STEP_SEC = 0.5
STEP_SIZE = int(FS * STEP_SEC)        # 250 samples

LABELS = ["Clean", "Powerline", "Switching", "Motion", "Broadband"]

# ---------------- USER INPUT ----------------
print("\nSelect signal type to simulate:")
print("0 → Clean ECG")
print("1 → Powerline EMI")
print("2 → Switching EMI")
print("3 → Motion Artifact")
print("4 → Broadband EMI")

while True:
    try:
        TRUE_LABEL = int(input("\nEnter choice (0–4): "))
        if TRUE_LABEL in range(5):
            break
        else:
            print("❌ Invalid choice. Enter number between 0 and 4.")
    except ValueError:
        print("❌ Invalid input. Enter an integer.")

print(f"\nSimulating: {LABELS[TRUE_LABEL]}\n")

# ---------------- LOAD MODEL ----------------
model = tf.keras.models.load_model("emi_ecg_model.h5")

# ---------------- STREAM ----------------
stream = ecg_stream(label=TRUE_LABEL)

buffer = deque(maxlen=WINDOW_SIZE)
sample_count = 0

print("==============================")
print(" REAL-TIME EMI DETECTION STARTED ")
print("==============================\n")

# ---------------- REAL-TIME LOOP ----------------
while True:
    sample, true_lbl = next(stream)
    buffer.append(sample)
    sample_count += 1

    # wait until buffer is full
    if len(buffer) < WINDOW_SIZE:
        continue

    # sliding window inference
    if sample_count % STEP_SIZE == 0:
        window = np.array(buffer)

        # normalize
        window = window / np.max(np.abs(window))
        window = window.reshape(1, WINDOW_SIZE, 1)

        probs = model.predict(window, verbose=0)[0]
        pred_idx = np.argmax(probs)
        confidence = np.max(probs)

        print(
            f"Actual: {LABELS[true_lbl]:<10} | "
            f"Detected: {LABELS[pred_idx]:<10} | "
            f"Confidence: {confidence:.2f}"
        )

        time.sleep(STEP_SIZE / FS)
