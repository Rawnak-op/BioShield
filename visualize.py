import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

LABELS = ["Clean", "Powerline", "Switching", "Motion", "Broadband"]
NUM_SAMPLES = 10   # <<< CHANGE HERE

model = tf.keras.models.load_model("emi_ecg_model.h5")
X_test = np.load("X_test.npy")
y_test = np.load("y_test.npy")

y_pred_probs = model.predict(X_test)
y_pred = y_pred_probs.argmax(axis=1)
y_true = y_test.argmax(axis=1)

start = np.random.randint(0, len(X_test) - NUM_SAMPLES)

plt.figure(figsize=(14, 10))  # taller figure for 10 plots

for i in range(NUM_SAMPLES):
    idx = start + i

    plt.subplot(NUM_SAMPLES, 1, i + 1)
    plt.plot(X_test[idx].squeeze())

    plt.title(
        f"Sample {idx} | "
        f"True: {LABELS[y_true[idx]]} | "
        f"Pred: {LABELS[y_pred[idx]]} | "
        f"Conf: {y_pred_probs[idx].max():.2f}"
    )
    plt.grid()

plt.tight_layout()
plt.show()
