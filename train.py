import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.utils import to_categorical
from keras.optimizers import Adam

from signal_generator import create_dataset, N, NUM_CLASSES
from model import build_model

LABELS = ["Clean", "Powerline", "Switching", "Motion", "Broadband"]

print("Generating dataset...")
X, y = create_dataset(3000)

X = X[..., np.newaxis]
y_cat = to_categorical(y, NUM_CLASSES)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_cat, test_size=0.2, random_state=42
)

model = build_model(N, NUM_CLASSES)
model.compile(
    optimizer=Adam(),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

model.fit(
    X_train, y_train,
    epochs=30,
    batch_size=32,
    validation_split=0.2
)

y_pred = model.predict(X_test).argmax(axis=1)
y_true = y_test.argmax(axis=1)

print(classification_report(y_true, y_pred, target_names=LABELS))

model.save("emi_ecg_model.h5")

np.save("X_test.npy", X_test)
np.save("y_test.npy", y_test)
