from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, GlobalAveragePooling1D, Dense

def build_model(input_length, num_classes):
    model = Sequential([
        Conv1D(16, 7, activation="relu", input_shape=(input_length,1)),
        MaxPooling1D(2),
        Conv1D(32, 5, activation="relu"),
        MaxPooling1D(2),
        Conv1D(64, 3, activation="relu"),
        GlobalAveragePooling1D(),
        Dense(num_classes, activation="softmax")
    ])
    return model
