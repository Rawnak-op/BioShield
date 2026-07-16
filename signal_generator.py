import numpy as np

FS = 500
DURATION = 2.0
N = int(FS * DURATION)
NUM_CLASSES = 5

t = np.linspace(0, DURATION, N, endpoint=False)

#ECG GENERATION
def generate_ecg(t):
    hr = np.random.uniform(55, 90)
    rr = 60 / hr
    ecg = np.zeros_like(t)

    def g(x, mu, s, a):
        return a * np.exp(-(x - mu)**2 / (2*s**2))

    beats = np.arange(0, t[-1], rr)
    for b in beats:
        ecg += g(t, b+0.10, 0.02,  np.random.uniform(0.05,0.12))
        ecg += g(t, b+0.20, 0.01, -0.15)
        ecg += g(t, b+0.22, 0.012, np.random.uniform(0.8,1.2))
        ecg += g(t, b+0.25, 0.01, -0.25)
        ecg += g(t, b+0.40, 0.04,  np.random.uniform(0.2,0.35))

    return ecg

#BASE NOISE
def baseline_wander():
    f = np.random.uniform(0.1, 0.3)
    return 0.05 * np.sin(2*np.pi*f*t)

def sensor_noise():
    return 0.01 * np.random.randn(N)

#EMI NOISE TYPES
def powerline_noise():
    return 0.5 * np.sin(2*np.pi*50*t)

def switching_noise():
    amp = np.random.uniform(0.02, 0.2)
    freq = np.random.uniform(1000, 20000)
    noise = np.zeros_like(t)

    burst_len = np.random.randint(20, 80)
    for i in range(0, len(t), burst_len * 4):
        noise[i:i + burst_len] = np.sin(2 * np.pi * freq * t[i:i + burst_len])

    return amp * noise


def motion_noise():
    amp = 0.3
    n = np.random.randn(N)
    return amp * np.convolve(n, np.ones(15)/15, mode="same")

def broadband_noise():
    return 0.2 * np.random.randn(N)

#SAMPLE GENERATION
def generate_sample(label, return_clean=False):
    clean_ecg = generate_ecg(t) + baseline_wander() + sensor_noise()
    signal = clean_ecg.copy()

    if label == 1:
        signal += powerline_noise()
    elif label == 2:
        signal += switching_noise()
    elif label == 3:
        signal += motion_noise()
    elif label == 4:
        signal += broadband_noise()

    # normalize BOTH using same scale
    scale = np.max(np.abs(signal))
    signal /= scale
    clean_ecg /= scale

    if return_clean:
        return signal, clean_ecg
    return signal
# DATASET CREATION
def create_dataset(n_samples=3000):
    X = []
    y = []

    for _ in range(n_samples):
        label = np.random.randint(0, NUM_CLASSES)
        signal = generate_sample(label)
        X.append(signal)
        y.append(label)

    return np.array(X), np.array(y)
 