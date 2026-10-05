import streamlit as st
import numpy as np
import tensorflow as tf
import time
from collections import deque
import matplotlib.pyplot as plt
from scipy.signal import iirnotch, filtfilt
import pywt

from realtime_stream_sim import ecg_stream

#CONFIG
FS = 500
WINDOW_SIZE = 1000        # 2 seconds
STEP_SIZE = 250           # update every 0.5 sec

LABELS = ["Clean", "Powerline", "Switching", "Motion", "Broadband"]

#METRICS
def rms(x):
    return np.sqrt(np.mean(x ** 2))

def snr_db(clean, noisy):
    return 20 * np.log10(rms(clean) / rms(noisy - clean))

#CANCELLERS
def notch_50(signal):
    b, a = iirnotch(50 / (FS / 2), Q=30)
    return filtfilt(b, a, signal)

def wavelet_denoise(signal):
    coeffs = pywt.wavedec(signal, "db6", level=4)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    thresh = sigma * np.sqrt(2 * np.log(len(signal)))  
    coeffs_f = [coeffs[0]] + [pywt.threshold(c, thresh, mode="soft") for c in coeffs[1:]]
    return pywt.waverec(coeffs_f, "db6")[:len(signal)]


# STREAMLIT UI
st.set_page_config(layout="wide")
st.title("ECG EMI Detection, Confidence & Quantitative Evaluation")

if "run" not in st.session_state:
    st.session_state.run = False

emi_choice = st.sidebar.selectbox(
    "Select EMI Type (Simulator)",
    range(5),
    format_func=lambda x: LABELS[x]
)

if st.sidebar.button("▶ START"):
    st.session_state.run = True
if st.sidebar.button("⏹ STOP"):
    st.session_state.run = False

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("emi_ecg_model.h5")

model = load_model()

plot_box = st.empty()
conf_box = st.empty()
info_box = st.empty()

# STATE MANAGEMENT FOR STREAMING
if "current_emi" not in st.session_state or st.session_state.current_emi != emi_choice:
    st.session_state.current_emi = emi_choice
    st.session_state.stream = ecg_stream(emi_choice)
    st.session_state.buf = deque(maxlen=WINDOW_SIZE)
    st.session_state.clean_buf = deque(maxlen=WINDOW_SIZE)

# REAL-TIME CYCLE (Re-runs automatically instead of blocking while loop)
if st.session_state.run:
    # 1. Fill the buffer with the next STEP_SIZE samples
    # If the buffer isn't full yet (startup), fill it all the way to WINDOW_SIZE
    samples_to_fetch = WINDOW_SIZE if len(st.session_state.buf) == 0 else STEP_SIZE
    
    for _ in range(samples_to_fetch):
        noisy_sample, clean_sample, true_lbl = next(st.session_state.stream)
        st.session_state.buf.append(noisy_sample)
        st.session_state.clean_buf.append(clean_sample)

    raw = np.array(st.session_state.buf)
    clean = np.array(st.session_state.clean_buf)

    # 2. ML DETECTION
    ml_input = raw / np.max(np.abs(raw))
    probs = model.predict(ml_input.reshape(1, -1, 1), verbose=0)[0]
    pred = np.argmax(probs)

    true_emi = LABELS[true_lbl]
    detected_emi = LABELS[pred]

    # 3. CONFIDENCE DISTRIBUTION
    confidence = {LABELS[i]: probs[i] * 100 for i in range(len(LABELS))}
    sorted_conf = sorted(confidence.items(), key=lambda x: x[1], reverse=True)
    top1, top2 = sorted_conf[0], sorted_conf[1]

    # 4. DECISION & ACTION
    if detected_emi == "Powerline":
        cleaned = notch_50(raw)
        action = "50 Hz Notch Filter Applied"
    elif detected_emi == "Switching":
        cleaned = wavelet_denoise(raw)
        action = "Wavelet Denoising Applied"
    elif detected_emi == "Motion":
        cleaned = raw
        action = "Detection Only (Cancellation Unsafe)"
    elif detected_emi == "Broadband":
        cleaned = raw
        action = "Detection Only (Broadband Overlaps ECG)"
    else:
        cleaned = raw
        action = "No Action Needed"

    # 5. METRICS
    rms_before = rms(raw - clean)
    rms_after = rms(cleaned - clean)
    rms_reduction = 100 * (rms_before - rms_after) / rms_before if rms_before != 0 else 0

    snr_before = snr_db(clean, raw)
    snr_after = snr_db(clean, cleaned)
    snr_gain = snr_after - snr_before

    # 6. ECG PLOTS
    fig, ax = plt.subplots(2, 1, figsize=(10, 4), sharex=True)
    ax[0].plot(raw)
    ax[0].set_title("Noisy ECG (Input)")
    ax[0].grid()

    ax[1].plot(cleaned)
    ax[1].set_title("ECG After System Action")
    ax[1].grid()

    plot_box.pyplot(fig)
    plt.close(fig)

    # 7. CONFIDENCE BAR PLOT
    fig_c, ax_c = plt.subplots(figsize=(6, 3))
    ax_c.bar(confidence.keys(), confidence.values())
    ax_c.set_ylim(0, 100)
    ax_c.set_ylabel("Confidence (%)")
    ax_c.set_title("EMI Detection Confidence Distribution")
    ax_c.grid(axis="y")

    conf_box.pyplot(fig_c)
    plt.close(fig_c)

    # 8. INFO PANEL
    info_box.markdown(f"""
## 🧠 EMI Interpretation

- **Actual EMI (Simulator):** `{true_emi}`
- **Detected EMI (Top-1):** `{top1[0]} ({top1[1]:.1f}%)`
- **Second Likely EMI:** `{top2[0]} ({top2[1]:.1f}%)`
- **System Action:** `{action}`

---

## 📊 Quantitative EMI Reduction

- **RMS Noise Reduction:** `{rms_reduction:.2f} %`
- **SNR Before:** `{snr_before:.2f} dB`
- **SNR After:** `{snr_after:.2f} dB`
- **SNR Improvement:** `{snr_gain:.2f} dB`
""")

    # 9. TRIGGER NEXT FRAME (Yields back to Streamlit server gracefully)
    time.sleep(0.5)
    st.rerun()
