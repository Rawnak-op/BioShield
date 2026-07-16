# BioShield: AI-Based EMI Detection in ECG Signals

## Overview

BioShield is a software-based framework developed to detect and analyze electromagnetic interference (EMI) in ECG signals using deep learning. The project simulates different types of EMI, classifies the interference using a 1D Convolutional Neural Network (CNN), and applies EMI-specific mitigation techniques where appropriate.

The objective is to improve ECG signal quality while preserving clinically important waveform characteristics.

---

## Features

- Synthetic ECG signal generation using Gaussian P-Q-R-S-T wave modeling
- Simulation of multiple EMI types:
  - Clean ECG
  - Powerline interference
  - Switching interference
  - Motion artifacts
  - Broadband noise
- EMI classification using a lightweight 1D CNN
- Real-time inference simulation
- EMI-specific mitigation
- Quantitative evaluation using RMS Noise Reduction and SNR Improvement
- Streamlit-based visualization interface

---

## Project Structure

```
BioShield/
│
├── app.py
├── model.py
├── signal_generator.py
├── realtime_stream_sim.py
├── realtime_inference.py
├── train.py
├── visualize.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Technologies Used

- Python
- TensorFlow / Keras
- NumPy
- SciPy
- Scikit-learn
- PyWavelets
- Matplotlib
- Streamlit

---

## Model Architecture

The classifier is a lightweight 1D CNN consisting of:

- Conv1D (16 filters, kernel size 7)
- MaxPooling1D
- Conv1D (32 filters, kernel size 5)
- MaxPooling1D
- Conv1D (64 filters, kernel size 3)
- Global Average Pooling
- Dense Softmax Output Layer

The model classifies ECG signals into five categories:

1. Clean
2. Powerline
3. Switching
4. Motion
5. Broadband

---

## Running the Project

Install the required packages:

```bash
pip install -r requirements.txt
```

Train the model:

```bash
python train.py
```

Launch the application:

```bash
streamlit run app.py
```

---

## Results

The model performs EMI classification on simulated ECG signals and displays:

- Actual EMI type
- Predicted EMI type
- Prediction confidence
- ECG before mitigation
- ECG after mitigation
- RMS Noise Reduction
- SNR Before
- SNR After
- SNR Improvement

Powerline interference is significantly reduced using a notch filter, while other EMI types are handled conservatively to preserve ECG morphology.

---

## Future Work

- Evaluation using real ECG datasets
- Hardware implementation with AD8232 and ESP32
- Adaptive filtering techniques
- Transformer-based temporal models
- Deployment on embedded systems

---

## License

This project is intended for academic and research purposes.
