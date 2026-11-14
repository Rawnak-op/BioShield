import os
import sys
import numpy as np
import tensorflow as tf
import pywt
from scipy.signal import iirnotch, filtfilt
from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from signal_generator import generate_sample, FS, NUM_CLASSES, N

# ── CONFIGURATION & CONSTANTS ────────────────────────────────────────────────
LABELS = ["Clean", "Powerline", "Switching", "Motion", "Broadband"]
MODEL_PATH = os.path.join(BASE_DIR, "emi_ecg_model.h5")

app = FastAPI(
    title="BioShield: AI-Guided ECG EMI Detection & Mitigation API",
    description="Production REST API and real-time biomedical DSP microservice for ECG signal quality enhancement.",
    version="2.0.0"
)

# Enable CORS for external web dashboards, medical portals, and mobile clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── MODEL INITIALIZATION ─────────────────────────────────────────────────────
print(f"Loading trained 1D-CNN BioShield model from {MODEL_PATH}...")
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    # Warmup forward pass
    dummy = np.zeros((1, N, 1), dtype=np.float32)
    model.predict(dummy, verbose=0)
    print("Model loaded and warmed up successfully.")
except Exception as e:
    print(f"Warning: Failed to load {MODEL_PATH}: {e}")
    model = None

# ── DSP FILTERING & METRICS UTILITIES ─────────────────────────────────────────
def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x ** 2)))

def snr_db(clean: np.ndarray, noisy: np.ndarray) -> float:
    noise_rms = rms(noisy - clean)
    clean_rms = rms(clean)
    if noise_rms == 0:
        return 100.0
    if clean_rms == 0:
        return 0.0
    return float(20.0 * np.log10(clean_rms / noise_rms))

def notch_50(signal: np.ndarray) -> np.ndarray:
    """Forward-backward zero-phase 2nd-order IIR Notch filter (Q=30)"""
    b, a = iirnotch(50.0 / (FS / 2.0), Q=30.0)
    return filtfilt(b, a, signal)

def wavelet_denoise(signal: np.ndarray) -> np.ndarray:
    """4-level Daubechies-6 DWT with Donoho's VisuShrink soft thresholding"""
    coeffs = pywt.wavedec(signal, "db6", level=4)
    # Estimate noise standard deviation from detail coefficients level 1
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    thresh = sigma * np.sqrt(2.0 * np.log(len(signal)))
    coeffs_f = [coeffs[0]] + [pywt.threshold(c, thresh, mode="soft") for c in coeffs[1:]]
    reconstructed = pywt.waverec(coeffs_f, "db6")
    return reconstructed[:len(signal)]

def process_ecg_window(raw: np.ndarray, clean: Optional[np.ndarray] = None, true_label_idx: Optional[int] = None) -> dict:
    if model is None:
        raise HTTPException(status_code=500, detail="Neural network model not initialized")

    # Co-normalize or peak-normalize for CNN inference
    peak = np.max(np.abs(raw))
    norm_factor = peak if peak > 1e-6 else 1.0
    ml_input = (raw / norm_factor).reshape(1, -1, 1).astype(np.float32)

    probs = model.predict(ml_input, verbose=0)[0]
    pred_idx = int(np.argmax(probs))
    detected_emi = LABELS[pred_idx]

    confidence_dist = {LABELS[i]: round(float(probs[i] * 100.0), 2) for i in range(len(LABELS))}

    # Apply Selective DSP Mitigation Policy
    if detected_emi == "Powerline":
        cleaned = notch_50(raw)
        action = "50 Hz Notch Filter Applied"
        safety_status = "ACTIVE_MITIGATION"
    elif detected_emi == "Switching":
        cleaned = wavelet_denoise(raw)
        action = "Wavelet Denoising Applied"
        safety_status = "ACTIVE_MITIGATION"
    elif detected_emi == "Motion":
        cleaned = raw.copy()
        action = "Detection Only (In-Band Motion: ST-Segment Preserved)"
        safety_status = "CLINICAL_SAFETY_FALLBACK"
    elif detected_emi == "Broadband":
        cleaned = raw.copy()
        action = "Detection Only (Broadband Overlaps ECG Spectrum)"
        safety_status = "CLINICAL_SAFETY_FALLBACK"
    else:  # Clean
        cleaned = raw.copy()
        action = "Passthrough (Signal Clean, No Action Needed)"
        safety_status = "CLEAN_PASSTHROUGH"

    # Quantitative Metrics (if ground truth clean signal is available)
    if clean is not None:
        rms_before = rms(raw - clean)
        rms_after = rms(cleaned - clean)
        rms_reduction = round(float(100.0 * (rms_before - rms_after) / rms_before), 2) if rms_before > 1e-7 else 0.0
        
        snr_before = round(snr_db(clean, raw), 2)
        snr_after = round(snr_db(clean, cleaned), 2)
        snr_gain = round(snr_after - snr_before, 2)
    else:
        rms_before, rms_after, rms_reduction = None, None, None
        snr_before, snr_after, snr_gain = None, None, None

    true_emi_str = LABELS[true_label_idx] if (true_label_idx is not None and 0 <= true_label_idx < 5) else "Unknown"

    return {
        "true_emi": true_emi_str,
        "detected_emi": detected_emi,
        "prediction_confidence": confidence_dist[detected_emi],
        "confidence_distribution": confidence_dist,
        "action": action,
        "safety_status": safety_status,
        "metrics": {
            "rms_noise_before": rms_before,
            "rms_noise_after": rms_after,
            "rms_reduction_percent": rms_reduction,
            "snr_before_db": snr_before,
            "snr_after_db": snr_after,
            "snr_gain_db": snr_gain
        },
        "signal_data": {
            "raw": [round(float(v), 4) for v in raw],
            "cleaned": [round(float(v), 4) for v in cleaned],
            "clean_reference": [round(float(v), 4) for v in clean] if clean is not None else None
        }
    }

# ── REQUEST / RESPONSE SCHEMAS ───────────────────────────────────────────────
class ProcessRequest(BaseModel):
    signal: List[float] = Field(..., description="ECG sample window (ideally 1000 samples at 500 Hz)")
    clean_reference: Optional[List[float]] = Field(None, description="Optional ground-truth clean ECG for metrics calculation")
    true_label: Optional[int] = Field(None, ge=0, le=4, description="Optional simulated class index (0-4)")

# ── API ENDPOINTS ────────────────────────────────────────────────────────────
@app.get("/health")
def health_check():
    """Health check endpoint for Render, UptimeRobot, and cloud load balancers."""
    return {
        "status": "healthy",
        "service": "BioShield REST API",
        "model_loaded": model is not None,
        "timestamp": time.time()
    }

@app.get("/api/labels")
def get_labels():
    """Returns available EMI class mapping."""
    return {
        "classes": {i: LABELS[i] for i in range(len(LABELS))}
    }

@app.get("/api/step")
def simulate_step(emi_type: int = Query(0, ge=0, le=4, description="EMI class index (0: Clean, 1: Powerline, 2: Switching, 3: Motion, 4: Broadband)")):
    """Simulates a 2-second ECG window with requested EMI, classifies it, applies DSP filtering, and returns metrics & waveforms."""
    raw, clean = generate_sample(emi_type, return_clean=True)
    result = process_ecg_window(raw, clean=clean, true_label_idx=emi_type)
    return JSONResponse(content=result)

@app.post("/api/process")
def process_custom_signal(payload: ProcessRequest):
    """Processes an arbitrary ECG segment supplied via POST."""
    raw_arr = np.array(payload.signal, dtype=np.float32)
    if len(raw_arr) != N:
        # Resize or pad/crop to window length N
        if len(raw_arr) < N:
            raw_arr = np.pad(raw_arr, (0, N - len(raw_arr)), mode='edge')
        else:
            raw_arr = raw_arr[:N]

    clean_arr = np.array(payload.clean_reference, dtype=np.float32) if payload.clean_reference else None
    if clean_arr is not None and len(clean_arr) != N:
        if len(clean_arr) < N:
            clean_arr = np.pad(clean_arr, (0, N - len(clean_arr)), mode='edge')
        else:
            clean_arr = clean_arr[:N]

    result = process_ecg_window(raw_arr, clean=clean_arr, true_label_idx=payload.true_label)
    return JSONResponse(content=result)

# ── HIGH-PERFORMANCE WEB DASHBOARD (SERVED AT /) ─────────────────────────────
@app.get("/", response_class=HTMLResponse)
def index_dashboard():
    """Ultra-responsive, GPU-accelerated client-side dashboard with live REST API streaming."""
    return HTMLResponse(content=DASHBOARD_HTML)

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>BioShield: Clinical ECG EMI Mitigation Console</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-base: #0B0F17;
            --bg-card: #121824;
            --bg-card-hover: #182233;
            --border-subtle: #1F293D;
            --text-main: #E2E8F0;
            --text-muted: #8292AB;
            --accent-cyan: #00F0FF;
            --accent-green: #10B981;
            --accent-red: #EF4444;
            --accent-amber: #F59E0B;
            --accent-purple: #8B5CF6;
        }

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }

        body {
            background-color: var(--bg-base);
            color: var(--text-main);
            min-height: 100vh;
            padding: 24px;
        }

        .container {
            max-width: 1400px;
            margin: 0 auto;
        }

        /* HEADER */
        header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding-bottom: 20px;
            border-bottom: 1px solid var(--border-subtle);
            margin-bottom: 24px;
        }

        .title-group h1 {
            font-size: 1.5rem;
            font-weight: 700;
            display: flex;
            align-items: center;
            gap: 10px;
            color: #FFFFFF;
        }

        .badge-pulse {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            font-size: 0.75rem;
            padding: 4px 10px;
            border-radius: 9999px;
            background: rgba(16, 185, 129, 0.15);
            color: var(--accent-green);
            border: 1px solid rgba(16, 185, 129, 0.3);
            font-family: 'JetBrains Mono', monospace;
        }

        .pulse-dot {
            width: 7px;
            height: 7px;
            background: var(--accent-green);
            border-radius: 50%;
            animation: pulse 1.5s infinite;
        }

        @keyframes pulse {
            0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
            70% { transform: scale(1); box-shadow: 0 0 0 6px rgba(16, 185, 129, 0); }
            100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
        }

        .title-group p {
            color: var(--text-muted);
            font-size: 0.85rem;
            margin-top: 4px;
        }

        .header-links a {
            color: var(--accent-cyan);
            text-decoration: none;
            font-size: 0.85rem;
            font-weight: 500;
            padding: 8px 14px;
            background: rgba(0, 240, 255, 0.08);
            border: 1px solid rgba(0, 240, 255, 0.25);
            border-radius: 8px;
            transition: all 0.2s;
        }

        .header-links a:hover {
            background: rgba(0, 240, 255, 0.18);
        }

        /* CONTROLS BAR */
        .controls-card {
            background: var(--bg-card);
            border: 1px solid var(--border-subtle);
            border-radius: 12px;
            padding: 16px 20px;
            display: flex;
            flex-wrap: wrap;
            gap: 16px;
            align-items: center;
            margin-bottom: 24px;
        }

        .control-item {
            display: flex;
            align-items: center;
            gap: 10px;
        }

        label {
            font-size: 0.85rem;
            font-weight: 600;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        select {
            background: #0D131F;
            border: 1px solid var(--border-subtle);
            color: #FFFFFF;
            padding: 9px 16px;
            border-radius: 8px;
            font-size: 0.9rem;
            outline: none;
            cursor: pointer;
            transition: border-color 0.2s;
        }

        select:focus {
            border-color: var(--accent-cyan);
        }

        .btn {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 10px 20px;
            border-radius: 8px;
            font-size: 0.9rem;
            font-weight: 600;
            border: none;
            cursor: pointer;
            transition: all 0.2s;
        }

        .btn-start {
            background: #10B981;
            color: #064E3B;
            box-shadow: 0 4px 14px rgba(16, 185, 129, 0.3);
        }

        .btn-start:hover {
            background: #059669;
            color: #FFFFFF;
        }

        .btn-stop {
            background: #334155;
            color: #94A3B8;
        }

        .btn-stop.active {
            background: #EF4444;
            color: #FFFFFF;
            box-shadow: 0 4px 14px rgba(239, 68, 68, 0.3);
        }

        .btn-step {
            background: #1E293B;
            color: #94A3B8;
            border: 1px solid var(--border-subtle);
        }

        .btn-step:hover {
            background: #27354A;
            color: #FFFFFF;
        }

        /* METRICS GRID */
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
            gap: 16px;
            margin-bottom: 24px;
        }

        .metric-card {
            background: var(--bg-card);
            border: 1px solid var(--border-subtle);
            border-radius: 12px;
            padding: 18px;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
        }

        .metric-header {
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-muted);
            margin-bottom: 8px;
        }

        .metric-value {
            font-size: 1.5rem;
            font-weight: 700;
            color: #FFFFFF;
            font-family: 'JetBrains Mono', monospace;
        }

        .metric-sub {
            font-size: 0.75rem;
            color: var(--text-muted);
            margin-top: 6px;
        }

        .badge-status {
            display: inline-block;
            font-size: 0.8rem;
            padding: 3px 8px;
            border-radius: 6px;
            font-weight: 600;
            font-family: 'JetBrains Mono', monospace;
        }

        .status-Powerline { background: rgba(239, 68, 68, 0.2); color: #FCA5A5; border: 1px solid #EF4444; }
        .status-Switching { background: rgba(245, 158, 11, 0.2); color: #FCD34D; border: 1px solid #F59E0B; }
        .status-Motion { background: rgba(139, 92, 246, 0.2); color: #C4B5FD; border: 1px solid #8B5CF6; }
        .status-Broadband { background: rgba(236, 72, 153, 0.2); color: #F472B6; border: 1px solid #EC4899; }
        .status-Clean { background: rgba(16, 185, 129, 0.2); color: #6EE7B7; border: 1px solid #10B981; }

        /* CHARTS SECTION */
        .charts-container {
            display: grid;
            grid-template-columns: 2fr 1fr;
            gap: 20px;
        }

        @media (max-width: 1024px) {
            .charts-container {
                grid-template-columns: 1fr;
            }
        }

        .chart-card {
            background: var(--bg-card);
            border: 1px solid var(--border-subtle);
            border-radius: 12px;
            padding: 20px;
        }

        .chart-title {
            font-size: 0.95rem;
            font-weight: 600;
            margin-bottom: 16px;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }

        .canvas-box {
            position: relative;
            width: 100%;
            height: 220px;
            margin-bottom: 16px;
        }

        .canvas-box-bar {
            position: relative;
            width: 100%;
            height: 380px;
        }

        /* CLINICAL NOTICE BANNER */
        .notice-banner {
            margin-top: 24px;
            padding: 14px 18px;
            border-radius: 10px;
            background: rgba(30, 41, 59, 0.6);
            border: 1px solid var(--border-subtle);
            font-size: 0.8rem;
            color: var(--text-muted);
            display: flex;
            align-items: center;
            gap: 12px;
        }

        .notice-banner b {
            color: var(--accent-cyan);
        }
    </style>
</head>
<body>
    <div class="container">
        <!-- HEADER -->
        <header>
            <div class="title-group">
                <h1>
                    <span>BioShield Console</span>
                    <span class="badge-pulse"><span class="pulse-dot"></span> REST API Active</span>
                </h1>
                <p>AI-Guided 1D-CNN Real-Time Electromagnetic Interference Detection & Adaptive DSP Mitigation</p>
            </div>
            <div class="header-links">
                <a href="/docs" target="_blank">📖 Interactive REST API Docs</a>
            </div>
        </header>

        <!-- CONTROLS -->
        <div class="controls-card">
            <div class="control-item">
                <label for="emiSelect">Simulator EMI Type:</label>
                <select id="emiSelect">
                    <option value="0">Clean ECG</option>
                    <option value="1" selected>Powerline (50 Hz AC Hum)</option>
                    <option value="2">Switching Noise (SMPS 1-20 kHz)</option>
                    <option value="3">Motion Artifact (0.1-5 Hz Wander)</option>
                    <option value="4">Broadband Noise (Gaussian White)</option>
                </select>
            </div>

            <div class="control-item">
                <button id="btnStart" class="btn btn-start" onclick="startStreaming()">▶ START STREAM</button>
                <button id="btnStop" class="btn btn-stop" onclick="stopStreaming()">⏹ STOP</button>
                <button id="btnStep" class="btn btn-step" onclick="fetchStep()">⏭ STEP ONCE</button>
            </div>

            <div class="control-item" style="margin-left: auto;">
                <label for="speedSelect">Interval:</label>
                <select id="speedSelect" onchange="updateInterval()">
                    <option value="500">500 ms (Fast)</option>
                    <option value="1000" selected>1000 ms (Standard)</option>
                    <option value="2000">2000 ms (Slow)</option>
                </select>
            </div>
        </div>

        <!-- KPI METRICS GRID -->
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-header">Detected Interference</div>
                <div class="metric-value" id="valDetected">--</div>
                <div class="metric-sub" id="valActual">Actual: --</div>
            </div>

            <div class="metric-card">
                <div class="metric-header">Detection Confidence</div>
                <div class="metric-value" id="valConfidence" style="color: var(--accent-cyan);">-- %</div>
                <div class="metric-sub" id="valTop2">Runner-up: --</div>
            </div>

            <div class="metric-card">
                <div class="metric-header">RMS Noise Reduction</div>
                <div class="metric-value" id="valRms" style="color: var(--accent-green);">-- %</div>
                <div class="metric-sub" id="valRmsSub">Noise attenuation</div>
            </div>

            <div class="metric-card">
                <div class="metric-header">SNR Improvement</div>
                <div class="metric-value" id="valSnr" style="color: #60A5FA;">-- dB</div>
                <div class="metric-sub" id="valSnrSub">Before: -- | After: --</div>
            </div>
        </div>

        <!-- CHARTS SECTION -->
        <div class="charts-container">
            <!-- WAVEFORMS -->
            <div class="chart-card">
                <div class="chart-title">
                    <span>🫀 Live ECG Waveforms (500 Hz Sampling)</span>
                    <span id="txtAction" style="font-size: 0.8rem; color: var(--accent-cyan); font-family: 'JetBrains Mono';">System Action: Idle</span>
                </div>
                
                <div class="canvas-box">
                    <canvas id="rawChart"></canvas>
                </div>
                <div class="canvas-box">
                    <canvas id="cleanChart"></canvas>
                </div>
            </div>

            <!-- CONFIDENCE DISTRIBUTION -->
            <div class="chart-card">
                <div class="chart-title">
                    <span>🧠 Model Confidence Distribution</span>
                    <span style="font-size: 0.75rem; color: var(--text-muted);">1D-CNN + GAP</span>
                </div>
                <div class="canvas-box-bar">
                    <canvas id="barChart"></canvas>
                </div>
            </div>
        </div>

        <!-- CLINICAL NOTICE -->
        <div class="notice-banner">
            <span>🛡️</span>
            <div>
                <b>Clinical Safety Notice:</b> For Motion Artifacts and Broadband Noise, BioShield executes a <b>Detection-Only safety policy</b>. The signal is preserved without filtering to eliminate the risk of artificially distorting the ST-segment or hallucinating diagnostic cardiac features.
            </div>
        </div>
    </div>

    <script>
        let streamTimer = null;
        let isRunning = false;

        // Initialize Chart.js
        const labels = Array.from({length: 1000}, (_, i) => (i * 2) + "ms");

        const commonOptions = {
            responsive: true,
            maintainAspectRatio: false,
            animation: false,
            scales: {
                x: { display: false },
                y: {
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    ticks: { color: '#64748B', font: { family: 'JetBrains Mono', size: 10 } }
                }
            },
            plugins: {
                legend: { display: true, labels: { color: '#94A3B8', boxWidth: 12, font: { size: 11 } } },
                tooltip: { enabled: false }
            }
        };

        const rawCtx = document.getElementById('rawChart').getContext('2d');
        const rawChart = new Chart(rawCtx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Noisy ECG (Raw Input with Simulated EMI)',
                    data: new Array(1000).fill(0),
                    borderColor: '#EF4444',
                    borderWidth: 1.5,
                    pointRadius: 0,
                    tension: 0.1
                }]
            },
            options: commonOptions
        });

        const cleanCtx = document.getElementById('cleanChart').getContext('2d');
        const cleanChart = new Chart(cleanCtx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Cleaned ECG (After BioShield Adaptive Mitigation)',
                    data: new Array(1000).fill(0),
                    borderColor: '#10B981',
                    borderWidth: 1.5,
                    pointRadius: 0,
                    tension: 0.1
                }]
            },
            options: commonOptions
        });

        const barCtx = document.getElementById('barChart').getContext('2d');
        const barChart = new Chart(barCtx, {
            type: 'bar',
            data: {
                labels: ['Clean', 'Powerline', 'Switching', 'Motion', 'Broadband'],
                datasets: [{
                    data: [0, 0, 0, 0, 0],
                    backgroundColor: [
                        'rgba(16, 185, 129, 0.7)',
                        'rgba(239, 68, 68, 0.7)',
                        'rgba(245, 158, 11, 0.7)',
                        'rgba(139, 92, 246, 0.7)',
                        'rgba(236, 72, 153, 0.7)'
                    ],
                    borderColor: ['#10B981', '#EF4444', '#F59E0B', '#8B5CF6', '#EC4899'],
                    borderWidth: 1,
                    borderRadius: 6
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: { duration: 300 },
                indexAxis: 'y',
                scales: {
                    x: {
                        min: 0,
                        max: 100,
                        grid: { color: 'rgba(255, 255, 255, 0.05)' },
                        ticks: { color: '#64748B', font: { family: 'JetBrains Mono', size: 10 } }
                    },
                    y: {
                        ticks: { color: '#E2E8F0', font: { family: 'Inter', weight: 600, size: 11 } }
                    }
                },
                plugins: {
                    legend: { display: false }
                }
            }
        });

        async function fetchStep() {
            const emiType = document.getElementById('emiSelect').value;
            try {
                const response = await fetch(`/api/step?emi_type=${emiType}`);
                if (!response.ok) throw new Error("REST API response error");
                const data = await response.json();

                // Update KPIs
                const detectedElem = document.getElementById('valDetected');
                detectedElem.innerHTML = `<span class="badge-status status-${data.detected_emi}">${data.detected_emi}</span>`;
                document.getElementById('valActual').innerText = `Actual: ${data.true_emi}`;
                
                document.getElementById('valConfidence').innerText = `${data.prediction_confidence}%`;
                
                // Sort confidence to get runner up
                const sortedConf = Object.entries(data.confidence_distribution).sort((a,b) => b[1] - a[1]);
                if (sortedConf.length > 1) {
                    document.getElementById('valTop2').innerText = `Runner-up: ${sortedConf[1][0]} (${sortedConf[1][1]}%)`;
                }

                if (data.metrics.rms_reduction_percent !== null) {
                    document.getElementById('valRms').innerText = `${data.metrics.rms_reduction_percent}%`;
                    document.getElementById('valSnr').innerText = `+${data.metrics.snr_gain_db} dB`;
                    document.getElementById('valSnrSub').innerText = `Before: ${data.metrics.snr_before_db} dB | After: ${data.metrics.snr_after_db} dB`;
                }

                document.getElementById('txtAction').innerText = `Action: ${data.action}`;

                // Update Charts
                rawChart.data.datasets[0].data = data.signal_data.raw;
                rawChart.update();

                cleanChart.data.datasets[0].data = data.signal_data.cleaned;
                cleanChart.update();

                barChart.data.datasets[0].data = [
                    data.confidence_distribution['Clean'],
                    data.confidence_distribution['Powerline'],
                    data.confidence_distribution['Switching'],
                    data.confidence_distribution['Motion'],
                    data.confidence_distribution['Broadband']
                ];
                barChart.update();

            } catch (err) {
                console.error("Failed to fetch step from REST API:", err);
            }
        }

        function startStreaming() {
            if (isRunning) return;
            isRunning = true;
            document.getElementById('btnStart').style.opacity = '0.5';
            document.getElementById('btnStop').classList.add('active');
            
            const interval = parseInt(document.getElementById('speedSelect').value);
            fetchStep(); // immediate first step
            streamTimer = setInterval(fetchStep, interval);
        }

        function stopStreaming() {
            if (!isRunning) return;
            isRunning = false;
            clearInterval(streamTimer);
            streamTimer = null;
            document.getElementById('btnStart').style.opacity = '1.0';
            document.getElementById('btnStop').classList.remove('active');
        }

        function updateInterval() {
            if (isRunning) {
                stopStreaming();
                startStreaming();
            }
        }

        // Run once on load
        window.addEventListener('load', () => {
            fetchStep();
        });
    </script>
</body>
</html>
"""
