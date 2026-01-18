import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import time
from pynput import keyboard
import os

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Cortex: Insider Threat Detection",
    page_icon="🛡️",
    layout="wide"
)

# --- CONSTANTS ---
TARGET_PASSWORD = "test001"
EXPECTED_FEATURE_COUNT = 22 

# Adaptive Logic Constants
WARMUP_ATTEMPTS = 5
HISTORY_SIZE = 20

MODEL_FILE = "model.h5"
SCALER_FILE = "scaler.pkl"
THRESHOLD_FILE = "threshold.npy"

# --- STYLE CSS ---
st.markdown("""
    <style>
    .verified { background-color: #d4edda; color: #155724; padding: 20px; border-radius: 10px; text-align: center; }
    .intruder { background-color: #f8d7da; color: #721c24; padding: 20px; border-radius: 10px; text-align: center; }
    .stButton>button { width: 100%; height: 50px; font-size: 20px; }
    </style>
""", unsafe_allow_html=True)

# --- LOAD RESOURCES ---
@st.cache_resource
def load_resources():
    try:
        model = tf.keras.models.load_model(MODEL_FILE, compile=False)
        scaler = joblib.load(SCALER_FILE)
        threshold = np.load(THRESHOLD_FILE)
        return model, scaler, float(threshold)
    except FileNotFoundError:
        return None, None, None

model, scaler, static_threshold = load_resources()

# --- INITIALIZE SESSION STATE FOR ADAPTIVE BIOMETRICS ---
if 'mse_history' not in st.session_state:
    st.session_state['mse_history'] = []

# --- HELPER FUNCTIONS ---
def extract_features(press_times, release_times):
    """
    Extracts strictly 22 features from raw timestamps.
    """
    # We expect 8 keys (7 chars + 1 enter)
    if len(press_times) != 8 or len(release_times) != 8:
        return None

    features = []
    
    # 1. Hold Times (H)
    for p, r in zip(press_times, release_times):
        features.append(r - p)
        
    # 2. Down-Down Latencies (DD)
    for i in range(len(press_times) - 1):
        features.append(press_times[i+1] - press_times[i])
        
    # 3. Up-Down Latencies (UD)
    for i in range(len(release_times) - 1):
        features.append(press_times[i+1] - release_times[i])
        
    if len(features) != EXPECTED_FEATURE_COUNT:
        return None
        
    return features

def capture_timings_only():
    """
    Captures raw timestamps ONLY via pynput global hook.
    """
    press_times = []
    release_times = []
    
    state = {'running': True}
    
    def on_press(key):
        if not state['running']: return False
        t = time.time()
        press_times.append(t)

    def on_release(key):
        if not state['running']: return False
        t = time.time()
        
        if key == keyboard.Key.enter:
            state['running'] = False
            release_times.append(t) 
            return False
            
        release_times.append(t)

    # Start Listener
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()
    
    # Wait Loop
    start_time = time.time()
    while listener.is_alive() and state['running']:
        if time.time() - start_time > 10.0:
            listener.stop()
            state['running'] = False
            break
        time.sleep(0.01)
        
    try:
        listener.stop()
    except:
        pass
        
    return press_times, release_times

# --- MAIN PAGE STRUCTURE ---

st.title("🛡️ Cortex: Insider Threat Detection")
st.markdown("### Dual-Factor Authentication System (Adaptive)")
st.markdown("---")

col1, col2 = st.columns([1, 2])

# --- Col 1: Authentication Flow ---
with col1:
    st.subheader("Step 1: Identity Claim")
    
    # Deterministic Streamlit Input
    password_input = st.text_input("Enter Password", type="password", placeholder="Type test001 here...", key="pwd_input")
    
    if password_input:
        if password_input == TARGET_PASSWORD:
            st.success("✅ Password Correct")
            st.markdown("---")
            st.subheader("Step 2: Biometric Verification")
            st.info("Keystroke Dynamics Analysis")
            
            # Biometric Trigger
            if st.button("🔴 START SENSOR", use_container_width=True):
                if model is None:
                    st.error("System Error: Model missing.")
                else:
                    # --- FOCUS CAPTURE SEQUENCE (MANDATORY FIX) ---
                    status_box = st.empty()
                    
                    # 1. Guide user to fix focus
                    status_box.info("🖱️ Click your DESKTOP wallpaper to ensure capture...")
                    time.sleep(1.5)
                    
                    # 2. Countdown
                    status_box.warning("⏳ Sensor starting in 3...")
                    time.sleep(1)
                    status_box.warning("⏳ Sensor starting in 2...")
                    time.sleep(1)
                    status_box.warning("⏳ Sensor starting in 1...")
                    time.sleep(1)
                    
                    # 3. GO Signal
                    status_box.error(f"🔴 TYPE '{TARGET_PASSWORD}' + ENTER NOW!")
                    
                    # 4. Capture
                    p_times, r_times = capture_timings_only()
                    status_box.empty()
                    
                    # Store results
                    st.session_state['biometric_result'] = {
                        'p_times': p_times,
                        'r_times': r_times
                    }
        else:
            st.error("❌ Incorrect Password")
            if 'biometric_result' in st.session_state:
                del st.session_state['biometric_result']
            # Reset history on wrong password? 
            # Ideally session history should persist for the Valid User session, but simpler to keep it.

# --- Col 2: Analysis Results ---
with col2:
    st.markdown("#### Analysis Results")
    banner_holder = st.empty()
    chart_holder = st.empty()
    debug_expander = st.expander("Adaptive Calibration Details")
    
    if 'biometric_result' in st.session_state:
        # Process the captured biometric data
        data = st.session_state['biometric_result']
        p_t = data['p_times']
        r_t = data['r_times']
        
        # 1. Check for Focus/Input Failure
        if len(p_t) == 0:
             banner_holder.warning("## ⚠️ FOCUS ERROR\n**Keyboard not captured**")
             st.error("The system did not detect any keystrokes.\n\n**Solution:**\n1. Click 'START SENSOR'\n2. Click outside the browser (on your desktop taskbar/wallpaper)\n3. Type the password immediately.")
        
        # 2. Validation (Length Check)
        elif len(p_t) != (len(TARGET_PASSWORD) + 1):
             banner_holder.error(f"## 🚨 BIOMETRIC MISMATCH\n**Key Count Mismatch**")
             st.caption(f"Expected {len(TARGET_PASSWORD)+1} keys, captured {len(p_t)}. Ensure you type exactly '{TARGET_PASSWORD}' + Enter.")
        
        # 3. Feature Extraction & Inference
        else:
            features = extract_features(p_t, r_t)
            
            if features:
                feats_arr = np.array([features])
                feats_scaled = scaler.transform(feats_arr)
                feats_3d = feats_scaled.reshape(1, 1, EXPECTED_FEATURE_COUNT)
                
                reconst = model.predict(feats_3d)
                mse = np.mean(np.power(feats_3d - reconst, 2))
                
                # --- ADAPTIVE THRESHOLD LOGIC ---
                history = st.session_state['mse_history']
                is_authorized = False
                current_threshold = static_threshold
                decision_reason = "Static Threshold"
                
                # A. Warm-Up Phase
                if len(history) < WARMUP_ATTEMPTS:
                    is_authorized = True
                    decision_reason = f"Warm-Up Phase ({len(history)+1}/{WARMUP_ATTEMPTS})"
                    # Always append during warmup (assuming user is trying properly)
                    st.session_state['mse_history'].append(mse)
                    
                # B. Adaptive Phase
                else:
                    # Rolling Statistics
                    mu = np.mean(history)
                    sigma = np.std(history)
                    adaptive_threshold = mu + (2.0 * sigma)
                    
                    # Logic: Accept if below static OR below adaptive
                    # But if very high (Intruder), assume it breaks both.
                    
                    # The prompt says: "Comparison to adaptive threshold handles drift"
                    # We employ the Two-Stage Logic
                    
                    if mse <= static_threshold:
                        is_authorized = True
                        decision_reason = "Static Verification"
                        current_threshold = static_threshold
                    elif mse <= adaptive_threshold:
                        is_authorized = True
                        decision_reason = f"Adaptive Verification (Drift) - Thresh: {adaptive_threshold:.4f}"
                        current_threshold = adaptive_threshold
                    else:
                        is_authorized = False
                        decision_reason = "Biometric Rejection"
                        # For chart purposes, show the one they failed closest to? Or just static.
                        current_threshold = max(static_threshold, adaptive_threshold)
                    
                    # Update history only if authorized
                    if is_authorized:
                        st.session_state['mse_history'].append(mse)
                        # Keep window size fixed
                        if len(st.session_state['mse_history']) > HISTORY_SIZE:
                            st.session_state['mse_history'].pop(0)

                # --- DISPLAY ---
                if is_authorized:
                    # Provide visual indication of mode
                    if "Warm-Up" in decision_reason:
                        banner_holder.info(f"## 🛡️ CALIBRATING...\n**{decision_reason}**\nSample Accepted (Score: {mse:.4f})")
                    else:
                        conf = max(0, (1 - (mse / current_threshold)) * 100)
                        banner_holder.success(f"## ✅ AUTHORIZED USER\n**{decision_reason}**\nConfidence: {conf:.1f}%")
                else:
                    banner_holder.error(f"## 🚨 INTRUDER DETECTED\n**Access Denied**\nScore: {mse:.4f} > Limit: {current_threshold:.4f}")
                
                # --- CHART ---
                df_chart = pd.DataFrame({
                    "Metric": ["Your Score", "Active Threshold"],
                    "Value": [mse, current_threshold]
                })
                chart_color = "#2ecc71" if is_authorized else "#ff4b4b"
                chart_holder.bar_chart(df_chart, x="Metric", y="Value", color=chart_color)
                
                with debug_expander:
                    st.write(f"**Current MSE**: {mse:.6f}")
                    st.write(f"**History Size**: {len(history)}")
                    if len(history) > 0:
                        st.write(f"**Baseline Mean**: {np.mean(history):.6f}")
                        st.write(f"**Baseline Std**: {np.std(history):.6f}")
                        if len(history) >= WARMUP_ATTEMPTS:
                            st.write(f"**Adaptive Limit**: {np.mean(history) + 2*np.std(history):.6f}")
            else:
                banner_holder.error("## ⚠️ ERROR\n**Feature Extraction Failed**")
    else:
        banner_holder.info("Waiting for Step 2...")
