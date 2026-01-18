# Insider Threat Detection System

## 1. Project Overview
This project is a dual-factor authentication system designed to detect insider threats using behavioral biometrics. It combines traditional password authentication with keystroke dynamics analysis to verify user identity.

The system uses an LSTM (Long Short-Term Memory) autoencoder to learn the unique typing patterns of authorized users. By analyzing keystroke timing features (hold times and flight times), the system can distinguish between a genuine user and an imposter, even if they type the correct password.

**Note:** This is an academic demonstration project and is not intended for production security environments.

## 2. System Architecture
The system consists of three main components:
1.  **Data Collection (Enrollment)**: Captures keystroke timing data from the user to create a baseline dataset.
2.  **Model Training**: Trains an LSTM autoencoder on the collected data to learn normal typing behavior. This generates three artifacts:
    *   `model.h5`: The trained neural network.
    *   `scaler.pkl`: The data normalization scaler.
    *   `threshold.npy`: The anomaly detection threshold.
3.  **Inference (Dashboard)**: A real-time interface that verifies the password and analyzes keystroke dynamics against the trained model.

## 3. Prerequisites
*   **Python Version**: Python 3.10 or higher.
*   **Operating System**: Windows, Linux, or macOS (requires a physical keyboard).
*   **Hardware**: A standard desktop or laptop keyboard is required for accurate timing capture.
*   **Internet**: Required only for initial installation. The system runs entirely offline.

## 4. Installation Steps

1.  Clone the repository:
    ```bash
    git clone https://github.com/Vittal-236/Insider_Threat_Detection_System.git
    ```

2.  Navigate to the project directory:
    ```bash
    cd insider-threat-detection
    ```

3.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## 5. How to Run the Project

Follow these steps in order to set up and run the full system.

### Step 1: User Enrollment
Run the enrollment script to collect your biometric data. You will be asked to type the password `test001` repeatedly.
```bash
python enrollment.py
```

### Step 2: Model Training
Train the LSTM model on the data collected in Step 1.
```bash
python train_brain.py
```

### Step 3: Launch Dashboard
Start the real-time detection interface.
```bash
streamlit run dashboard.py
```

## 6. How to Use the Application

1.  **Identity Claim**: In the browser dashboard, enter the password `test001` in the text box.
2.  **Start Biometric Check**: If the password is correct, click the **START SENSOR** button.
3.  **Ensure Focus**:
    *   The system uses `pynput`, which requires OS-level window focus.
    *   **Click anywhere on your desktop wallpaper or taskbar** immediately after clicking the button.
    *   Wait for the "GO" signal.
4.  **Type Password**: Type `test001` and press **Enter**.
5.  **View Results**: The system will display whether you are an "Authorized User" or an "Intruder" based on your typing rhythm.

## 7. Common Issues & Fixes

### "NO INPUT – Sensor Timed Out"
*   **Cause**: The Python script did not receive keyboard events because the browser window held the focus.
*   **Fix**: After clicking "START SENSOR", you **must** click outside the browser (e.g., on your desktop background) to give focus to the operating system.

### False Intruder Detection
*   **Cause**: Natural variations in human typing speed or rhythm.
*   **Fix**: The system includes an adaptive calibration feature. Continue typing correctly for 5-6 attempts. The system will learn your current session's typing style and adjust the threshold automatically.

### Application Crashes during Capture
*   **Cause**: Conflict between Streamlit's event loop and the keyboard listener.
*   **Fix**: Ensure you are running the application locally on a machine with a physical keyboard, not in a cloud environment (like Google Colab or Streamlit Cloud).

## 8. Repository Structure

*   `enrollment.py`: Script for collecting initial user keystroke data.
*   `train_brain.py`: Script for training the LSTM autoencoder and calculating thresholds.
*   `dashboard.py`: Main Streamlit application for real-time testing and visualization.
*   `model.h5`: Saved Keras model file (generated after training).
*   `scaler.pkl`: Saved Scikit-learn scaler for data normalization.
*   `threshold.npy`: Saved anomaly detection threshold value.
*   `user_biometrics.csv`: Raw data file containing enrolled keystroke timings.
*   `requirements.txt`: List of Python dependencies.

## 9. Disclaimer
This software is provided for educational and demonstration purposes only. The accuracy of keystroke dynamics can suffer from environmental factors, hardware latency, and user fatigue. It should not be relied upon as a primary security measure in production environments.
