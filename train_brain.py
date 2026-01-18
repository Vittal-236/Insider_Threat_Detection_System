"""
train_brain.py - The Model Builder

Purpose:
    Trains an LSTM Autoencoder unsupervised model to learn the user's typing pattern.
    It reads 'user_biometrics.csv', creates a MinMax scaler, trains the model, and determines
    the anomaly threshold based on reconstruction error (MSE).

Architecture:
    Input (1, 22) -> LSTM(64) -> LSTM(32) -> Latent Space -> RepeatVector
    -> LSTM(32) -> LSTM(64) -> Output (1, 22)

Outputs:
    - model.h5: Trained Keras model.
    - scaler.pkl: Fitted MinMaxScaler.
    - threshold.npy: Float value defining the boundary for "normal" behavior.
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, RepeatVector, TimeDistributed

# Configuration
INPUT_FILE = "user_biometrics.csv"
MODEL_FILE = "model.h5"
SCALER_FILE = "scaler.pkl"
THRESHOLD_FILE = "threshold.npy"
EPOCHS = 120
BATCH_SIZE = 4

def build_model(input_shape):
    """
    Constructs the LSTM Autoencoder.
    """
    model = Sequential([
        # Encoder
        LSTM(64, activation='relu', input_shape=input_shape, return_sequences=True),
        LSTM(32, activation='relu', return_sequences=False),
        
        # Latent Space / Bottle Neck representation
        RepeatVector(input_shape[0]), # Repeats the context vector to match time steps
        
        # Decoder
        LSTM(32, activation='relu', return_sequences=True),
        LSTM(64, activation='relu', return_sequences=True),
        
        # Output
        TimeDistributed(Dense(input_shape[1]))
    ])
    
    model.compile(optimizer='adam', loss='mse')
    return model

def main():
    # 1. Check for data
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found. Please run enrollment.py first.")
        sys.exit(1)
        
    print(f"Loading data from {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)
    
    if df.empty:
        print("Error: Dataset is empty.")
        sys.exit(1)

    data = df.values
    print(f"Data shape: {data.shape}")

    # 2. Normalize Data
    print("Normalizing data...")
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)
    
    # Save Scaler for use in dashboard
    joblib.dump(scaler, SCALER_FILE)
    print(f"Scaler saved to {SCALER_FILE}")

    # 3. Reshape for LSTM: [samples, time_steps, features]
    # We treat the entire sequence of 31 features as 1 time step with 31 dimensions
    # for this specific autoencoder architecture requested (based on Dense(31)).
    # However, standard LSTM usage usually takes sequence length.
    # The prompt explicitly asks for: Reshape data to (samples, 1, 31).
    X_train = data_scaled.reshape((data_scaled.shape[0], 1, data_scaled.shape[1]))
    print(f"Training shape: {X_train.shape}")

    # 4. Build Model
    model = build_model((X_train.shape[1], X_train.shape[2]))
    model.summary()

    # 5. Train
    print(f"Training for {EPOCHS} epochs...")
    history = model.fit(
        X_train, X_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.1,
        verbose=1,
        shuffle=True
    )

    # 6. Calculate Threshold
    print("Calculating threshold...")
    # Predict on training data to see reconstruction error
    X_pred = model.predict(X_train)
    
    # Calculate Mean Squared Error (MSE) per sample
    # flatten to (samples, 31) for calculation
    # or keep 3D. let's just use np.mean over axis 1 and 2
    mse = np.mean(np.power(X_train - X_pred, 2), axis=(1, 2))
    
    max_mse = np.max(mse)
    # OPTION B: Robust Safety Margin (3.0x)
    # Human typing has high variance. 1.5x is too strict for a demo.
    threshold = max_mse * 3.0 
    
    print(f"Max MSE on training data: {max_mse:.6f}")
    print(f"Setting Anomaly Threshold: {threshold:.6f} (Multiplier: 3.0)")
    
    # 7. Save Artifacts
    model.save(MODEL_FILE)
    np.save(THRESHOLD_FILE, threshold)
    print("Model and threshold saved successfully.")

if __name__ == "__main__":
    main()
