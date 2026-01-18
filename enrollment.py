"""
enrollment.py - The Data Collector

Purpose:
    Captures the user's keystroke dynamics to create a baseline dataset for the Insider Threat Detection System.
    It prompts the user to type a specific password multiple times and extracts temporal features.

Password: .tie5Roanl
Total Key Presses required per attempt: 11 (10 characters + Enter)
Features extracted (31 total):
    - 11 Hold Times (H): Time between Press and Release for each key.
    - 10 Down-Down Latencies (DD): Time between Press[i] and Press[i+1].
    - 10 Up-Down Latencies (UD): Time between Release[i] and Press[i+1].

Output:
    Saves validation attempts to 'user_biometrics.csv'.
"""

import pynput
from pynput.keyboard import Key, Listener
import time
import pandas as pd
import numpy as np
import os
import sys

# Configuration
PASSWORD_TEXT = "test001"
REQUIRED_REPETITIONS = 20
OUTPUT_FILE = "user_biometrics.csv"

# Global storage for current attempt
press_times = []
release_times = []
key_sequence = []

def on_press(key):
    """Callback for key press events."""
    try:
        # Record timestamp immediately
        t = time.time()
        press_times.append(t)
        
        # Convert key to string for validation
        if hasattr(key, 'char'):
            k = key.char
        else:
            k = str(key) # Handle special keys like Key.enter
        
        key_sequence.append(k)
        
        # specific check for enter to stop listening if we have enough keys or just wait
        if key == Key.enter:
            return False # Stop listener
            
    except Exception as e:
        print(f"Error on press: {e}")

def on_release(key):
    """Callback for key release events."""
    t = time.time()
    release_times.append(t)

def extract_features(press_times, release_times):
    """
    Calculates the 22 temporal features from timestamps.
    (7 chars + Enter = 8 keys)
    
    Args:
        press_times (list): List of press timestamps.
        release_times (list): List of release timestamps.
        
    Returns:
        list: 22 floating point features or None if invalid.
    """
    # We expect 8 keys (7 chars + Enter)
    if len(press_times) != 8 or len(release_times) != 8:
        return None

    features = []

    # 1. Hold Times (H) - 8 features
    # H[i] = Release[i] - Press[i]
    for p, r in zip(press_times, release_times):
        features.append(r - p)

    # 2. Down-Down Latencies (DD) - 7 features
    # DD[i] = Press[i+1] - Press[i]
    for i in range(len(press_times) - 1):
        features.append(press_times[i+1] - press_times[i])

    # 3. Up-Down Latencies (UD) - 7 features
    # UD[i] = Press[i+1] - Release[i]
    for i in range(len(release_times) - 1):
        features.append(press_times[i+1] - release_times[i])

    return features

def main():
    print("==============================================")
    print("   INSIDER THREAT DETECTION - ENROLLMENT      ")
    print("==============================================")
    print(f"Target Password: {PASSWORD_TEXT}")
    print("Please type the password and press ENTER.")
    print(f"We need {REQUIRED_REPETITIONS} valid samples.\n")

    valid_samples = []
    attempts = 0

    while len(valid_samples) < REQUIRED_REPETITIONS:
        # Reset buffers
        global press_times, release_times, key_sequence
        press_times = []
        release_times = []
        key_sequence = []
        
        print(f"Attempt {len(valid_samples) + 1}/{REQUIRED_REPETITIONS}: ", end="", flush=True)

        # Start listener (blocking until Enter is pressed)
        with Listener(on_press=on_press, on_release=on_release) as listener:
            listener.join()
            
        # Validate Input
        # Clean key sequence to form string (remove Key.enter)
        
        # Convert captured sequence to string for comparison
        chars = []
        for k in key_sequence:
            if k == 'Key.enter':
                continue
            if isinstance(k, str):
                if len(k) == 1:
                    chars.append(k)
        
        typed_str = "".join(chars)

        if typed_str == PASSWORD_TEXT and len(press_times) == 8:
            feats = extract_features(press_times, release_times)
            if feats and len(feats) == 22:
                valid_samples.append(feats)
                print(" [ACCEPTED]")
            else:
                print(" [ERROR: Calculation fail]")
        else:
            print(f" [REJECTED] - You typed: '{typed_str}'")
            if len(press_times) != 8:
                print(f"    (Expected 8 keystrokes, got {len(press_times)})")

    # Save to CSV
    columns = [f"H_{i}" for i in range(8)] + \
              [f"DD_{i}" for i in range(7)] + \
              [f"UD_{i}" for i in range(7)]
    
    df = pd.DataFrame(valid_samples, columns=columns)
    
    if os.path.exists(OUTPUT_FILE):
        df.to_csv(OUTPUT_FILE, index=False)
    else:
        df.to_csv(OUTPUT_FILE, index=False)
        
    print(f"\nEncoding complete. Data saved to {OUTPUT_FILE}.")

if __name__ == "__main__":
    main()
