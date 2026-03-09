import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set style to match the original script
sns.set_theme(context="talk", style="ticks")

def run_sanity_check():
    print("--- Cross-Correlation Sanity Check ---")
    
    # 1. Generate Synthetic Data
    # We simulate a time series of 1000 trials
    n_trials = 1000
    t = np.arange(n_trials)
    
    # Create a "Signal 1" (e.g., HP Location) that switches state periodically
    # Using a square wave with a period of 200 trials
    # Values: -1 and 1
    sig1_vals = np.sign(np.sin(t * 2 * np.pi / 200))
    
    df = pd.DataFrame({'Trial': t, 'Signal1': sig1_vals})
    
    # 2. Create "Signal 2" (Behavior) as a lagged copy of Signal 1
    # We introduce a known delay (lag).
    # Positive lag means Signal 2 happens AFTER Signal 1.
    TRUE_LAG = 25 
    
    # shift(k) shifts data down by k rows. 
    # value at index i becomes value from index i-k.
    # This simulates a delay.
    df['Signal2'] = df['Signal1'].shift(TRUE_LAG)
    
    # Drop NaNs created by the shift
    df = df.dropna().reset_index(drop=True)
    
    # Add some Gaussian noise to Signal 2 to make it look like real behavioral data
    # (e.g., RT bias or Accuracy bias)
    noise = np.random.normal(0, 0.5, size=len(df))
    df['Signal2'] = df['Signal2'] + noise
    
    # 3. Apply the Cross-Correlation Logic from the original script
    # -----------------------------------------------------------
    lags = np.arange(-100, 100)
    
    # Extract series
    s1 = df['Signal1']
    s2 = df['Signal2']
    
    # Standardize (Z-score) - exactly as in the original script
    sig1_norm = (s1 - s1.mean()) / s1.std()
    sig2_norm = (s2 - s2.mean()) / s2.std()
    
    # Calculate Cross-Correlation
    # Logic: cc = [sig1.corr(sig2.shift(-lag)) for lag in lags]
    cc = [sig1_norm.corr(sig2_norm.shift(-lag)) for lag in lags]
    # -----------------------------------------------------------

    # 4. Visualization
    plt.figure(figsize=(12, 8))
    
    # Plot A: The Signals
    plt.subplot(2, 1, 1)
    plt.plot(df['Trial'], df['Signal1'], label='Signal 1 (Stimulus)', color='black', linestyle='--')
    plt.plot(df['Trial'], df['Signal2'], label='Signal 2 (Behavior/Lagged)', color='tab:blue', alpha=0.7)
    plt.title(f"Synthetic Signals (First 300 trials)\nTrue Delay = {TRUE_LAG} trials")
    plt.ylabel("Value")
    plt.legend(loc='upper right')
    
    # Plot B: The Cross-Correlation Result
    plt.subplot(2, 1, 2)
    plt.plot(lags, cc, color='tab:red', label='Calculated Cross-Corr')
    plt.axvline(TRUE_LAG, color='k', linestyle='--', label=f'True Lag ({TRUE_LAG})')
    plt.axvline(0, color='k', linestyle=':', alpha=0.3)
    plt.title("Cross-Correlation Function")
    plt.xlabel("Lag (Trials)")
    plt.ylabel("Correlation Coefficient")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_sanity_check()