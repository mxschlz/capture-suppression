import pandas as pd
import numpy as np
import os
import SPACECUE
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(context="talk", style="ticks")

def run_shuffled_check():
    # --- Configuration ---
    experiment_folder = "pilot/distractor-switch"
    WINDOW = 45

    # --- Data Loading ---
    print("Loading data for shuffled check...")
    data_path = SPACECUE.get_data_path()
    full_path = os.path.join(data_path, experiment_folder)

    files = [f for f in os.listdir(full_path) if f.endswith('.csv')]
    if not files:
        print("No files found.")
        return

    df = pd.concat([pd.read_csv(os.path.join(full_path, f)) for f in files], ignore_index=True)
    
    # Basic Filtering
    df = df[df["TargetLoc"] != "Front"]
    
    # Ensure subject_id
    if 'Subject ID' in df.columns:
        df['subject_id'] = df['Subject ID'].astype(int, errors="ignore")
    elif 'subject_id' in df.columns:
        df['subject_id'] = df['subject_id'].astype(int, errors="ignore")

    # Map Locations
    if 'SingletonLoc' in df.columns and pd.api.types.is_numeric_dtype(df['SingletonLoc']):
        df['SingletonLoc'] = df['SingletonLoc'].map({0: 'Absent', 1: 'Left', 2: 'Front', 3: 'Right'})
    
    if 'HP_Distractor_Loc' in df.columns:
         df['HP_Distractor_Loc'] = df['HP_Distractor_Loc'].replace({1: 'Left', 2: 'Front', 3: 'Right', 0: 'Absent'})

    # Filter for relevant HP locations (Left/Right)
    df = df[df['HP_Distractor_Loc'].isin(['Left', 'Right'])].reset_index(drop=True)

    # --- Calculate Rolling Metrics (RT Index) ---
    # We need to calculate this per subject before shuffling
    def calculate_metrics(sub_df):
        # RT Bias
        sub_df['RT_Left_Dist'] = sub_df['IsCorrect'].where(sub_df['SingletonLoc'] == 'Left')
        sub_df['RT_Right_Dist'] = sub_df['IsCorrect'].where(sub_df['SingletonLoc'] == 'Right')

        sub_df['RT_Left_Roll'] = sub_df['RT_Left_Dist'].rolling(window=WINDOW, min_periods=1, center=True).mean()
        sub_df['RT_Right_Roll'] = sub_df['RT_Right_Dist'].rolling(window=WINDOW, min_periods=1, center=True).mean()

        sub_df['RT_Index'] = (sub_df['RT_Left_Roll'] - sub_df['RT_Right_Roll']) / (sub_df['RT_Left_Roll'] + sub_df['RT_Right_Roll'])
        return sub_df

    df = df.groupby('subject_id', group_keys=False).apply(calculate_metrics)

    # --- Shuffled Cross-Correlation ---
    print("Running shuffled cross-correlation...")
    subjects = df['subject_id'].unique()
    n_subs = len(subjects)
    
    if n_subs < 2:
        print("Error: Need at least 2 subjects to perform shuffled cross-correlation.")
        return

    lags = np.arange(-100, 100)
    shuffled_cc_list = []

    plt.figure(figsize=(12, 8))

    # Iterate through subjects and pair them with a RANDOM other subject
    for i in range(n_subs):
        sub_stim_id = subjects[i]
        # Randomly select a different subject
        sub_beh_id = np.random.choice([s for s in subjects if s != sub_stim_id])

        # Extract Data
        # Reset index to ensure we align by "Trial Number" (0 to N)
        df_stim = df[df['subject_id'] == sub_stim_id].reset_index(drop=True)
        df_beh = df[df['subject_id'] == sub_beh_id].reset_index(drop=True)

        # Determine overlap length
        min_len = min(len(df_stim), len(df_beh))
        
        # Signal 1: HP Location from Subject A
        # Map Left=-1, Right=1
        s1_raw = df_stim['HP_Distractor_Loc'].iloc[:min_len].map({'Left': -1, 'Right': 1})
        
        # Signal 2: RT Index from Subject B
        s2_raw = df_beh['RT_Index'].iloc[:min_len]

        # Create a temp DF to handle NaNs (from rolling window) and alignment
        temp = pd.DataFrame({'S1': s1_raw, 'S2': s2_raw}).dropna()

        if len(temp) < 100:
            print(f"Skipping pair {sub_stim_id}-{sub_beh_id}: Not enough overlapping data.")
            continue

        s1 = temp['S1']
        s2 = temp['S2']

        # Normalize (Z-score)
        if s1.std() == 0 or s2.std() == 0:
            continue
            
        s1 = (s1 - s1.mean()) / s1.std()
        s2 = (s2 - s2.mean()) / s2.std()

        # Calculate Cross-Corr
        cc = [s1.corr(s2.shift(-lag)) for lag in lags]
        shuffled_cc_list.append(cc)

        # Plot individual trace
        plt.plot(lags, cc, color='gray', alpha=0.2)

    # Plot Mean
    if shuffled_cc_list:
        mean_cc = np.mean(shuffled_cc_list, axis=0)
        plt.plot(lags, mean_cc, color='tab:red', linewidth=3, label='Mean Shuffled Correlation')
        
        # Add confidence interval (SEM)
        sem_cc = np.std(shuffled_cc_list, axis=0) / np.sqrt(len(shuffled_cc_list))
        plt.fill_between(lags, mean_cc - sem_cc, mean_cc + sem_cc, color='tab:red', alpha=0.2)

    plt.axhline(0, color='k', linestyle='--', label="Zero Correlation")
    plt.axvline(0, color='k', linestyle=':', alpha=0.5)
    
    plt.title("Sanity Check: Shuffled Cross-Correlation\n(Stimulus of Subject N vs. Behavior of Random Subject M)")
    plt.xlabel("Lag (Trials)")
    plt.ylabel("Correlation Coefficient")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_shuffled_check()
