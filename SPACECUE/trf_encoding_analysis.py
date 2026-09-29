import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

import SPACECUE
from utils import calculate_trajectory_projections
import utils

sns.set_theme(context="talk", style="ticks")
plt.ion()

# --- Script Parameters ---
SUBJECT_IDS = [1]
EPOCH_TMIN, EPOCH_TMAX = -0.1, 0.7  # Time window for the TRF filter
WINDOW = 23  # Rolling window for behavioral spatial bias
TRF_LAGS = np.arange(-50, 50)  # Trials relative to current trial (e.g., -5 means 5 trials in the future)

# --- Helper Functions for Behavioral Data ---


def get_trial_towardness(row):
    locations_map = {
        1: (row['target_x_aligned'], row['target_y_aligned']),
        2: (row['distractor_x_aligned'], row['distractor_y_aligned']),
        3: (row['control_x_aligned'], row['control_y_aligned'])
    }
    row_copy = row.copy()
    row_copy['TargetDigit'] = 1
    row_copy['SingletonDigit'] = 2
    row_copy['SingletonPresent'] = 1
    row_copy['Non-Singleton2Digit'] = 3

    scores = calculate_trajectory_projections(row_copy, locations_map=locations_map)
    scores['distractor_towardness'] = scores['proj_distractor']
    scores["target_towardness"] = scores["proj_target"]
    return scores

def load_and_preprocess_behavior(subject_ids):
    print(f"Loading behavioral data for subjects: {subject_ids}...")
    data_path = SPACECUE.get_data_path()
    experiment_folder = "/derivatives/preprocessing"
    
    # 1. Load basic behavioral CSVs
    df_list = []
    for sub_id in subject_ids:
        beh_files = glob.glob(f"{data_path}{experiment_folder}/sci-{sub_id}/beh/*.csv")
        for file in beh_files:
            df_list.append(pd.read_csv(file))

    df = pd.concat(df_list, ignore_index=True)
    df['subject_id'] = pd.to_numeric(df['Subject ID'], errors='coerce').astype('Int64')
    df['block'] = pd.to_numeric(df['Block'], errors='coerce').astype('Int64')
    df['trial_nr'] = pd.to_numeric(df['Trial Nr'], errors='coerce').astype('Int64')
    # Safely replace regardless of whether they are parsed as ints or strings
    df['HP_Distractor_Loc'] = df['HP_Distractor_Loc'].replace({1: 'Left', '1': 'Left', 2: 'Front', '2': 'Front', 3: 'Right', '3': 'Right'})

    # 2. Load Mouse Trajectories
    df_mouse = utils.load_raw_mouse_trajectories(subject_ids, data_path, calculate_phase=False)

    # 3. Align and Calculate Towardness
    df_mouse_aligned = utils.align_raw_mouse_trajectories(df_mouse, df)

    agg_dict = {
        'avg_x_dva': ('x_aligned', 'mean'), 'avg_y_dva': ('y_aligned', 'mean'),
        'target_x_aligned': ('target_x_aligned', 'first'), 'target_y_aligned': ('target_y_aligned', 'first'),
        'distractor_x_aligned': ('distractor_x_aligned', 'first'), 'distractor_y_aligned': ('distractor_y_aligned', 'first'),
        'control_x_aligned': ('control_x_aligned', 'first'), 'control_y_aligned': ('control_y_aligned', 'first'),
    }
    avg_traj_vectors = df_mouse_aligned.groupby(['subject_id', 'block', 'trial_nr']).agg(**agg_dict).reset_index()
    df = pd.merge(df, avg_traj_vectors, on=['subject_id', 'block', 'trial_nr'], how='left')
    
    towardness_scores = df.apply(get_trial_towardness, axis=1)
    tt_df = pd.concat([df, towardness_scores], axis=1)
    loc_col = "Non-Singleton2Loc" if "control" in experiment_folder else "SingletonLoc"
    return tt_df, loc_col

def compute_encoding_predictors(tt_df, loc_col):
    print("Computing continuous predictor features...")
    beh_features = []
    
    # Helper to safely map physical locations to standard regression weights
    def map_loc(val):
        v = str(val).lower().strip()
        if v in ['left', '1', '1.0']: return -1
        if v in ['right', '3', '3.0']: return 1
        return 0
        
    for sub_id, sub_df in tt_df.groupby('subject_id'):
        sub_df = sub_df.sort_values(['block', 'trial_nr']).copy()
        
        # 1. Environment Predictor: -1 (Left) and +1 (Right)
        sub_df['HP_Env'] = sub_df['HP_Distractor_Loc'].map({'Left': -1, 'Right': 1, 'Front': 0}).fillna(0)
        
        # 2. Behavioral Bias Predictor (Target Towardness)
        raw_bias = sub_df['target_towardness'].rolling(window=WINDOW, center=True, min_periods=1).mean()
        
        # Standardize (Z-score) behavioral bias so beta weights are in comparable scales (µV per 1 Std Dev)
        scaler = StandardScaler()
        sub_df['Behavior_Bias'] = np.nan
        
        valid_bias = ~raw_bias.isna()
        if valid_bias.sum() > 0: # Ensures StandardScaler does not crash on NaNs
            sub_df.loc[valid_bias, 'Behavior_Bias'] = scaler.fit_transform(raw_bias[valid_bias].values.reshape(-1, 1)).flatten()
        
        # 3. Physical Stimulus Predictors (Crucial for absorbing massive physical ERPs)
        sub_df['DistLoc_Mapped'] = sub_df[loc_col].replace({1: 'Left', '1': 'Left', 2: 'Front', '2': 'Front', 3: 'Right', '3': 'Right'})
        sub_df['Dist_Loc_Current'] = sub_df['DistLoc_Mapped'].apply(map_loc)
        
        # 4. Switch Predictors
        sub_df['HP_Env_prev'] = sub_df.groupby('block')['HP_Env'].shift(1)
        sub_df['HP_Prob'] = pd.to_numeric(sub_df.get('HP_Distractor_Prob', np.nan), errors='coerce')
        sub_df['HP_Prob_prev'] = sub_df.groupby('block')['HP_Prob'].shift(1)
        
        # Absolute Side Switch (1 = Side Changed, 0 = No Change)
        sub_df['side_switch'] = 0.0
        sub_df.loc[(sub_df['HP_Env_prev'] != sub_df['HP_Env']) & (sub_df['HP_Env_prev'] != 0) & (sub_df['HP_Env'] != 0) & sub_df['HP_Env_prev'].notna(), 'side_switch'] = 1.0
        
        # Absolute Probability Switch (1 = Prob Changed on same side, 0 = No Change)
        prob_diff = sub_df['HP_Prob'] - sub_df['HP_Prob_prev']
        is_pure_prob_switch = (sub_df['HP_Env'] == sub_df['HP_Env_prev']) & (prob_diff != 0) & prob_diff.notna()
        sub_df['prob_switch'] = 0.0
        sub_df.loc[is_pure_prob_switch, 'prob_switch'] = 1.0
        
        # 5. Single-Trial bias (optimal for TRF estimation to avoid double-filtering)
        valid_single = ~sub_df['target_towardness'].isna()
        sub_df['Behavior_Bias_Single'] = np.nan
        if valid_single.sum() > 0:
            sub_df.loc[valid_single, 'Behavior_Bias_Single'] = scaler.fit_transform(sub_df.loc[valid_single, 'target_towardness'].values.reshape(-1, 1)).flatten()

        beh_features.append(sub_df[['subject_id', 'block', 'trial_nr', 'HP_Env', 'HP_Prob', 'Behavior_Bias', 
                                    'Behavior_Bias_Single', 'side_switch', 'prob_switch', 'Dist_Loc_Current']])
        
    return pd.concat(beh_features)

# --- 1. Load and Prepare Predictors (X) ---
tt_df, loc_col = load_and_preprocess_behavior(SUBJECT_IDS)
beh_df = compute_encoding_predictors(tt_df, loc_col)

# --- Visualize Regressors and Behavior (Sanity Check Plot) ---
print("Plotting Regressors and Behavioral Metric over trials...")
for sub_id in [SUBJECT_IDS[0]]:
    sub_df = beh_df[beh_df['subject_id'] == sub_id].sort_values(['block', 'trial_nr']).reset_index(drop=True)
    
    fig, ax_full = plt.subplots(figsize=(18, 7))
    
    # Plot Smoothed Behavior (using rolling average just for visual trend clarity)
    smoothed_beh = sub_df['Behavior_Bias_Single'].rolling(WINDOW, center=True, min_periods=1).mean()
    ax_full.plot(sub_df.index, smoothed_beh, color='green', label=f'Target Towardness (Smoothed)', lw=2)
    ax_full.axhline(0, color='gray', linestyle='--')
    ax_full.set_ylabel("Target Towardness (Z-scored)")
    ax_full.set_xlabel("Trial Index")
    ax_full.set_title(f"Trial Series & Regressors for Subject {sub_id}")
    
    # Secondary axis for the discrete rule and regressors
    ax_twin = ax_full.twinx()
    
    # Plot full HP rule
    full_hp_signal = sub_df['HP_Env'] * sub_df['HP_Prob']
    ax_twin.step(sub_df.index, full_hp_signal, where='mid', color='black', linestyle='--', label='Full HP Rule (Env * Prob)', alpha=0.5)
    
    # Plot side switches (using stem lines for clear isolated impulses)
    side_switches = sub_df[sub_df['side_switch'] != 0]
    ax_twin.stem(side_switches.index, side_switches['side_switch'], linefmt='b-', markerfmt='bo', basefmt=' ', label='Side Switch Impulse')
    
    # Plot prob switches
    prob_switches = sub_df[sub_df['prob_switch'] != 0]
    ax_twin.stem(prob_switches.index, prob_switches['prob_switch'], linefmt='r-', markerfmt='ro', basefmt=' ', label='Prob Switch Impulse')
    
    # Highlight probability blocks
    sub_df['prob_block_id'] = (sub_df['HP_Prob'].diff() != 0).cumsum()
    added_labels = set()
    for _, block_df in sub_df.groupby('prob_block_id'):
        prob_val = block_df['HP_Prob'].iloc[0]
        if pd.notna(prob_val):
            color = 'lightblue' if prob_val >= 0.8 else 'lightcoral'
            label = f'Prob {prob_val} Blocks' if prob_val not in added_labels else None
            ax_full.axvspan(block_df.index.min(), block_df.index.max(), facecolor=color, alpha=0.3, zorder=-1)
            if label:
                ax_full.fill_between([], [], color=color, alpha=0.3, label=label) # Dummy for legend
                added_labels.add(prob_val)
                
    ax_twin.set_ylabel("Regressor Impulses & Rule Signal")
    ax_twin.set_ylim(-1.5, 1.5)
    
    # Combine legends
    lines, labels = ax_full.get_legend_handles_labels()
    lines2, labels2 = ax_twin.get_legend_handles_labels()
    ax_full.legend(lines + lines2, labels + labels2, loc='upper left')
    
    plt.tight_layout()
    plt.show()

# --- 2. Fit Behavioral TRF (Trial-wise FIR Model) ---
print("Fitting Behavioral TRF (FIR Model)...")

TARGET_VAR = 'Behavior_Bias_Single' 
PREDICTORS = ['side_switch', 'prob_switch', 'Dist_Loc_Current']

trf_results = []

for sub_id in SUBJECT_IDS:
    sub_df = beh_df[beh_df['subject_id'] == sub_id].sort_values(['block', 'trial_nr']).reset_index(drop=True)
    
    X_lagged = []
    feature_names = []
    y = sub_df[TARGET_VAR].values
    
    for pred in PREDICTORS:
        for lag in TRF_LAGS:
            # Shift > 0: current trial is predicted by a past event
            # Shift < 0: current trial is predicted by a future event (sanity check baseline)
            shifted = sub_df.groupby('block')[pred].shift(lag).values
            X_lagged.append(shifted)
            feature_names.append((pred, lag))
            
    X_lagged = np.column_stack(X_lagged)
    
    # Drop rows with NaNs caused by boundary shifts
    valid_idx = ~np.isnan(X_lagged).any(axis=1) & ~np.isnan(y)
    X_valid = X_lagged[valid_idx]
    y_valid = y[valid_idx]
    
    if len(X_valid) > 50:
        # Ridge regularizes multi-collinear lagged regressors
        model = Ridge(alpha=10.0) 
        model.fit(X_valid, y_valid)
        
        for i, (pred, lag) in enumerate(feature_names):
            trf_results.append({
                'subject_id': sub_id,
                'predictor': pred,
                'lag': lag,
                'beta': model.coef_[i]
            })

trf_df = pd.DataFrame(trf_results)

# --- 3. Plot Behavioral TRF Results ---
print("Plotting TRF Results...")
fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=True, sharey=False)
axes = axes.flatten()

for i, pred in enumerate(PREDICTORS):
    ax = axes[i]
    sns.lineplot(data=trf_df[trf_df['predictor'] == pred], x='lag', y='beta', ax=ax, errorbar='se', lw=2.5, color='darkblue')
    
    ax.axhline(0, color='k', linestyle='--', alpha=0.7)
    ax.axvline(0, color='k', linestyle='-', alpha=0.4)
    ax.axvspan(min(TRF_LAGS), 0, color='gray', alpha=0.1)  # Pre-event zone
    
    ax.set_title(f"TRF: {pred}")
    ax.set_xlabel("Lag (Trials from Event)")
    ax.set_ylabel("Beta (Impact on Target Towardness)")

plt.suptitle("Behavioral TRF: Effect of Rule Switches on Target Towardness", fontsize=16)
plt.tight_layout()
plt.show()
