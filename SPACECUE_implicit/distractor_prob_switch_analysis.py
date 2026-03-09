import pandas as pd
import numpy as np
import os
import SPACECUE_implicit
import matplotlib.pyplot as plt
import seaborn as sns
from stats import remove_outliers
from scipy.stats import ttest_1samp

sns.set_theme(context="talk", style="ticks")

# --- Configuration ---
OUTLIER_THRESH = 2
experiment_folder = "pilot/distractor-switch"

# --- Data Loading ---
print("Loading data...")
data_path = SPACECUE_implicit.get_data_path()
full_path = os.path.join(data_path, experiment_folder)

files = [f for f in os.listdir(full_path) if f.endswith('.csv')]
df = pd.concat([pd.read_csv(os.path.join(full_path, f)) for f in files], ignore_index=True)
df = df[df["TargetLoc"] != "Front"]


# --- Preprocessing ---
# Ensure types
if 'Subject ID' in df.columns:
    df['subject_id'] = df['Subject ID'].astype(int, errors="ignore")
elif 'subject_id' in df.columns:
    df['subject_id'] = df['subject_id'].astype(int, errors="ignore")

# Map Locations
if 'SingletonLoc' in df.columns:
    if pd.api.types.is_numeric_dtype(df['SingletonLoc']):
        df['SingletonLoc'] = df['SingletonLoc'].map({0: 'Absent', 1: 'Left', 2: 'Front', 3: 'Right'})
if 'target_loc' in df.columns:
    df['TargetLoc'] = df['target_loc']
if pd.api.types.is_numeric_dtype(df['TargetLoc']):
    df['TargetLoc'] = df['TargetLoc'].replace({1: 'Left', 2: 'Front', 3: 'Right'})

df['HP_Distractor_Loc'] = df['HP_Distractor_Loc'].replace({1: 'Left', 2: 'Front', 3: 'Right'})

# Determine location column
loc_col = "SingletonLoc"

# Define probability condition based on Subject ID and Location
def get_probability(row):
    if row[loc_col] == 'Absent':
        return 'Absent'
    # The HP_distractor_loc switches dynamically
    return 'High' if row[loc_col] == row['HP_Distractor_Loc'] else 'Low'

df['Probability'] = df.apply(get_probability, axis=1)
df['DistractorProb'] = df['Probability']

# --- Data Splitting ---
df_acc = df.copy()

# Remove outliers (to match analysis conditions) - Only for RT analysis
if "rt" in df.columns:
    pass
    #df = remove_outliers(df, threshold=OUTLIER_THRESH, column_name="rt", subject_id_column="Subject ID").reset_index(drop=True)

# --- Analysis: Adaptation to Probability Switch ---
print("Analyzing adaptation to probability switch...")

# 1. Prepare HP Distractor Location Data
# Ensure readable labels if numeric
if pd.api.types.is_numeric_dtype(df['HP_Distractor_Loc']):
    df['HP_Distractor_Loc'] = df['HP_Distractor_Loc'].map({1: 'Left', 2: 'Front', 3: 'Right', 0: 'Absent'})
    df_acc['HP_Distractor_Loc'] = df_acc['HP_Distractor_Loc'].map({1: 'Left', 2: 'Front', 3: 'Right', 0: 'Absent'})

# --- DIAGNOSTIC: Check trial counts before and after filtering ---
print("\n--- Diagnostic: Trial Counts ---")
print(f"Total trials loaded (after removing Front targets): {len(df)}")
print("Value counts for 'HP_Distractor_Loc' before filtering:")
print(df['HP_Distractor_Loc'].value_counts(dropna=False))

# Filter for relevant HP locations (Left/Right)
df = df[df['HP_Distractor_Loc'].isin(['Left', 'Right'])].reset_index(drop=True)
df_acc = df_acc[df_acc['HP_Distractor_Loc'].isin(['Left', 'Right'])].reset_index(drop=True)

print(f"\nTotal trials after filtering for HP_Distractor_Loc in ['Left', 'Right']: {len(df)}")
print("This filtering is the reason the 'TrialIndex' in the plots is shorter than the total experiment duration.")
print("The analysis is focusing only on trials where the high-probability location was explicitly Left or Right.")
print("--------------------------------\n")

# 2. Calculate Running Averages (Rolling Window)
WINDOW = 45

def calculate_rolling_metrics(sub_df):
    # A. General RT (Speed)
    sub_df['RT_Rolling'] = sub_df['rt'].rolling(window=WINDOW, center=True, min_periods=1).mean()

    # B. Spatial Bias (RT Difference: Right - Left)
    # If HighProb is Left, we expect suppression of Left -> RT_Left < RT_Right -> Bias > 0.
    # If HighProb is Right, we expect suppression of Right -> RT_Right < RT_Left -> Bias < 0.
    # This should correlate positively with HP_Signal (Left=1, Right=-1).
    sub_df['RT_Left_Dist'] = sub_df['rt'].where(sub_df['SingletonLoc'] == 'Left')
    sub_df['RT_Right_Dist'] = sub_df['rt'].where(sub_df['SingletonLoc'] == 'Right')

    sub_df['RT_Left_Roll'] = sub_df['RT_Left_Dist'].rolling(window=WINDOW, min_periods=1, center=True).mean()
    sub_df['RT_Right_Roll'] = sub_df['RT_Right_Dist'].rolling(window=WINDOW, min_periods=1, center=True).mean()

    sub_df['RT_Index'] = (sub_df['RT_Left_Roll'] - sub_df['RT_Right_Roll']) / (sub_df['RT_Left_Roll'] + sub_df['RT_Right_Roll'])

    # C. Accuracy
    sub_df['Acc_Rolling'] = sub_df['IsCorrect'].rolling(window=WINDOW, center=True, min_periods=1).mean()

    sub_df['Acc_Left_Dist'] = sub_df['IsCorrect'].where(sub_df['SingletonLoc'] == 'Left')
    sub_df['Acc_Right_Dist'] = sub_df['IsCorrect'].where(sub_df['SingletonLoc'] == 'Right')

    sub_df['Acc_Left_Roll'] = sub_df['Acc_Left_Dist'].rolling(window=WINDOW, min_periods=1, center=True).mean()
    sub_df['Acc_Right_Roll'] = sub_df['Acc_Right_Dist'].rolling(window=WINDOW, min_periods=1, center=True).mean()

    sub_df['Acc_Index'] = (sub_df['Acc_Left_Roll'] - sub_df['Acc_Right_Roll']) / (sub_df['Acc_Left_Roll'] + sub_df['Acc_Right_Roll'])
    return sub_df

df = df.groupby('subject_id', group_keys=False).apply(calculate_rolling_metrics)
df['TrialIndex'] = df.groupby('subject_id').cumcount()

df_acc = df_acc.groupby('subject_id', group_keys=False).apply(calculate_rolling_metrics)
df_acc['TrialIndex'] = df_acc.groupby('subject_id').cumcount()

# 3. Plotting
subjects = df['subject_id'].unique()
n_subs = len(subjects)
cols = 3
rows = (n_subs + cols - 1) // cols

fig, axes = plt.subplots(rows, cols, figsize=(18, 4 * rows), sharex=True)
axes = axes.flatten()

for i, sub in enumerate(subjects):
    ax = axes[i]
    sub_df = df[df['subject_id'] == sub]

    # Plot HP Location
    # Map to numeric for plotting position: Left -> -1 (Bottom), Right -> 1 (Top)
    hp_plot_vals = sub_df['HP_Distractor_Loc'].map({'Left': -1, 'Right': 1})
    ax.plot(sub_df['TrialIndex'], hp_plot_vals, color='black', linestyle='--', alpha=0.5)
    ax.set_title(f"Subject {sub}")
    ax.set_ylabel("HP Location")
    ax.set_yticks([-1, 1])
    ax.set_yticklabels(["Left", "Right"])
    ax.set_ylim(-1.5, 1.5)

    # Plot RT Bias
    ax2 = ax.twinx()
    sns.lineplot(data=sub_df, x='TrialIndex', y='RT_Index', ax=ax2, color='tab:blue')
    ax2.set_ylabel("RT Index", color='tab:blue')
    ax2.tick_params(axis='y', labelcolor='tab:blue')
    ax2.axhline(0, color='gray', linestyle=':', alpha=0.5)

# Hide empty subplots
for j in range(i + 1, len(axes)):
    axes[j].axis('off')

plt.tight_layout()
plt.show()

# 3b. Plotting Accuracy
fig, axes = plt.subplots(rows, cols, figsize=(18, 4 * rows), sharex=True)
axes = axes.flatten()
fig.suptitle("Accuracy Bias Adaptation")

for i, sub in enumerate(subjects):
    ax = axes[i]
    sub_df = df_acc[df_acc['subject_id'] == sub]

    # Plot HP Location
    hp_plot_vals = sub_df['HP_Distractor_Loc'].map({'Left': 1, 'Right': -1})
    ax.plot(sub_df['TrialIndex'], hp_plot_vals, color='black', linestyle='--', alpha=0.5)
    ax.set_title(f"Subject {sub}")
    ax.set_ylabel("HP Location")
    ax.set_yticks([-1, 1])
    ax.set_yticklabels(["Right", "Left"])
    ax.set_ylim(-1.5, 1.5)

    # Plot Acc Bias
    ax2 = ax.twinx()
    sns.lineplot(data=sub_df, x='TrialIndex', y='Acc_Index', ax=ax2, color='tab:green')
    ax2.set_ylabel("Acc Index", color='tab:green')
    ax2.tick_params(axis='y', labelcolor='tab:green')
    ax2.axhline(0, color='gray', linestyle=':', alpha=0.5)

# Hide empty subplots
for j in range(i + 1, len(axes)):
    axes[j].axis('off')

plt.tight_layout()
plt.show()

# 4. Cross-Correlation
lags = np.arange(-100, 100)
cross_corr_dict = {}

for sub_id, sub_df in df.groupby('subject_id'):
    valid_data = sub_df[['HP_Distractor_Loc', 'RT_Index']].dropna()
    # Create numeric signal for correlation: Left=-1, Right=1
    hp_numeric = valid_data['HP_Distractor_Loc'].map({'Left': -1, 'Right': 1})
    
    if not valid_data.empty and hp_numeric.std() > 0 and valid_data['RT_Index'].std() > 0:
        sig1 = (hp_numeric - hp_numeric.mean()) / hp_numeric.std()
        sig2 = (valid_data['RT_Index'] - valid_data['RT_Index'].mean()) / valid_data['RT_Index'].std()
        # Use shift(-lag) so that Positive Lag = Behavior follows Signal (Delay)
        cc = [sig1.corr(sig2.shift(-lag)) for lag in lags]
        cross_corr_dict[sub_id] = cc

if cross_corr_dict:
    # Plot Subject-wise
    subjects = list(cross_corr_dict.keys())
    n_subs = len(subjects)
    cols = 3
    rows = (n_subs + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(18, 4 * rows), sharex=True, sharey=True)
    axes = axes.flatten()
    fig.suptitle("Subject-wise Cross-Correlation: HPD Location vs. RT Index")

    for i, sub in enumerate(subjects):
        ax = axes[i]
        ax.plot(lags, cross_corr_dict[sub])
        ax.set_title(f"Subject {sub}")
        ax.axvline(0, color='k', linestyle='--')
        ax.grid(True, alpha=0.3)

    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()

# 4b. Cross-Correlation (Accuracy)
cross_corr_dict_acc = {}

for sub_id, sub_df in df_acc.groupby('subject_id'):
    valid_data = sub_df[['HP_Distractor_Loc', 'Acc_Index']].dropna()
    hp_numeric = valid_data['HP_Distractor_Loc'].map({'Left': 1, 'Right': -1})
    
    if not valid_data.empty and hp_numeric.std() > 0 and valid_data['Acc_Index'].std() > 0:
        sig1 = (hp_numeric - hp_numeric.mean()) / hp_numeric.std()
        sig2 = (valid_data['Acc_Index'] - valid_data['Acc_Index'].mean()) / valid_data['Acc_Index'].std()
        cc = [sig1.corr(sig2.shift(-lag)) for lag in lags]
        cross_corr_dict_acc[sub_id] = cc

if cross_corr_dict_acc:
    # Plot Subject-wise
    subjects = list(cross_corr_dict_acc.keys())
    n_subs = len(subjects)
    cols = 3
    rows = (n_subs + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(18, 4 * rows), sharex=True, sharey=True)
    axes = axes.flatten()
    fig.suptitle("Subject-wise Cross-Correlation: HPD Location vs. Accuracy Index")

    for i, sub in enumerate(subjects):
        ax = axes[i]
        ax.plot(lags, cross_corr_dict_acc[sub], color='tab:green')
        ax.set_title(f"Subject {sub}")
        ax.axvline(0, color='k', linestyle='--')
        ax.grid(True, alpha=0.3)

    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()

# 5. Group Level Analysis
print("\n--- Group Level Cross-Correlation Analysis ---")

def plot_group_cross_corr(cc_dict, title_suffix, color='tab:blue'):
    if not cc_dict:
        print(f"No data for {title_suffix}")
        return

    # Convert to DataFrame: rows=subjects, cols=lags
    cc_df = pd.DataFrame.from_dict(cc_dict, orient='index', columns=lags)
    
    # Mean and SEM across subjects
    cc_mean = cc_df.mean(axis=0)
    cc_sem = cc_df.sem(axis=0)
    
    # Subject-wise average (average all lags per subject)
    subject_avg_cc = cc_df.mean(axis=1)
    
    # T-test against 0 on the subject averages
    t_stat, p_val = ttest_1samp(subject_avg_cc, 0, nan_policy='omit')
    
    print(f"--- {title_suffix} ---")
    print(f"Subject-wise Mean Cross-Correlation (averaged over lags): {subject_avg_cc.mean():.5f}")
    print(f"One-sample t-test against 0: t = {t_stat:.4f}, p = {p_val:.4f}")
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(lags, cc_mean, color=color, label='Mean Correlation')
    plt.fill_between(lags, cc_mean - cc_sem, cc_mean + cc_sem, color=color, alpha=0.3, label='SEM')

    plt.axhline(0, color='k', linestyle='--')
    plt.axvline(0, color='k', linestyle=':', alpha=0.5)
    plt.title(f"Average Cross-Correlation: {title_suffix}")
    plt.xlabel("Lag (Trials)")
    plt.ylabel("Correlation Coefficient")
    plt.legend()
    plt.tight_layout()
    plt.show()

plot_group_cross_corr(cross_corr_dict, "HPD Location vs. RT", color='tab:blue')
sns.despine()

plot_group_cross_corr(cross_corr_dict_acc, "HPD Location vs. Accuracy", color='tab:green')
sns.despine()
