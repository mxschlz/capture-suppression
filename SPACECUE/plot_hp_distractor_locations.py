import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import SPACECUE
import glob
import seaborn as sns
sns.set_theme(style="ticks", context="talk", palette="colorblind")
plt.ion()


subject_id = 3

n_blocks = 5
n_trials = 360


# --- Data Loading ---
print(f"Loading data for subject {subject_id}...")
data_path = SPACECUE.get_data_path()
# Assuming this is the main experiment, not control
experiment_folder = "/derivatives/preprocessing"

# Load behavioral data for the single subject according to BIDS format
beh_data_base_path = f"{data_path}{experiment_folder}"
subject_folder = f"{beh_data_base_path}/sci-{subject_id}"

df_list = []
if os.path.exists(subject_folder):
    beh_files = glob.glob(f"{subject_folder}/beh/*.csv")
    for file in beh_files:
        df_list.append(pd.read_csv(file))
else:
    print(f"Error: Subject folder not found at {subject_folder}")

if not df_list:
    print(f"Error: No behavioral .csv files found for subject {subject_id}")

df = pd.concat(df_list, ignore_index=True)

# Check for the new column
if "HP_Distractor_Loc" not in df.columns:
    print("Error: 'HP_Distractor_Loc' column missing. Please regenerate sequences with the updated logic.")

# Determine which column represents the actual distractor location
cue_type = "distractor"
if cue_type == "distractor":
    loc_col = "SingletonLoc"
elif cue_type == "control":
    loc_col = "Non-Singleton2Loc"
else:
    loc_col = "SingletonLoc"  # Fallback

# Filter for Singleton Present trials
# Assuming SingletonPresent is 1 (Present) and 0 (Absent)
if "SingletonPresent" in df.columns:
    sp_trials = df[df["SingletonPresent"] == 1].copy()
else:
    sp_trials = df.copy()

# Encode combined (loc, prob) state into 4 levels:
# 1 = Left (80%), 2 = Left (60%), 3 = Right (60%), 4 = Right (80%)
def encode_hp_state(loc, prob):
    try:
        prob = round(float(prob), 2)
    except:
        pass
    loc_str = str(loc).strip().lower()
    if loc_str in ['1', '1.0', 'left']:  # Left
        return 1 if prob == 0.8 else 2
    else:  # Right
        return 3 if prob == 0.6 else 4

df["HP_State"] = df.apply(lambda r: encode_hp_state(r["HP_Distractor_Loc"], r["HP_Distractor_Prob"]), axis=1)
df.dropna(subset=[loc_col], inplace=True)

# --- TRANSITION COVERAGE CHECK ---
hp_states = df["HP_State"].values
observed_transitions = set(zip(hp_states[:-1], hp_states[1:]))
all_states = sorted(df["HP_State"].unique())
all_transitions = {(a, b) for a in all_states for b in all_states if a != b}
missing = all_transitions - observed_transitions
if missing:
    state_labels = {1: "Left(80%)", 2: "Left(60%)", 3: "Right(60%)", 4: "Right(80%)"}
    missing_str = ", ".join(f"{state_labels[a]}→{state_labels[b]}" for a, b in sorted(missing))
    print(f"WARNING: {len(missing)} transition(s) never observed: {missing_str}")
else:
    print(f"OK: all {len(all_transitions)} transitions observed at least once.")

# --- PLOTTING ---
fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(14, 8), sharex=True)

# 1. TOP PANEL: Plot the HP Rule (The intended high-probability condition)
# drawstyle="steps-post" ensures the line stays flat until the change occurs
ax1.plot(df.index, df["HP_State"], color="#2ca02c", linewidth=2.5, drawstyle="steps-post", alpha=0.9)

ax1.set_yticks([1, 2, 3, 4])
ax1.set_yticklabels(["Left (80%)", "Left (60%)", "Right (60%)", "Right (80%)"])
ax1.set_ylabel("HP Rule")
ax1.grid(True, axis='y', linestyle=':', alpha=0.4)
ax1.set_ylim(0.5, 4.5)

# 2. BOTTOM PANEL: Plot Actual Distractor Locations
# Map physical locations strictly to y-axis without probability conflation
def get_actual_loc_y(row):
    loc = row[loc_col]
    loc_str = str(loc).strip().lower()
    if loc_str in ['1', '1.0', 'left']:  # Left
        return -1
    elif loc_str in ['3', '3.0', 'right']:  # Right
        return 1
    elif loc_str in ['2', '2.0', 'front', 'center']:  # Frontal/Center
        return 0
    else:
        return np.nan

y_actual = sp_trials.apply(get_actual_loc_y, axis=1)

# Add jitter to y-axis to visualize density
y_jitter = np.random.uniform(-0.15, 0.15, size=len(sp_trials))

# Color code dots by location for extra clarity
colors = y_actual.map({-1: "#d62728", 0: "grey", 1: "#1f77b4"}).fillna("black")

ax2.scatter(sp_trials.index, y_actual + y_jitter,
            alpha=0.4, c=colors.values, s=20, edgecolors='none', label=f"Actual {loc_col}")

ax2.set_yticks([-1, 0, 1])
ax2.set_yticklabels(["Left", "Frontal", "Right"])
ax2.set_ylabel("Distractor location")
ax2.set_xlabel("Trial number")
ax2.grid(True, axis='y', linestyle=':', alpha=0.4)
ax2.set_ylim(-1.5, 1.5)

# Add Block Boundaries to both panels
for ax in [ax1, ax2]:
    for b in range(n_blocks + 1):
        boundary = b * n_trials
        ax.axvline(x=boundary, color='gray', linestyle='--', alpha=0.5)
        if ax == ax1 and b < n_blocks:
            # Label blocks
            ax.text(boundary + (n_trials / 2), 4.2, f"Block {b}",
                     ha='center', va='bottom', fontsize=9, color='gray')

plt.tight_layout()
sns.despine()