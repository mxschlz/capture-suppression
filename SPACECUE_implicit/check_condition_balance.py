import pandas as pd
import os
import SPACECUE_implicit
import matplotlib.pyplot as plt
import seaborn as sns
from stats import remove_outliers

sns.set_theme(context="talk", style="ticks")

# --- Configuration ---
OUTLIER_THRESH = 2
experiment_folder = "pilot/distractor"

# --- Data Loading ---
print("Loading data...")
data_path = SPACECUE_implicit.get_data_path()
full_path = os.path.join(data_path, experiment_folder)

files = [f for f in os.listdir(full_path) if f.endswith('.csv')]
df = pd.concat([pd.read_csv(os.path.join(full_path, f)) for f in files], ignore_index=True)

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

# Determine location column
loc_col = "Non-Singleton2Loc" if "control" in experiment_folder else "SingletonLoc"

# Define Probability Logic
def get_probability(row):
    if row[loc_col] == 'Absent':
        return 'Absent'
    # Even subjects: Left is High, others are Low
    # Odd subjects: Right is High, others are Low
    is_even = row['subject_id'] % 2 == 0
    high_loc = 'Left' if is_even else 'Right'
    return 'High' if row[loc_col] == high_loc else 'Low'

df['DistractorProb'] = df.apply(get_probability, axis=1)

# Remove outliers (to match analysis conditions)
if "rt" in df.columns:
    df = remove_outliers(df, threshold=OUTLIER_THRESH, column_name="rt", subject_id_column="subject_id")

# --- Define Scenarios ---

# Scenario 1: Remove ONLY Frontal Targets
# Result: Low Prob contains both Lateral and Frontal distractors.
df_s1 = df[df["TargetLoc"] != "Front"].copy()
df_s1["Scenario"] = "1. No Front Targets"

# Scenario 2: Remove Frontal Targets AND Frontal Distractors
# Result: Low Prob contains only Lateral distractors (symmetric to High Prob).
df_s2 = df[(df["TargetLoc"] != "Front") & (df[loc_col] != "Front")].copy()
df_s2["Scenario"] = "2. No Front Targets/Distractors"

# Scenario 3: Remove ONLY Frontal Distractors
# Result: Targets can be Frontal (Harder). Low Prob contains only Lateral distractors.
df_s3 = df[df[loc_col] != "Front"].copy()
df_s3["Scenario"] = "3. No Front Distractors"

# --- Analysis 1: Trial Counts per Condition ---
print("Calculating trial counts...")
counts_s1 = df_s1.groupby(['subject_id', 'DistractorProb']).size().reset_index(name='TrialCount')
counts_s1["Scenario"] = "1. No Front Targets"

counts_s2 = df_s2.groupby(['subject_id', 'DistractorProb']).size().reset_index(name='TrialCount')
counts_s2["Scenario"] = "2. No Front Targets/Distractors"

counts_s3 = df_s3.groupby(['subject_id', 'DistractorProb']).size().reset_index(name='TrialCount')
counts_s3["Scenario"] = "3. No Front Distractors"

all_counts = pd.concat([counts_s1, counts_s2, counts_s3])

plt.figure(figsize=(10, 6))
sns.barplot(data=all_counts, x="DistractorProb", y="TrialCount", hue="Scenario",
            order=["High", "Low", "Absent"], palette="viridis", errorbar=("se", 1))
plt.title("Average Trial Counts per Subject by Condition")
plt.ylabel("Number of Trials")
sns.despine()
plt.show()

# --- Analysis 2: Location Matrices (Heatmaps) ---
# This visualizes exactly which geometric configurations remain in the dataset
def plot_matrix(dataframe, title):
    if dataframe.empty:
        print(f"Skipping heatmap for '{title}': Dataframe is empty.")
        return

    # Count occurrences of Distractor vs Target locations
    matrix = dataframe.groupby([loc_col, 'TargetLoc']).size().unstack(fill_value=0)
    
    # Reorder for visual clarity
    loc_order = [l for l in ['Left', 'Front', 'Right', 'Absent'] if l in matrix.index]
    target_order = [t for t in ['Left', 'Front', 'Right'] if t in matrix.columns]
    
    if not loc_order or not target_order:
        print(f"Skipping heatmap for '{title}': No matching locations found (Indices: {matrix.index.tolist()}, Columns: {matrix.columns.tolist()}).")
        return

    matrix = matrix.reindex(index=loc_order, columns=target_order)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(title)
    plt.ylabel("Distractor Location")
    plt.xlabel("Target Location")
    plt.show()

print("Plotting Location Matrices (Aggregated across all subjects)...")
plot_matrix(df_s1, "Scenario 1: No Front Targets\n(Trial Counts)")
plot_matrix(df_s2, "Scenario 2: No Front Targets & No Front Distractors\n(Trial Counts)")
plot_matrix(df_s3, "Scenario 3: No Front Distractors\n(Trial Counts)")

# --- Analysis 3: Balance Statistics ---
piv_s1 = counts_s1.pivot(index='subject_id', columns='DistractorProb', values='TrialCount')
piv_s1['Low_High_Ratio'] = piv_s1['Low'] / piv_s1['High']

piv_s2 = counts_s2.pivot(index='subject_id', columns='DistractorProb', values='TrialCount')
piv_s2['Low_High_Ratio'] = piv_s2['Low'] / piv_s2['High']

piv_s3 = counts_s3.pivot(index='subject_id', columns='DistractorProb', values='TrialCount')
piv_s3['Low_High_Ratio'] = piv_s3['Low'] / piv_s3['High']

print("\n--- Balance Statistics (Low / High Trial Ratio) ---")
print(f"Scenario 1 (No Front T) Mean Ratio: {piv_s1['Low_High_Ratio'].mean():.2f} (Low has ~2x trials)")
print(f"Scenario 2 (No Front T/D) Mean Ratio: {piv_s2['Low_High_Ratio'].mean():.2f} (Balanced)")
print(f"Scenario 3 (No Front D) Mean Ratio: {piv_s3['Low_High_Ratio'].mean():.2f} (Balanced)")

# --- Analysis 4: Performance Confound Check (RT by Target Location) ---
print("\nPlotting Performance Confound (RT by Target Location)...")
if "rt" in df.columns:
    # Filter for High/Low to focus on the specific comparison
    df_confound = df[df["DistractorProb"].isin(["High", "Low"])].copy()

    g = sns.catplot(
        data=df_confound, x="DistractorProb", y="rt", col="TargetLoc",
        col_order=["Left", "Front", "Right"], order=["High", "Low"],
        kind="bar", palette="viridis", errorbar=("se", 1),
        height=5, aspect=0.8
    )
    g.set_axis_labels("Distractor Probability", "RT (s)")
    g.fig.suptitle("RT by Distractor Probability split by Target Location\n(Check for Frontal Target Difficulty)", y=1.05)
    plt.show()

# --- Analysis 5: Geometric Balance Check ---
print("\nPlotting Geometric Balance (Target Counts per Distractor Location)...")
# Filter out Absent trials to focus on Distractor-Target relationship
df_geo = df[df[loc_col] != "Absent"].copy()

plt.figure(figsize=(10, 6))
sns.countplot(data=df_geo, x=loc_col, hue="TargetLoc",
              order=["Left", "Front", "Right"],
              hue_order=["Left", "Front", "Right"],
              palette="Set2")
plt.title("Balance Check: Target Locations per Distractor Location\n(Are targets balanced within each distractor position?)")
plt.xlabel("Distractor Location")
plt.ylabel("Trial Count")
plt.legend(title="Target Location")
sns.despine()
plt.show()
