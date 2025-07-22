import matplotlib.pyplot as plt
import SPACEPRIME
import pandas as pd
import statsmodels.formula.api as smf
import seaborn as sns
from stats import remove_outliers, calculate_bis  # Assuming this is your custom outlier removal function
from scipy.stats import ttest_rel

plt.ion()

# --- Script Configuration Parameters ---

# --- 1. Data Loading & Preprocessing ---
OUTLIER_RT_THRESHOLD = 2.0
FILTER_PHASE = 2

# --- 2. Column Names ---
SUBJECT_ID_COL = 'subject_id'
TARGET_COL = 'TargetLoc'
DISTRACTOR_COL = 'SingletonLoc'
REACTION_TIME_COL = 'rt'
ACCURACY_COL = 'select_target'
PHASE_COL = 'phase'
PRIMING_COL = 'Priming'
TRIAL_NUMBER_COL = 'total_trial_nr'
ACCURACY_INT_COL = 'select_target_int'
BLOCK_COL = 'block'


# --- MERGED: ERP component columns for both Latency and Amplitude ---
ERP_N2AC_LATENCY_COL = 'N2ac_latency'
ERP_PD_LATENCY_COL = 'Pd_latency'
ERP_N2AC_AMPLITUDE_COL = 'N2ac_amplitude'
ERP_PD_AMPLITUDE_COL = 'Pd_amplitude'
RT_SPLIT_COL = "RT_split"

# --- Mappings and Reference Levels ---
TARGET_LOC_MAP = {1: "left", 2: "mid", 3: "right"}
DISTRACTOR_LOC_MAP = {0: "absent", 1: "left", 2: "mid", 3: "right"}
PRIMING_MAP = {-1: "np", 0: "no-p", 1: "pp"}
TARGET_REF_STR = TARGET_LOC_MAP.get(2)
DISTRACTOR_REF_STR = DISTRACTOR_LOC_MAP.get(2)
PRIMING_REF_STR = PRIMING_MAP.get(0)

# --- 3. ERP Component Definitions ---
PD_TIME_WINDOW = (0.2, 0.4)
PD_ELECTRODES = [("FC3", "FC4"), ("FC5", "FC6"), ("C3", "C4"), ("C5", "C6"), ("CP3", "CP4"), ("CP5", "CP6")]
N2AC_TIME_WINDOW = (0.2, 0.4)
N2AC_ELECTRODES = [("FC3", "FC4"), ("FC5", "FC6"), ("C3", "C4"), ("C5", "C6"), ("CP3", "CP4"), ("CP5", "CP6")]

# --- Latency Robustness Check Configuration ---
PERCENTAGES_TO_TEST = [0.3, 0.5, 0.7] # Using 50% as the standard

# --- Main Script ---
print("Loading and concatenating epochs...")
epochs = SPACEPRIME.load_concatenated_epochs("spaceprime")
df = epochs.metadata.copy()
sfreq = epochs.info["sfreq"]
print(f"Original number of trials: {len(df)}")

# --- Preprocessing Steps (unchanged) ---
if PHASE_COL in df.columns and FILTER_PHASE is not None:
    df = df[df[PHASE_COL] != FILTER_PHASE]
if REACTION_TIME_COL in df.columns:
    df = remove_outliers(df, column_name=REACTION_TIME_COL, threshold=OUTLIER_RT_THRESHOLD)
if SUBJECT_ID_COL in df.columns:
    df[TRIAL_NUMBER_COL] = df.groupby(SUBJECT_ID_COL).cumcount()
if ACCURACY_COL in df.columns:
    df[ACCURACY_INT_COL] = df[ACCURACY_COL].astype(int)
# Map categorical variables to strings
df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors='coerce').map(TARGET_LOC_MAP)
df[DISTRACTOR_COL] = pd.to_numeric(df[DISTRACTOR_COL], errors='coerce').map(DISTRACTOR_LOC_MAP)
df[PRIMING_COL] = pd.to_numeric(df[PRIMING_COL], errors='coerce').map(PRIMING_MAP)
df[SUBJECT_ID_COL] = df[SUBJECT_ID_COL].astype(str)

print("Preprocessing and column mapping complete.")

df[RT_SPLIT_COL] = df.groupby(SUBJECT_ID_COL)[REACTION_TIME_COL].transform(
    lambda x: pd.qcut(x, 2, labels=['fast', 'slow'], duplicates='drop'))

priming_df = df.groupby([SUBJECT_ID_COL, PRIMING_COL])[["rt", "select_target_int"]].mean().reset_index()
priming_df_bis = calculate_bis(priming_df, rt_col="rt", pc_col="select_target_int")

# 3. Now, create the plot using the 'df' DataFrame, which has all the columns we need.
print("Generating plot...")
plt.figure(figsize=(8, 6)) # Create a new figure for the plot
sns.barplot(data=priming_df_bis, x="Priming", y="bis", order=["np", "no-p", "pp"], errorbar=("se", 1))
plt.title("Balanced Integration Score by Priming Condition")
plt.ylabel("Balanced Integration Score (BIS)")
plt.xlabel("Priming Condition")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 6))
sns.barplot(data=priming_df_bis, x="Priming", y="rt", order=["np", "no-p", "pp"], errorbar=("se", 1))
plt.title("Reaction Time by Priming Condition")
plt.ylabel("Reaction Time (s)")
plt.xlabel("Priming Condition")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 6))
sns.barplot(data=priming_df_bis, x="Priming", y="select_target_int", order=["np", "no-p", "pp"], errorbar=("se", 1))
plt.title("Reaction Time by Priming Condition")
plt.ylabel("PC")
plt.xlabel("Priming Condition")
plt.tight_layout()
plt.show()

rt_split_df = df.groupby([SUBJECT_ID_COL, RT_SPLIT_COL])[["rt", "select_target_int"]].mean().reset_index()
rt_split_df_bis = calculate_bis(rt_split_df, rt_col="rt", pc_col="select_target_int")

plt.figure(figsize=(8, 6))
sns.barplot(data=rt_split_df_bis, x="RT_split", y="bis", errorbar=("se", 1))
plt.title("Reaction Time by Priming Condition")
plt.ylabel("BIS")
plt.xlabel("Priming Condition")
plt.tight_layout()
plt.show()

# run ttest
ttest_rel(priming_df_bis.query("Priming=='no-p'")["bis"], priming_df_bis.query("Priming=='pp'")["bis"])
ttest_rel(priming_df_bis.query("Priming=='no-p'")["bis"], priming_df_bis.query("Priming=='np'")["bis"])

# Now, run a Linear Mixed-Effects Model
# This models 'bis' as a function of 'Priming',
# while accounting for random variations between subjects.
# The (1 | subject_id) part tells the model that trials are nested within subjects.
model = smf.mixedlm("bis ~ Priming", data=priming_df_bis, groups=priming_df_bis[SUBJECT_ID_COL])
result = model.fit()

# Print the summary of the statistical model
print(result.summary())
