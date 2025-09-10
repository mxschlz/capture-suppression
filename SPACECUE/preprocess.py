import pandas as pd
from stats import remove_outliers
import os
import SPACECUE
import matplotlib.pyplot as plt
import seaborn as sns
import pingouin as pg


FILTER_PHASE = 4
OUTLIER_THRESH = 2

subjects = sorted(os.listdir(f"{SPACECUE.get_data_path()}derivatives\\preprocessing")[1:])
df = pd.concat([pd.read_csv(f"{SPACECUE.get_data_path()}derivatives\\preprocessing\\{subject}\\beh\\{subject}_clean.csv") for subject in subjects])

df = df.query(f"phase!={FILTER_PHASE}")
df = remove_outliers(df, threshold=OUTLIER_THRESH, column_name="rt")

# further preprocessing
df["rt"] = df["rt"] - df["cue_stim_delay_jitter"]

# do quantile split
df['delay'] = pd.qcut(df['cue_stim_delay_jitter'], q=3, labels=['low', 'medium', 'high'])

# Starting from your df_mean DataFrame
df_mean = df.groupby(["subject_id", "CueInstruction", "delay"])[["rt", "select_target"]].mean().reset_index()

# Pivot the table to get cue instructions as columns for the 'select_target' metric
df_pivot = df_mean.pivot_table(
    index=['subject_id', 'delay'],
    columns='CueInstruction',
    values='select_target'
).reset_index()

# The column index will have a name 'CueInstruction', which we can remove for clarity
df_pivot.columns.name = None

print("Pivoted DataFrame:")
print(df_pivot.head())

# Calculate the difference from the neutral condition
df_pivot['target_effect'] = df_pivot['cue_target_location'] - df_pivot['cue_neutral']
df_pivot['distractor_effect'] = df_pivot['cue_distractor_location'] - df_pivot['cue_neutral']

print("\nDataFrame with Calculated Effects:")
print(df_pivot.head())

# Melt the DataFrame to a long format for easy plotting
df_effects = pd.melt(
    df_pivot,
    id_vars=['subject_id', 'delay'],
    value_vars=['target_effect', 'distractor_effect'],
    var_name='effect_type',
    value_name='accuracy_effect'
)

# Now, create a bar plot of the effects
plt.figure(figsize=(8, 6))
sns.barplot(data=df_effects, x='delay', y='accuracy_effect', hue='effect_type', palette=['g', 'r'])

plt.title('Cueing Effect Relative to Neutral Condition')
plt.ylabel('Difference in Target Selection Rate')
plt.xlabel('Cue-Stimulus Delay')
plt.axhline(0, color='black', linestyle='--') # Add a line at 0 for reference
plt.show()

# --- Response Time Effect Analysis ---

# 1. Pivot the table to get cue instructions as columns for the 'rt' metric
df_pivot_rt = df_mean.pivot_table(
    index=['subject_id', 'delay'],
    columns='CueInstruction',
    values='rt'
).reset_index()

# Clean up the column index name
df_pivot_rt.columns.name = None

print("\nPivoted RT DataFrame:")
print(df_pivot_rt.head())


# 2. Calculate the RT benefit and cost relative to the neutral condition
# Target Benefit: Positive values indicate a faster response than neutral
df_pivot_rt['target_rt_benefit'] = df_pivot_rt['cue_neutral'] - df_pivot_rt['cue_target_location']

# Distractor Cost: Positive values indicate a slower response than neutral
df_pivot_rt['distractor_rt_cost'] = df_pivot_rt['cue_distractor_location'] - df_pivot_rt['cue_neutral']

print("\nRT DataFrame with Calculated Effects:")
print(df_pivot_rt.head())


# 3. Melt the DataFrame for easy plotting
df_rt_effects = pd.melt(
    df_pivot_rt,
    id_vars=['subject_id', 'delay'],
    value_vars=['target_rt_benefit', 'distractor_rt_cost'],
    var_name='effect_type',
    value_name='rt_effect_ms'
)

# 4. Create a bar plot of the RT effects
plt.figure(figsize=(8, 6))
sns.barplot(data=df_rt_effects, x='delay', y='rt_effect_ms', hue='effect_type', palette=['g', 'r'])

plt.title('Cueing Effect on Response Time Relative to Neutral')
plt.ylabel('Response Time Effect (s)')
plt.xlabel('Cue-Stimulus Delay')
plt.axhline(0, color='black', linestyle='--') # A line at 0 for reference
plt.legend(title='Effect Type')
plt.show()


# --- ANOVA Data Preparation ---

# 1. Get a unique age for each subject from the original dataframe
# We use .first() because age is constant for each subject
df_age = df.groupby('subject_id')['age'].first().reset_index()

# 2. Create the 'age_group' column
# Bins: [0, 35) is 'young', [35, infinity) is 'old'
df_age['age_group'] = pd.cut(df_age['age'],
                           bins=[0, 35, float('inf')],
                           labels=['young', 'old'],
                           right=False)

# --- 3x3x2 ANOVA Preparation (Using Raw Scores) ---

# 1. Start with the mean data per subject and condition.
# We use .copy() to ensure we don't modify the original df_mean.
df_anova_raw = df_mean.copy()

# 2. Merge the age_group information (df_age was created in the previous step)
df_anova_raw = pd.merge(df_anova_raw, df_age[['subject_id', 'age_group']], on='subject_id')

# 3. Create the combined within-subjects factor.
# This time, it will have 3 (cue) x 3 (delay) = 9 levels.
df_anova_raw['within_factor'] = df_anova_raw['CueInstruction'].astype(str) + "_" + df_anova_raw['delay'].astype(str)

print("\nSample of data prepared for 3x3x2 ANOVA:")
print(df_anova_raw.head())

# --- FIX: Ensure Dependent Variables are Numeric ---
# We explicitly convert the DV columns to a numeric type before the ANOVA.
# `errors='coerce'` will handle any problematic values by turning them into NaN.
df_anova_raw['select_target'] = pd.to_numeric(df_anova_raw['select_target'], errors='coerce')
df_anova_raw['rt'] = pd.to_numeric(df_anova_raw['rt'], errors='coerce')

# --- 3x3x2 Mixed ANOVA on Raw Accuracy (select_target) ---
print("\n--- Running 3x3x2 Mixed ANOVA on Raw Accuracy ---")

aov_acc_raw = pg.mixed_anova(data=df_anova_raw,
                             dv='select_target',
                             within='within_factor',
                             between='age_group',
                             subject='subject_id')

pg.print_table(aov_acc_raw)


# --- 3x3x2 Mixed ANOVA on Raw Response Time (rt) ---
print("\n--- Running 3x3x2 Mixed ANOVA on Raw Response Time ---")

aov_rt_raw = pg.mixed_anova(data=df_anova_raw,
                            dv='rt',
                            within='within_factor',
                            between='age_group',
                            subject='subject_id')

pg.print_table(aov_rt_raw)

# --- Prepare and Save Single-Trial DataFrame for Jamovi ---

# You are correct! Mixed models are most powerful with single-trial data.
# We will use the original 'df' DataFrame and add the 'age_group' to it.

# 1. Merge the 'age_group' information into the single-trial dataframe.
#    df_age was created earlier in the script.
df_single_trials = pd.merge(df, df_age[['subject_id', 'age_group']], on='subject_id')

# 2. Create the 'experimenter' column based on subject_id ranges.
#    We use pd.cut for a clean and efficient way to bin the numeric subject IDs.
bins = [0, 12, 24, 37, 49, 60]
labels = ['exp1', 'exp2', 'exp3', 'exp4', 'exp5']
df_single_trials['experimenter'] = pd.cut(df_single_trials['subject_id'],
                                          bins=bins,
                                          labels=labels,
                                          right=True)

# 3. Select the columns needed for the mixed model analysis in jamovi.
#    We'll include both the categorical 'delay' and the continuous 'cue_stim_delay_jitter'.
columns_to_keep = [
    'subject_id', 'age_group', 'experimenter', 'CueInstruction',
    'delay', 'cue_stim_delay_jitter', 'rt', 'select_target', 'absolute_trial_nr'
]
df_for_jamovi_single_trial = df_single_trials[columns_to_keep].copy()

# 4. Convert subject_id to a string to ensure it's treated as a nominal/categorical factor.
df_for_jamovi_single_trial["subject_id"] = df_for_jamovi_single_trial["subject_id"].astype(str)

# 5. Define a clear output path and a more descriptive filename.
#    Saving to the 'derivatives' folder for consistency.
output_path = os.path.join(SPACECUE.get_data_path(), 'concatenated')
output_filename = 'data_for_jamovi_single_trials.csv'
full_output_path = os.path.join(output_path, output_filename)

# 6. Ensure the output directory exists and save the file.
os.makedirs(output_path, exist_ok=True)
df_for_jamovi_single_trial.to_csv(full_output_path, index=False)

print(f"\nSingle-trial DataFrame successfully saved for jamovi import at:\n{full_output_path}")
print("\nThis file contains the following columns:")
print(df_for_jamovi_single_trial.columns.tolist())
print(f"\nTotal trials in the file: {len(df_for_jamovi_single_trial)}")
print("\nSample of the saved single-trial data:")
print(df_for_jamovi_single_trial.head())