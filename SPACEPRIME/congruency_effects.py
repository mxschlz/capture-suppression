import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme(context="talk", style="ticks")
from SPACEPRIME import load_concatenated_csv
from scipy.stats import ttest_rel


df = load_concatenated_csv("target_towardness_all_variables.csv", index_col=0)

# Filter to only include the specific digits (4, 6) and lateral locations (1, 3)
# This ensures the "Incongruent" condition is not polluted by neutral trials (e.g. Front or Digit 5)
df = df[df['TargetDigit'].isin([4, 6]) & df['TargetLoc'].isin([1, 3])].copy()

# 1. Define the spatial mapping of the numpad
# Left column: 1, 4, 7 | Center column: 2, 5, 8 | Right column: 3, 6, 9
numpad_map = {4: 'Left', 6: 'Right'}

# 2. Map the TargetDigit to its expected spatial location
df['digit_spatial_loc'] = df['TargetDigit'].map(numpad_map)

# Map TargetLoc (1, 2, 3) to spatial strings to match the digit mapping
loc_map = {1: 'Left', 3: 'Right'}
df['target_spatial_loc'] = df['TargetLoc'].map(loc_map)

# 3. Determine Congruency
# We assume 'TargetLoc' contains strings like 'Left', 'Front', 'Right'.
# If your CSV uses different labels (e.g., lowercase or azimuth angles), adjust the comparison below.
df['is_congruent'] = df['target_spatial_loc'] == df['digit_spatial_loc']
df['congruency_label'] = df['is_congruent'].map({True: 'Congruent', False: 'Incongruent'})

# 4. Calculate Accuracy (Performance)
# Returns 1 if response matches target, 0 otherwise
df['accuracy'] = (df['response'] == df['TargetDigit']).astype(int)

# Calculate subject-wise mean accuracy
df_subject = df.groupby(['subject_id', 'congruency_label'])[['rt', 'accuracy', "target_towardness"]].mean().reset_index()

# define test param
test_param = "target_towardness"

# Perform paired t-test
pivot_df = df_subject.pivot(index='subject_id', columns='congruency_label', values=test_param).dropna()
t_stat, p_val = ttest_rel(pivot_df['Congruent'], pivot_df['Incongruent'])
print(f"Paired t-test: t({len(pivot_df)-1})={t_stat:.3f}, p={p_val:.4f}")

# 5. Visualize the results
plt.figure(figsize=(8, 6))

# Plot accuracy
ax = sns.boxplot(
    data=df_subject,
    x='congruency_label',
    y=test_param,
    palette="viridis"
)

ax.set_title(f'Effect of Spatial-Numerical Congruency (p={p_val:.3f})')
ax.set_xlabel('Condition')

sns.despine()
plt.show()
