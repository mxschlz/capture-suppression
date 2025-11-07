import pandas as pd
from stats import remove_outliers
import os
import SPACECUE_implicit
import matplotlib.pyplot as plt
import seaborn as sns
import pingouin as pg
import numpy as np


FILTER_PHASE = 2
OUTLIER_THRESH = 2

data_path = SPACECUE_implicit.get_data_path()
subjects = sorted(f"{data_path}/derivatives/preprocessing/sci-99/beh/")
df = pd.concat([pd.read_csv(f"{data_path}/derivatives/preprocessing/sci-99/beh/control/{file}") for file in os.listdir(f"{data_path}/derivatives/preprocessing/sci-99/beh/control")])

df = df.query(f"phase!={FILTER_PHASE}")
df = df.query('DistractorProb != "distractor-absent"')

df["select_target"] = df["select_target"].astype(float)
df = remove_outliers(df, threshold=OUTLIER_THRESH, column_name="rt")

df_mean = df.groupby(["subject_id", "DistractorProb"])[["rt", "select_target"]].mean().reset_index()

df_mean['DistractorProb'] = pd.Categorical(df_mean['DistractorProb'], categories=["low-probability", "high-probability"], ordered=True)
df_mean = df_mean.sort_values('DistractorProb')

fig, ax = plt.subplots()

order = ["low-probability", "high-probability"]
sns.barplot(data=df_mean, x="DistractorProb", y="select_target", ax=ax,
            errorbar=("se", 1), facecolor=(0, 0, 0, 0), edgecolor=".2", order=order)

sns.lineplot(data=df_mean, x="DistractorProb", y="select_target",
             hue="subject_id", estimator=None,
             linewidth=1, ax=ax, palette="tab10")

# --- Pairwise comparisons ---
pairwise_tests = pg.pairwise_tests(data=df_mean, dv='select_target', within='DistractorProb', subject='subject_id', effsize='cohen')

def p_to_asterisks(p):
    if p < 0.001: return '***'
    if p < 0.01: return '**'
    if p < 0.05: return '*'
    return 'ns'

y_max = df_mean['select_target'].max()
y_pos = y_max + 0.05
h = 0.02

# Comparison: low-probability vs high-probability
stat2 = pairwise_tests[(pairwise_tests.A == 'low-probability') & (pairwise_tests.B == 'high-probability')].iloc[0]
x1, x2 = 0, 1
y, col = y_pos, 'k'
ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
ax.text((x1+x2)*.5, y+h, f"d={stat2['cohen']:.2f}", ha='center', va='bottom', color=col)

ax.set_ylim(top=y_pos + 0.2)

ax.set_ylabel("Accuracy (%)")
ax.set_xlabel("Distractor Probability")
plt.show()

# --- Performance when target is at the high-probability distractor location ---

# NOTE: This assumes a column 'high_prob_loc' exists in your dataframe,
# which we will create now.

# 1. Find the single high-probability location for each subject.
# We can get this by looking at the 'SingletonLoc' on any 'high-probability' trial for that subject.
high_prob_loc_map = df[df['DistractorProb'] == 'high-probability'].drop_duplicates('subject_id').set_index('subject_id')['SingletonLoc']

# 2. Map this location back to all trials for each subject.
df['high_prob_loc'] = df['subject_id'].map(high_prob_loc_map)

# 3. Now, correctly identify trials where the target appeared at that high-probability location.
df["target_at_high_prob_loc"] = df["TargetLoc"] == df["high_prob_loc"]
# Calculate the mean performance for each subject under each condition
df_targetloc_mean = df.groupby(["subject_id", "target_at_high_prob_loc"])["select_target"].mean().reset_index()

# Print the overall mean accuracy for each condition
print("Mean accuracy based on target location relative to high-probability distractor location:")
print(df_targetloc_mean.groupby("target_at_high_prob_loc")["select_target"].mean())

# Create a new figure to visualize the results
fig2, ax2 = plt.subplots(figsize=(6, 5))

order2 = [False, True]
sns.barplot(data=df_targetloc_mean, x="target_at_high_prob_loc", y="select_target",
            ax=ax2, errorbar=("ci", 95), capsize=.1, palette="viridis", order=order2)

sns.lineplot(data=df_targetloc_mean, x="target_at_high_prob_loc", y="select_target",
             hue="subject_id", estimator=None,
             linewidth=1, ax=ax2, palette="tab10")

# --- Pairwise comparisons ---
pairwise_tests2 = pg.pairwise_tests(data=df_targetloc_mean, dv='select_target', within='target_at_high_prob_loc', subject='subject_id', effsize='cohen')
stat_2 = pairwise_tests2.iloc[0]
x1, x2 = 0, 1
y_max2 = df_targetloc_mean['select_target'].max()
y, h, col = y_max2 + 0.01, 0.01, 'k'
ax2.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
ax2.text((x1+x2)*.5, y+h, f"d={stat_2['cohen']:.2f}", ha='center', va='bottom', color=col)

ax2.set_ylim(top=y_max2 + 0.05)

ax2.set_xlabel("Target at High-Probability Distractor Location")
ax2.set_ylabel("Accuracy (%)")
plt.tight_layout()
plt.show()
