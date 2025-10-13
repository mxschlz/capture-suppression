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

subjects = sorted("/home/max/Insync/schulz.max5@gmail.com/GoogleDrive/PhD/data/SPACECUE_implicit/derivatives/preprocessing/sci-99/beh/")
df = pd.concat([pd.read_csv(f"/home/max/Insync/schulz.max5@gmail.com/GoogleDrive/PhD/data/SPACECUE_implicit/derivatives/preprocessing/sci-99/beh/{file}") for file in os.listdir("/home/max/Insync/schulz.max5@gmail.com/GoogleDrive/PhD/data/SPACECUE_implicit/derivatives/preprocessing/sci-99/beh/")])

#df = df.query(f"phase!={FILTER_PHASE}")
df["select_target"] = df["select_target"].astype(float)
#df = df[df["response"]!=np.nan]
#df = remove_outliers(df, threshold=OUTLIER_THRESH, column_name="rt")

df_mean = df.groupby(["subject_id", "DistractorProb"])[["rt", "select_target"]].mean().reset_index()

fig, ax = plt.subplots()

sns.barplot(data=df_mean, x="DistractorProb", y="select_target", ax=ax,
            errorbar=None, facecolor=(0, 0, 0, 0), edgecolor=".2")

sns.lineplot(data=df_mean, x="DistractorProb", y="select_target",
             hue="subject_id", estimator=None,
             linewidth=1, ax=ax, palette="tab10")

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

sns.barplot(data=df_targetloc_mean, x="target_at_high_prob_loc", y="select_target",
            ax=ax2, errorbar=("ci", 95), capsize=.1, palette="viridis")

sns.lineplot(data=df_targetloc_mean, x="target_at_high_prob_loc", y="select_target",
             hue="subject_id", estimator=None,
             linewidth=1, ax=ax2, palette="tab10")

ax2.set_xlabel("Target at High-Probability Distractor Location")
ax2.set_ylabel("Accuracy (%)")
plt.tight_layout()
plt.show()
