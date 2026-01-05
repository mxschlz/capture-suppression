import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import glob
from SPACEPRIME import get_data_path
from SPACEPRIME.subjects import subject_ids


subject_ids.pop(subject_ids.index(124))
subject_ids.pop(subject_ids.index(156))
subject_ids.pop(subject_ids.index(174))

# load behavioral data because it contains information of trial sequences from all subjects
df = pd.concat([pd.read_csv(f).assign(subject_id=subject) for subject in subject_ids for f in sorted(glob.glob(f"{get_data_path()}sequences/sub-{subject}/sub-{subject}_block*.csv"))])

# Calculate subject-wise percentages
priming_dist = df.groupby("subject_id")["Priming"].value_counts(normalize=True).mul(100).rename("Percentage").reset_index()
singleton_dist = df.groupby("subject_id")["SingletonPresent"].value_counts(normalize=True).mul(100).rename("Percentage").reset_index()

# Calculate and print descriptive statistics
print("Priming Distribution (Mean +/- SD):")
print(priming_dist.groupby("Priming")["Percentage"].agg(["mean", "std"]))

print("\nSingleton Presence Distribution (Mean +/- SD):")
print(singleton_dist.groupby("SingletonPresent")["Percentage"].agg(["mean", "std"]))

# Plot
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
sns.boxplot(data=priming_dist, x="Priming", y="Percentage", ax=axes[0])
sns.boxplot(data=singleton_dist, x="SingletonPresent", y="Percentage", ax=axes[1])
plt.tight_layout()
plt.show()
