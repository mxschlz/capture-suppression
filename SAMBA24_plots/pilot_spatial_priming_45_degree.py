import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import numpy as np
plt.ion()

# load dataframe
fp = "/home/max/data/behavior/pilot/all_subjects_additional_metrics_and_priming.csv"
df = pd.read_csv(fp)
# get only singleton present trials
df = df[df["Singletonpres"] == 1]
# Filter for 'positive_priming' and 'negative_priming'
filtered_df = df[df['spatial_priming'].isin(['positive_priming', 'negative_priming'])]
filtered_df2 = df[df['temporal_priming'].isin(['positive_priming', 'negative_priming'])]
# Calculate mean correct by subject and priming type
mean_correct_df = filtered_df.groupby(['subject', 'spatial_priming'])['correct'].mean().reset_index(name='mean_correct')

# look for correlation between spatial and temporal priming
fig = plt.figure(figsize=(15, 8))
sns.scatterplot(data=df["spatial_priming"], x="negative_priming", y="positive_priming", hue="subject")
