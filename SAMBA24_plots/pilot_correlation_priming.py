import pandas as pd
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
plt.ion()

# load dataframe
fp = "/home/max/data/behavior/pilot/all_subjects_additional_metrics_and_priming.csv"
df = pd.read_csv(fp)
# get only singleton present trials
df = df[df["Singletonpres"] == 1]
# Filter for 'positive_priming' and 'negative_priming'
pp_spatial = df[df['spatial_priming'].isin(['positive_priming'])]
np_spatial = df[df['spatial_priming'].isin(['negative_priming'])]
pp_temporal = df[df['temporal_priming'].isin(['positive_priming'])]
np_temporal = df[df['temporal_priming'].isin(['negative_priming'])]

pp_mean_spat = pp_spatial.groupby(['subject', 'spatial_priming'])['correct'].mean().reset_index()
np_mean_spat = np_spatial.groupby(['subject', 'spatial_priming'])['correct'].mean().reset_index()

pp_mean_temp = pp_temporal.groupby(['subject', 'temporal_priming'])['correct'].mean().reset_index()
np_mean_temp = np_temporal.groupby(['subject', 'temporal_priming'])['correct'].mean().reset_index()

print(f"negative priming: {pearsonr(np_mean_spat['correct'], np_mean_temp['correct'])} \n"
      f"positive priming: {pearsonr(pp_mean_spat['correct'], pp_mean_temp['correct'])}")

sns.regplot(x=np_mean_spat['correct'], y=np_mean_temp['correct'])
sns.regplot(x=pp_mean_spat['correct'], y=pp_mean_temp['correct'])