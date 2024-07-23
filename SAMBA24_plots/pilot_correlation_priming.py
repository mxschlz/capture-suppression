import pandas as pd
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
import bootstrap
plt.ion()

# load dataframe
fp = "/home/max/data/behavior/CAPSUP/all_subjects_additional_metrics_and_priming.csv"
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
diff_spat = pp_mean_spat.correct - np_mean_spat.correct

pp_mean_temp = pp_temporal.groupby(['subject', 'temporal_priming'])['correct'].mean().reset_index()
np_mean_temp = np_temporal.groupby(['subject', 'temporal_priming'])['correct'].mean().reset_index()
diff_temp = pp_mean_temp.correct - np_mean_temp.correct


print(f"Pearson r correlation: {pearsonr(diff_temp, diff_spat)}")

sns.regplot(x=diff_temp, y=diff_spat)


# load dataframe
fp = "/home/max/data/behavior/SPACEPRIME/results_July_01_2024_14_30_30.xlsx"
df = pd.read_excel(fp, index_col=0)

res = df[(df['event_type'] == 'response') & (df['rt'] != 0)]
bs = bootstrap.generate_simulated_subjects(res, num_new_subjects=10000, max_subject_id=99)
new = pd.concat([res, bs])
# Filter for 'positive_priming' and 'negative_priming'
pp_spatial = new[new['SpatialPriming'].isin([1])]
np_spatial = new[new['SpatialPriming'].isin([-1])]
pp_identity = new[new['IdentityPriming'].isin([1])]
np_identity = new[new['IdentityPriming'].isin([-1])]

pp_mean_spat = pp_spatial.groupby(['subject_id', 'SpatialPriming'])['iscorrect'].mean().reset_index()
np_mean_spat = np_spatial.groupby(['subject_id', 'SpatialPriming'])['iscorrect'].mean().reset_index()
diff_spat = pp_mean_spat.iscorrect - np_mean_spat.iscorrect

pp_mean_ident = pp_identity.groupby(['subject_id', 'IdentityPriming'])['iscorrect'].mean().reset_index()
np_mean_ident = np_identity.groupby(['subject_id', 'IdentityPriming'])['iscorrect'].mean().reset_index()
diff_ident = pp_mean_ident.iscorrect - np_mean_ident.iscorrect


print(f"Pearson r correlation: {pearsonr(diff_ident, diff_spat)}")

sns.regplot(x=diff_ident, y=diff_spat)

