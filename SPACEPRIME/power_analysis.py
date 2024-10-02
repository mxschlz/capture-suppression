# estimate sample size via power analysis
import statsmodels.stats.power as smp
import numpy as np
import pandas as pd
from stats import cohen_d_rm
import matplotlib.pyplot as plt

# load up dataframe
df = pd.read_excel("/home/max/data/behavior/SPACEPRIME/results_July_06_2024_14_16_40.xlsx", index_col=0)
df.pop("Unnamed: 0")
# some cleaning
df = df[(df['event_type'] == 'response') & (df['rt'] != 0)]
# Convert `iscorrect` to numeric (True -> 1, False -> 0)
df['iscorrect'] = df['iscorrect'].astype(int)
# Create pivot tables for mean iscorrect by SpatialPriming and IdentityPriming
spatial_means = df.pivot_table(index='subject_id', columns='SpatialPriming', values='iscorrect').reset_index().rename(columns={-1: 'negative_spatial_prime',
                                                                                                                               0: 'no_spatial_prime',
                                                                                                                               1: 'positive_spatial_prime'})
identity_means = df.pivot_table(index='subject_id', columns='IdentityPriming', values='iscorrect').reset_index().rename(columns={-1: 'negative_identity_prime',
                                                                                                                               0: 'no_identity_prime',
                                                                                                                               1: 'positive_identity_prime'})
# Merge with original dataframe
df = df.merge(spatial_means, on='subject_id').merge(identity_means, on='subject_id')
# Calculate standard deviations for each condition
spatial_sd_neg = df['iscorrect'][df['SpatialPriming'] == -1].std()
spatial_sd_no = df['iscorrect'][df['SpatialPriming'] == 0].std()
spatial_sd_pos = df['iscorrect'][df['SpatialPriming'] == 1].std()
identity_sd_neg = df['iscorrect'][df['IdentityPriming'] == -1].std()
identity_sd_no = df['iscorrect'][df['IdentityPriming'] == 0].std()
identity_sd_pos = df['iscorrect'][df['IdentityPriming'] == 1].std()
# Calculate correlations between conditions for each subject
spatial_corr = df[['negative_spatial_prime', 'positive_spatial_prime']].corr().iloc[0, 1]
identity_corr = df[['negative_identity_prime', 'positive_identity_prime']].corr().iloc[0, 1]
# Calculate mean differences
spatial_mean_diff = df['negative_spatial_prime'].mean() - df['positive_spatial_prime'].mean()
identity_mean_diff = df['negative_identity_prime'].mean() - df['positive_identity_prime'].mean()
# Calculate Cohen's d_rm
spatial_d_rm = cohen_d_rm(spatial_mean_diff, spatial_sd_neg, spatial_sd_pos, spatial_corr)
identity_d_rm = cohen_d_rm(identity_mean_diff, identity_sd_neg, identity_sd_pos, identity_corr)
# Print results
print(f"Cohen's d_rm for SpatialPriming: {spatial_d_rm:.3f}")
print(f"Cohen's d_rm for IdentityPriming: {identity_d_rm:.3f}")

# parameters for power analysis
alpha = 0.05
power = 0.8
# perform power analysis
model = smp.TTestPower()
result_spatial = model.solve_power(spatial_d_rm, power=power, alpha=alpha)
result_identity = model.solve_power(identity_d_rm, power=power, alpha=alpha)
# plot power as a function of sample size
sample_sizes = np.array(range(5, 100))
model.plot_power(dep_var="nobs", nobs=sample_sizes, effect_size=[spatial_d_rm, identity_d_rm])
plt.legend([f"Cohen's d [spatial priming]: {spatial_d_rm:.3f}", f"Cohen's d [identity priming]: {identity_d_rm:.3f}"])
plt.ylabel('Power')
plt.hlines(y=power, xmin=0, xmax=100, linestyle='dashed', colors="black")
plt.vlines(x=result_spatial, ymin=0, ymax=1, linestyle='dashed', colors="green")
plt.vlines(x=result_identity, ymin=0, ymax=1, linestyle='dashed', colors="grey")
plt.savefig("/home/max/figures/SPACEPRIME/power_analysis_WP1.svg")

print(result_spatial)
print(result_identity)

# calculate estimated power for WP2 and 3
n = 50
result = model.solve_power(effect_size=0.5, power=None, alpha=alpha, nobs=n)

model_ind = smp.TTestIndPower()
result_ind = model_ind.solve_power(effect_size=0.6, power=None, alpha=alpha, nobs1=n)