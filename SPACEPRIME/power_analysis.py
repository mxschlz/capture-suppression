# estimate sample size via power analysis
from statsmodels.stats.power import TTestIndPower
from stats import cohen_d
import pandas as pd
from stats import cohen_d_rm

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
analysis = TTestIndPower()
result = analysis.solve_power(spatial_d_rm, power=power, nobs1=None, ratio=1.0, alpha=alpha)
print('Sample Size: %.3f' % result)