import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Qt5Agg")
plt.ion()


# load up dataframe
df = pd.read_excel("/home/max/data/behavior/SPACEPRIME/simulated_subjects_appended.xlsx")
# some cleaning
res = df[(df['event_type'] == 'response') & (df['rt'] != 0)]
# remove outlier rt values
#res.rt[res.rt > 2 * np.std(res.rt)] = np.nan  # drop values over 2 standard deviations of the mean
no_bs = res[res.bootstrapped==0]
# sanity check
fig = plt.figure(figsize=(15, 8))
sns.barplot(data=no_bs, x="TargetDigit", y="iscorrect")
plt.savefig("/home/max/figures/SPACEPRIME/targetdigit_correct_responses.png", dpi=400)

fig = plt.figure(figsize=(15, 8))
sns.barplot(data=no_bs, x="TargetDigit", y="iscorrect", hue="block")
plt.savefig("/home/max/figures/SPACEPRIME/targetdigit_correct_responses_block.png", dpi=400)

fig = plt.figure(figsize=(15, 8))
sns.barplot(data=no_bs, x="SingletonPresent", y="iscorrect", hue="block")
plt.savefig("/home/max/figures/SPACEPRIME/singleton_abs_vs_pres_iscorrect_block.png", dpi=400)

fig = plt.figure(figsize=(15, 8))
sns.barplot(data=no_bs, x="SpatialPriming", y="iscorrect", hue="block")
plt.savefig("/home/max/figures/SPACEPRIME/spatial_priming_iscorrect_block.png", dpi=400)

fig = plt.figure(figsize=(15, 8))
sns.barplot(data=no_bs, x="IdentityPriming", y="iscorrect", hue="block")
plt.savefig("/home/max/figures/SPACEPRIME/identity_priming_iscorrect_block.png", dpi=400)
