import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Qt5Agg")
plt.ion()


# load up dataframe
df = pd.read_excel("/home/max/data/behavior/SPACEPRIME/results_July_01_2024_14_30_30.xlsx")
# some cleaning
res = df[(df['event_type'] == 'response') & (df['rt'] != 0)]
# remove outlier rt values
# sanity check
fig = plt.figure(figsize=(15, 8))
sns.barplot(data=res, x="TargetDigit", y="iscorrect")
plt.savefig("/home/max/figures/SPACEPRIME/targetdigit_correct_responses.png", dpi=400)

fig = plt.figure(figsize=(15, 8))
sns.barplot(data=res, x="TargetDigit", y="iscorrect", hue="block")
plt.savefig("/home/max/figures/SPACEPRIME/targetdigit_correct_responses_block.png", dpi=400)

fig = plt.figure(figsize=(15, 8))
sns.barplot(data=res, x="SingletonPresent", y="iscorrect", hue="block")
plt.savefig("/home/max/figures/SPACEPRIME/singleton_abs_vs_pres_iscorrect_block.png", dpi=400)

fig = plt.figure(figsize=(15, 8))
sns.barplot(data=res, x="SpatialPriming", y="iscorrect", hue="block")
plt.savefig("/home/max/figures/SPACEPRIME/spatial_priming_iscorrect_block.png", dpi=400)

fig = plt.figure(figsize=(15, 8))
sns.barplot(data=res, x="IdentityPriming", y="iscorrect", hue="block")
plt.savefig("/home/max/figures/SPACEPRIME/identity_priming_iscorrect_block.png", dpi=400)
