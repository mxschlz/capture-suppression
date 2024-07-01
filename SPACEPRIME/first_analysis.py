import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Qt5Agg")
import numpy as np
plt.ion()
from plotting import barplot_with_single_subjects


# load up dataframe
df = pd.read_excel("/home/max/data/behavior/SPACEPRIME/simulated_subjects_appended.xlsx")
# some cleaning
res = df[(df['event_type'] == 'response') & (df['rt'] != 0)]
# remove outlier rt values
#res.rt[res.rt > 2 * np.std(res.rt)] = np.nan  # drop values over 2 standard deviations of the mean
no_bs = res[res.bootstrapped==0]
# plot singleton absent vs present trials
fig = plt.figure(figsize=(15, 8))
barplot_with_single_subjects(df=no_bs, x="SingletonPresent", y="iscorrect", groupby="SingletonPresent")
plt.savefig("/home/max/figures/SPACEPRIME/singleton_pres_vs_abs_iscorrect.png", dpi=400)
# plot singleton absent vs present trials
fig = plt.figure(figsize=(15, 8))
barplot_with_single_subjects(df=no_bs, x="SingletonPresent", y="rt", groupby="SingletonPresent")
plt.savefig("/home/max/figures/SPACEPRIME/singleton_pres_vs_abs_rt.png", dpi=400)
# plot priming conditions
# present_only = no_bs[no_bs["SingletonPresent"] == 1]
fig = plt.figure(figsize=(15, 8))
barplot_with_single_subjects(df=no_bs, x="SpatialPriming", y="iscorrect", groupby="SpatialPriming")
plt.savefig("/home/max/figures/SPACEPRIME/spatial_priming_iscorrect_barplot.png", dpi=400)
# plot priming conditions
fig = plt.figure(figsize=(15, 8))
barplot_with_single_subjects(df=no_bs, x="SpatialPriming", y="rt", groupby="SpatialPriming")
plt.savefig("/home/max/figures/SPACEPRIME/spatial_priming_rt_barplot.png", dpi=400)
# plot priming conditions
fig = plt.figure(figsize=(15, 8))
barplot_with_single_subjects(df=no_bs, x="IdentityPriming", y="iscorrect", groupby="IdentityPriming")
plt.savefig("/home/max/figures/SPACEPRIME/identity_priming_iscorrect_barplot.png", dpi=400)
# plot priming conditions
fig = plt.figure(figsize=(15, 8))
barplot_with_single_subjects(df=no_bs, x="IdentityPriming", y="rt", groupby="IdentityPriming")
plt.savefig("/home/max/figures/SPACEPRIME/identity_priming_rt_barplot.png", dpi=400)

