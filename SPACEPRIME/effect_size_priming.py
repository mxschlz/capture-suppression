import matplotlib
matplotlib.use("Qt5Agg")
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from stats import cohen_d
plt.ion()


# load up dataframe
df = pd.read_excel("/home/max/data/behavior/SPACEPRIME/results_July_01_2024_14_30_30.xlsx", index_col=0)
df.pop("Unnamed: 0")

# some cleaning
df = df[(df['event_type'] == 'response') & (df['rt'] != 0)]

pp_spatial = df[df["SpatialPriming"]==1]["iscorrect"]
np_spatial = df[df["SpatialPriming"]==-1]["iscorrect"]
# get mean values for priming effects
mean_correct_df = df.groupby(['subject_id', 'SpatialPriming'])['iscorrect'].mean().reset_index(name='mean_correct')
# Pivot to get priming types as columns
pivot_df_spat = mean_correct_df.pivot(index='subject_id',
                                      columns='SpatialPriming',
                                      values='mean_correct').fillna(0).reset_index()

cohen_d(pivot_df_spat[1], pivot_df_spat[-1])
