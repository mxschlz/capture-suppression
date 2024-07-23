import matplotlib
matplotlib.use("Qt5Agg")
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
plt.ion()


# load up dataframe
df = pd.read_excel("/home/max/data/behavior/SPACEPRIME/results_July_01_2024_14_30_30.xlsx", index_col=0)
df.pop("Unnamed: 0")

# some cleaning
df = df[(df['event_type'] == 'response') & (df['rt'] != 0)]
# get mean values for priming effects
mean_correct_df = df.groupby(['subject_id', 'SpatialPriming'])['iscorrect'].mean().reset_index(name='mean_correct')
# Pivot to get priming types as columns
pivot_df_spat = mean_correct_df.pivot(index='subject_id',
                                      columns='SpatialPriming',
                                      values='mean_correct').fillna(0).reset_index()
# same thing for identity priming
mean_correct_df = df.groupby(['subject_id', 'IdentityPriming'])['iscorrect'].mean().reset_index(name='mean_correct')
# Pivot to get priming types as columns
pivot_df_ident = mean_correct_df.pivot(index='subject_id',
                                       columns='IdentityPriming',
                                       values='mean_correct').fillna(0).reset_index()

diff_spat = pivot_df_spat[-1] - pivot_df_spat[1]
diff_ident = pivot_df_ident[-1] - pivot_df_ident[1]

pearsonr(diff_spat, diff_ident)

sns.regplot(x=diff_spat, y=diff_ident)