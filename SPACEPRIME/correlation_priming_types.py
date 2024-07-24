import matplotlib
matplotlib.use("Qt5Agg")
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
plt.ion()
from bootstrap import generate_simulated_subjects


# load up dataframe
df = pd.read_excel("/home/max/data/behavior/SPACEPRIME/results_July_06_2024_14_16_40.xlsx", index_col=0)
df.pop("Unnamed: 0")

# some cleaning
df = df[(df['event_type'] == 'response') & (df['rt'] != 0)]
df_bs = generate_simulated_subjects(df, num_new_subjects=10000, max_subject_id=99)
# get mean values for priming effects
mean_correct_df = df_bs.groupby(['subject_id', 'SpatialPriming'])['iscorrect'].mean().reset_index(name='mean_correct')
# Pivot to get priming types as columns
pivot_df_spat = mean_correct_df.pivot(index='subject_id',
                                      columns='SpatialPriming',
                                      values='mean_correct').fillna(0).reset_index()
# same thing for identity priming
mean_correct_df = df_bs.groupby(['subject_id', 'IdentityPriming'])['iscorrect'].mean().reset_index(name='mean_correct')
# Pivot to get priming types as columns
pivot_df_ident = mean_correct_df.pivot(index='subject_id',
                                       columns='IdentityPriming',
                                       values='mean_correct').fillna(0).reset_index()

diff_spat = pivot_df_spat[-1] - pivot_df_spat[1]
diff_ident = pivot_df_ident[-1] - pivot_df_ident[1]

pearsonr(diff_spat, diff_ident)

sns.regplot(x=diff_spat, y=diff_ident)