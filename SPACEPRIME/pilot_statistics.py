import matplotlib
matplotlib.use("Qt5Agg")
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from stats import cohen_d
plt.ion()


# load up dataframe
df = pd.read_excel("/home/max/data/behavior/SPACEPRIME/results_July_06_2024_14_16_40.xlsx", index_col=0)
df.pop("Unnamed: 0")

# some cleaning
df = df[(df['event_type'] == 'response') & (df['rt'] != 0)]
# Convert `iscorrect` to numeric (True -> 1, False -> 0)
df['iscorrect'] = df['iscorrect'].astype(int)
# compute mean so that icorrect variable is continuous
mean_correct_df = df.groupby(['subject_id', 'SpatialPriming'])['iscorrect'].mean().reset_index()
# Fit the linear mixed model
md = smf.mixedlm("iscorrect ~ SpatialPriming",
                 df,
                 groups=df["subject_id"])
mdf = md.fit()

# Print the model summary
print(mdf.summary())