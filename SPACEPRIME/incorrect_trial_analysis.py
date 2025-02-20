import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import glob
from SPACEPRIME import get_data_path
from SPACEPRIME.subjects import subject_ids
plt.ion()


# load data from children
df = pd.concat([pd.read_csv(glob.glob(f"{get_data_path()}derivatives/preprocessing/sub-{subject}/beh/sub-{subject}_clean*.csv")[0]) for subject in subject_ids])
#df = df[df["SingletonPresent"] == True]
# Reshape the data
df_long = pd.melt(df, id_vars=['block', "subject_id"], value_vars=['select_target', 'select_distractor', 'select_control', 'select_other'])
sns.barplot(x='block', y='value', hue="variable", data=df_long, errorbar=("se", 1),
            palette=['forestgreen', 'red', 'grey', 'purple'])


# get only the incorrect trials
incorrects = df[df["select_target"] == False]
# plot a heatmap to see which response is given in incorrect trials
crosstab = pd.crosstab(index=incorrects["response"], columns=incorrects["TargetDigit"])
heatmap = sns.heatmap(crosstab)
heatmap.invert_yaxis()

sns.barplot(x='block', y='value', hue="variable", data=df_long.query("subject_id==116"), errorbar=None,
            palette=['forestgreen', 'red', 'grey', 'purple'])
