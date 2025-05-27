import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import glob
from SPACECUE import get_data_path

from stats import remove_outliers
plt.ion()


# load df
subject_ids = [1, 13, 25, 37, 99]
dfs = list()
for subject in subject_ids:
    if subject < 10:
        subject = f"0{subject}"
    dfs.append(pd.read_csv(glob.glob(f"{get_data_path()}derivatives/preprocessing/sub-{subject}/beh/sub-{subject}_clean*.csv")[0]))
df = pd.concat(dfs)

df = remove_outliers(df, column_name="rt", threshold=2)
df_mean = df.groupby(["subject_id", "CueInstruction"])[["select_target", "rt"]].mean().reset_index()

sns.barplot(data=df_mean, x="CueInstruction", y="rt")
sns.barplot(data=df_mean, x="CueInstruction", y="select_target")

sns.barplot(data=df_mean, x="CueInstruction", y="select_target", hue="subject_id", errorbar=None)

sns.barplot(data=df_mean, x="Priming", y="rt")
sns.barplot(data=df_mean, x="Priming", y="select_target")
