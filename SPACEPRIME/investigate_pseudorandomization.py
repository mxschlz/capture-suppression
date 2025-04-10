import pandas as pd
import seaborn as sns
import glob
from SPACEPRIME import get_data_path
from SPACEPRIME.subjects import subject_ids


# load behavioral data because it contains information of trial sequences from all subjects
df = pd.concat([pd.read_csv(glob.glob(f"{get_data_path()}derivatives/preprocessing/sub-{subject}/beh/sub-{subject}_clean*.csv")[0]) for subject in subject_ids])
# create a total_trial_nr column that ranges from 1 to 1800
df['total_trial_nr'] = df.groupby('subject_id').cumcount()  # cumcount starts at 0
# drop nan values
df_to_plot = df[["Priming", "total_trial_nr", "block"]].dropna().reset_index(drop=True)
# stripplot
sns.stripplot(x="Priming", y="total_trial_nr", data=df_to_plot, hue="block")
