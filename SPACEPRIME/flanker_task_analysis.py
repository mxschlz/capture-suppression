import pandas as pd
import seaborn as sns
from scipy.stats import ttest_rel
from stats import remove_outliers
import mne
import glob


# define subject
subject_id = 103
# load dataframe
df = pd.read_csv(glob.glob(f"/home/max/Insync/schulz.max5@gmail.com/GoogleDrive/PhD/data/SPACEPRIME/sourcedata/raw/sub-{subject_id}/beh/flanker_data_{subject_id}*.csv")[0])
# clean rt data
df = remove_outliers(df, column_name="rt", threshold=2)
# plot reaction time distribution
sns.displot(data=df["rt"])
# plot reaction time
sns.barplot(data=df, x="congruency", y="rt")
# transform categories into integers for statistics
mapping = dict(congruent=1, incongruent=0, neutral=2)
df["congruency_int"] = df["congruency"].map(mapping)
# do dependent sample t test
ttest_rel(df[df["congruency_int"]==2]["rt"], df[df["congruency_int"]==0]["rt"], nan_policy="omit")
# analyse neural data
epochs = mne.read_epochs(f"/home/max/Insync/schulz.max5@gmail.com/GoogleDrive/PhD/data/SPACEPRIME/derivatives/epoching/sub-{subject_id}/eeg/sub-{subject_id}_task-flanker-epo.fif", preload=True)
epochs.average().plot(picks="POz")
