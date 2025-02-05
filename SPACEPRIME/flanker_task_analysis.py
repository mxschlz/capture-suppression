import pandas as pd
import seaborn as sns
from scipy.stats import ttest_rel
from stats import remove_outliers
import mne
import os
import glob
import numpy as np
from SPACEPRIME.plotting import plot_individual_lines
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.multitest import multipletests


# define data root dir
data_root = "/home/max/Insync/schulz.max5@gmail.com/GoogleDrive/PhD/data/SPACEPRIME/derivatives/preprocessing/"
# get all the subject ids
subjects = os.listdir(data_root)
# load data from children
df = pd.concat([pd.read_csv(glob.glob(f"/home/max/Insync/schulz.max5@gmail.com/GoogleDrive/PhD/data/SPACEPRIME/sourcedata/raw/{subject}/beh/flanker_data_{subject.split("-")[1]}*.csv")[0]) for subject in subjects if int(subject.split("-")[1]) in [103, 104, 105, 106, 107]])
# clean rt data
df = remove_outliers(df, column_name="rt", threshold=2)
# plot reaction time distribution
sns.displot(data=df["rt"])
# plot reaction time
plot = sns.barplot(data=df, x="congruency", y="rt")
plot_individual_lines(ax=plot, data=df, x_col="congruency", y_col="rt")
# plot performance accuracy
plot = sns.barplot(data=df, x="congruency", y="correct")
plot_individual_lines(ax=plot, data=df, x_col="congruency", y_col="correct")
# transform categories into integers for statistics
mapping = dict(congruent=1, incongruent=0, neutral=2)
df["congruency_int"] = df["congruency"].map(mapping)
# do t test
# ad hoc stats
# some stats on behavior
# Perform repeated measures ANOVA for 'correct'
anova_correct = AnovaRM(df, depvar='rt', subject='subject_id', within=['congruency'], aggregate_func="mean").fit()
print(anova_correct.summary())
# Perform paired t-tests
t_stat_12, p_value_12 = ttest_rel(df.query("congruency=='neutral'")["rt"], df.query("congruency=='congruent'")["rt"],
                                  nan_policy="omit")
t_stat_13, p_value_13 = ttest_rel(df.query("congruency=='neutral'")["rt"], df.query("congruency=='incongruent'")["rt"],
                                  nan_policy="omit")
t_stat_23, p_value_23 = ttest_rel(df.query("congruency=='incongruent'")["rt"], df.query("congruency=='congruent'")["rt"],
                                  nan_policy="omit")
# Combine p-values
p_values = [p_value_12, p_value_13, p_value_23]
# Bonferroni correction
reject, p_values_corrected, _, _ = multipletests(p_values, alpha=0.05, method='bonferroni')
print("Corrected p-values (Bonferroni):", p_values_corrected)
print("Reject null hypothesis:", reject)

# get all the congruent and incongruent epochs
epochs = mne.concatenate_epochs([mne.read_epochs(glob.glob(f"/home/max/Insync/schulz.max5@gmail.com/GoogleDrive/PhD/data/SPACEPRIME/derivatives/epoching/{subject}/eeg/{subject}_task-flanker-epo.fif")[0]) for subject in subjects if int(subject.split("-")[1]) in [105, 107]])
epochs.average().plot("Oz")
congruent = ["congruent_left", "congruent_right"]
incongruent = ["incongruent_left", "incongruent_right"]
congruent_epochs = epochs[congruent]
incongruent_epochs = epochs[incongruent]
# compute time frequency bins
# some params
freqs = np.arange(8, 13, 1)  # 1 to 30 Hz
n_cycles = freqs / 2  # different number of cycle per frequency
method = "morlet"  # wavelet
decim = 1  # keep all the samples along the time axis
power = epochs.compute_tfr(method=method, freqs=freqs, n_cycles=n_cycles, decim=decim, n_jobs=-1, return_itc=False,
                           average=True)
power.plot(baseline=(None, 0), combine="mean", mode="logratio")
incongruent_power = incongruent_epochs.compute_tfr(method=method, freqs=freqs, n_cycles=n_cycles, decim=decim,
                                                   n_jobs=-1, return_itc=False, average=True)
congruent_power = congruent_epochs.compute_tfr(method=method, freqs=freqs, n_cycles=n_cycles, decim=decim,
                                                   n_jobs=-1, return_itc=False, average=True)
power_diff = incongruent_power - congruent_power
power_diff.plot(combine="mean", mode="logratio")
