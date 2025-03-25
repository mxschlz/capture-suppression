import mne
import matplotlib.pyplot as plt
import glob
from SPACEPRIME.subjects import subject_ids
from SPACEPRIME import get_data_path
import pandas as pd
import statsmodels.formula.api as smf
import seaborn as sns
plt.ion()


# In this script, we are going to load up some eeg together with behavioral data. Combined, these data will be the basis
# for a couple of linear mixed models. Overall, we want to aim for a mixed model which contains all possible relevant
# predictors and dependent variables. Let's go.
# First, we load up the EEG data, which already includes the behavior as metadata attribute.
epochs = mne.concatenate_epochs([mne.read_epochs(glob.glob(f"{get_data_path()}derivatives/epoching/sub-{subject}/eeg/sub-{subject}_task-spaceprime-epo.fif")[0], preload=False) for subject in subject_ids])
# crop to save RAM
epochs.crop(0, 0.7)
# We want to calculate some average values for N2ac and Pd, therefore, we have to do some data transformation magic.
# First, we save the dataframe in a separate variable to facilitate modification.
df = epochs.metadata
# Now, we add a column "total_trial_nr" which starts from 1 and goes to 1800 for each subject.
df['total_trial_nr'] = df.groupby('subject_id').cumcount() + 1
# Now, the only parameters we need to include in the dataframe are mean values for N2ac and Pd. In theory, I think it
# makes sense to calculate a Pd and N2ac component for every trial. In fact, these components should only be evident in
# trials where the distractor or target is presented laterally, and these predictor variables should explain the component
# amplitude reliably. Logically, Pd should be NaN in trials where a singleton is absent, whereas we can compute a N2ac for
# all trials because targets are always present.
# Before we retrieve the signature data, we need to define some parameters of interest
Pd_window = (0.25, 0.35)  # Pd time interval
Pd_elecs = ["C3", "C4"]  # electrodes of interest
N2ac_window = (0.2, 0.3)  # N2ac time interval
N2ac_elecs = ["FC5", "FC6"]  # electrodes of interest
# We want to convert the epochs data into a dataframe which we then append to the above dataframe.
Pd_df = epochs.to_data_frame(picks=Pd_elecs, index=None, copy=True, long_format=False, time_format=None)
# We filter the dataframe in the time intervals of interest
filtered_Pd_df = Pd_df[(Pd_df['time'] >= Pd_window[0]) & (Pd_df['time'] <= Pd_window[1])]
# Now, we group by epoch number and calculate the mean amplitude at respective electrode pairs.
grouped_Pd_df = filtered_Pd_df.groupby(['epoch', "condition"])[Pd_elecs].mean().reset_index()
# We repeat the steps from above for the N2ac, respectively.
N2ac_df = epochs.to_data_frame(picks=N2ac_elecs, index=None, copy=True, long_format=False, time_format=None)
# We filter the dataframe in the time intervals of interest
filtered_N2ac_df = N2ac_df[(N2ac_df['time'] >= N2ac_window[0]) & (N2ac_df['time'] <= N2ac_window[1])]
# Now, we group by epoch number and calculate the mean amplitude at respective electrode pairs.
grouped_N2ac_df = filtered_N2ac_df.groupby(['epoch', "condition"])[N2ac_elecs].mean().reset_index()
# We merge the two dataframes in order to effectively iterate over them.
merged_diff_wave_df = pd.merge(grouped_Pd_df, grouped_N2ac_df, on=["epoch", "condition"])
# I think it is beste if we iterate over all epochs in our dataset and look at the respective condition. Consecutively,
# we extract the data from the Pd/N2ac dataframes and and append the difference wave mean amplitude.
# Instantiate columns to store Pd and N2ac amplitudes.
df["N2ac_mean_amplitude"] = None
df["Pd_mean_amplitude"] = None
# Iterate over one of the created dataframes
for i, trial in merged_diff_wave_df.iterrows():
    # retrieve the condition for the current trial
    cond = trial.condition
    # check for lateralized targets and subtract contra - ipsi
    if "Target-1" in cond and "Singleton-2" in cond:
        # Now, determine what electrode is contra and ipsi
        df["N2ac_mean_amplitude"].iloc[i] = trial.FC6 - trial.FC5
        df["Pd_mean_amplitude"].iloc[i] = None
    elif "Target-3" in cond and "Singleton-2" in cond:
        df["N2ac_mean_amplitude"].iloc[i] = trial.FC5 - trial.FC6
        df["Pd_mean_amplitude"].iloc[i] = None
    # check for lateralized distractors and subtract contra - ipsi
    elif "Singleton-1" in cond and "Target-2" in cond:
        df["Pd_mean_amplitude"].iloc[i] = trial.C4 - trial.C3
        df["N2ac_mean_amplitude"].iloc[i] = None
    elif "Singleton-3" in cond and "Target-2" in cond:
        df["Pd_mean_amplitude"].iloc[i] = trial.C3 - trial.C4
        df["N2ac_mean_amplitude"].iloc[i] = None
    # insert None in case none of the above conditions are satisfied
    else:
        df["N2ac_mean_amplitude"].iloc[i] = None
        df["Pd_mean_amplitude"].iloc[i] = None
# Now that we have all the variables we need, we can do some statistical modeling.
# First, we clean up the dataframe
df.drop("duration", axis=1, inplace=True)  # drop duration because it is always NaN
formula = "Pd_mean_amplitude ~ SingletonLoc"
# some formatting
pd_df = df[["Pd_mean_amplitude", "SingletonLoc", "subject_id"]].dropna(subset=["Pd_mean_amplitude", "SingletonLoc"],
                                                                       ignore_index=True).astype(float)
model = smf.mixedlm(formula=formula, data=pd_df, groups="subject_id")
result = model.fit()
result.summary()
sns.lmplot(data=pd_df, x="SingletonLoc", y="Pd_mean_amplitude", hue="subject_id")
# Modeling N2ac
formula = "N2ac_mean_amplitude ~ TargetLoc"
# some formatting
n2ac_df = df[["N2ac_mean_amplitude", "TargetLoc", "subject_id"]].dropna(subset=["N2ac_mean_amplitude", "TargetLoc"],
                                                                       ignore_index=True).astype(float)
model = smf.mixedlm(formula=formula, data=n2ac_df, groups="subject_id")
result = model.fit()
result.summary()