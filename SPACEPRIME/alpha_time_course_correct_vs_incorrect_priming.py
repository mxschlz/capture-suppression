import mne
import matplotlib.pyplot as plt
import glob
from SPACEPRIME.subjects import subject_ids
from SPACEPRIME import get_data_path
import numpy as np
import seaborn as sns
plt.ion()


# load up the epochs from all subjects
epochs = mne.concatenate_epochs([mne.read_epochs(glob.glob(f"{get_data_path()}derivatives/epoching/sub-{subject}/eeg/sub-{subject}_task-spaceprime-epo.fif")[0], preload=False) for subject in subject_ids])
# First, define ROI electrode picks to reduce computation time.
left_roi = ["TP9", "TP7", "CP5", "CP3", "CP1", "P7", "P5", "P3", "P1", "PO7", "PO3", "O1"]
right_roi = ["TP10", "TP8", "CP6", "CP4", "CP2", "P8", "P6", "P4", "P2", "PO8", "PO4", "O2"]
epochs = epochs.pick(picks=left_roi+right_roi)
# define some TFR params
# define alpha frequency range
alpha_freqs = np.arange(8, 13, 1)
n_cycles = alpha_freqs / 2  # different number of cycle per frequency
#window_length = 0.5  # window lengths as in Wöstmann et al. (2019)
method = "morlet"  # wavelet
decim = 5  # keep all the samples along the time axis
mode = "mean"
baseline = (None, None)
# compute the absolute oscillatory power for all subjects
alpha_power = epochs.compute_tfr(method=method, freqs=alpha_freqs, n_cycles=n_cycles,
                           decim=decim, n_jobs=-1, return_itc=False, average=False)
# apply baseline to transform the y-axis into dB values
alpha_power.apply_baseline(baseline=baseline, mode=mode)
# store the within-subject alpha power
subjects_alpha = dict(np=dict(),
                      pp=dict(),
                      no_p=dict())
# iterate over subjects
# --- DO OVERALL ALPHA POWER FOR ALL SUBJECTS ---
for subject in subject_ids:
    print(f"Processing subject {subject}")
    # get TFR from single subject NEGATIVE PRIMING
    power_sub_np = alpha_power[f"subject_id=={subject}&Priming==-1"]
    subjects_alpha["np"][f"{subject}"] = power_sub_np.average().get_data().mean(axis=(0, 1))  # get alpha power frequencies over all channels
    # get TFR from single subject POSITIVE PRIMING
    power_sub_pp = alpha_power[f"subject_id=={subject}&Priming==1"]
    subjects_alpha["pp"][f"{subject}"] = power_sub_pp.average().get_data().mean(axis=(0, 1))  # get alpha power frequencies over all channels
    # get TFR from single subject NO PRIMING
    power_sub_no_p = alpha_power[f"subject_id=={subject}&Priming==0"]
    subjects_alpha["no_p"][f"{subject}"] = power_sub_no_p.average().get_data().mean(axis=(0, 1))  # get alpha power frequencies over all channels
# get the time vector for plotting purposes
times = np.linspace(alpha_power.tmin, alpha_power.tmax, alpha_power.get_data().shape[-1])

# Now, plot the single subject and grand average data on one canvas
# plot data
fig = plt.figure()
for k, d in subjects_alpha.items():
    for sub, alpha_time_series in d.items():
        if k == "np":
            color = "darkred"
        elif k == "pp":
            color = "darkgreen"
        elif k == "no_p":
            color = "grey"
        else:
            raise ValueError(f"Unknown key {k}")
        plt.plot(times, alpha_time_series, label=f"Priming: {k}, Subject {k}", alpha=0.2, color=color)

# get average of subjects alpha
grand_average_np = np.mean(list(subjects_alpha["np"].values()), axis=0)
grand_average_pp = np.mean(list(subjects_alpha["pp"].values()), axis=0)
grand_average_no_p = np.mean(list(subjects_alpha["no_p"].values()), axis=0)

# plot grand average in black
plt.plot(times, grand_average_np, label="Grand Average negative priming", color="darkred", linewidth=5)
plt.plot(times, grand_average_pp, label="Grand Average positive priming", color="darkgreen", linewidth=5)
plt.plot(times, grand_average_no_p, label="Grand Average no priming", color="grey", linewidth=5)

plt.xlabel("Time (s)")
plt.ylabel("Alpha Power")
plt.title("Alpha Time Course")
sns.despine(fig=fig)

# plot topography of alpha
alpha_power.average().plot_topomap(tmin=0, tmax=0.75)
# plot spectrogram of alpha
alpha_power.average().plot(combine="mean")

# --- NEW ANALYSIS PART ---
# Now, compute alpha time course for the difference between correct and incorrect responses too look for changes in
# priming conditions
# compute alpha difference between correct versus incorrect trials
power_corrects = alpha_power["select_target==True"]
power_incorrects = alpha_power["select_target==False"]
# Further, we balance the amount of trials by removing trials randomly from the larger portion.
mne.epochs.equalize_epoch_counts([power_incorrects, power_corrects], method="random")
alpha_diff = power_corrects - power_incorrects
subjects_diff = dict(np=dict(),
                     pp=dict(),
                     no_p=dict())
# iterate over subjects
# --- DO OVERALL ALPHA POWER FOR ALL SUBJECTS ---
for subject in subject_ids:
    print(f"Processing subject {subject}")
    # get TFR from single subject NEGATIVE PRIMING
    power_sub_np = alpha_diff[f"subject_id=={subject}&Priming==-1"]
    subjects_diff["np"][f"{subject}"] = power_sub_np.average().get_data().mean(axis=(0, 1))  # get alpha power frequencies over all channels
    # get TFR from single subject POSITIVE PRIMING
    power_sub_pp = alpha_diff[f"subject_id=={subject}&Priming==1"]
    subjects_diff["pp"][f"{subject}"] = power_sub_pp.average().get_data().mean(axis=(0, 1))  # get alpha power frequencies over all channels
    # get TFR from single subject NO PRIMING
    power_sub_no_p = alpha_diff[f"subject_id=={subject}&Priming==0"]
    subjects_diff["no_p"][f"{subject}"] = power_sub_no_p.average().get_data().mean(axis=(0, 1))  # get alpha power frequencies over all channels
# get the time vector for plotting purposes
times = np.linspace(alpha_diff.tmin, alpha_diff.tmax, alpha_diff.get_data().shape[-1])

# Now, plot the single subject and grand average data on one canvas
# plot data
fig = plt.figure()
for k, d in subjects_diff.items():
    for sub, diff_time_series in d.items():
        if k == "np":
            color = "darkred"
        elif k == "pp":
            color = "darkgreen"
        elif k == "no_p":
            color = "grey"
        else:
            raise ValueError(f"Unknown key {k}")
        plt.plot(times, diff_time_series, label=f"Priming: {k}, Subject {k}", alpha=0.2, color=color)

# get average of subjects alpha
grand_average_diff_np = np.mean(list(subjects_diff["np"].values()), axis=0)
grand_average_diff_pp = np.mean(list(subjects_diff["pp"].values()), axis=0)
grand_average_diff_no_p = np.mean(list(subjects_diff["no_p"].values()), axis=0)

# plot grand average in black
plt.plot(times, grand_average_diff_np, label="Grand Average difference (correct-incorrect) negative priming", color="darkred", linewidth=5)
plt.plot(times, grand_average_diff_pp, label="Grand Average difference (correct-incorrect) positive priming", color="darkgreen", linewidth=5)
plt.plot(times, grand_average_diff_no_p, label="Grand Average difference (correct-incorrect) no priming", color="grey", linewidth=5)

plt.xlabel("Time (s)")
plt.ylabel("Alpha Power")
plt.title("Alpha Time Course")
sns.despine(fig=fig)
