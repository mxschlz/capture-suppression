import mne
import matplotlib.pyplot as plt
import glob
from SPACEPRIME.subjects import subject_ids
from SPACEPRIME import get_data_path
import numpy as np
plt.ion()


# load up the epochs from all subjects
epochs = mne.concatenate_epochs([mne.read_epochs(glob.glob(f"{get_data_path()}derivatives/epoching/sub-{subject}/eeg/sub-{subject}_task-spaceprime-epo.fif")[0], preload=False) for subject in subject_ids])
# First, define ROI electrode picks to reduce computation time.
# left_roi = ["TP9", "TP7", "CP5", "CP3", "CP1", "P7", "P5", "P3", "P1", "PO7", "PO3", "O1"]
# right_roi = ["TP10", "TP8", "CP6", "CP4", "CP2", "P8", "P6", "P4", "P2", "PO8", "PO4", "O2"]
# epochs = epochs.pick(picks=left_roi+right_roi)
# define some TFR params
# define alpha frequency range
alpha_freqs = np.arange(7, 14, 1)
n_cycles = alpha_freqs / 2  # different number of cycle per frequency
#window_length = 0.5  # window lengths as in Wöstmann et al. (2019)
method = "morlet"  # wavelet
decim = 7  # keep all the samples along the time axis

# compute the absolute oscillatory power for all subjects
alpha_power = epochs.compute_tfr(method=method, freqs=alpha_freqs, n_cycles=n_cycles,
                           decim=decim, n_jobs=-1, return_itc=False, average=False)
# apply baseline to transform the y-axis into dB values
# alpha_power.apply_baseline((-0.9, -0.5), mode="logratio")
# store the within-subject alpha power
subjects_alpha = dict()

# iterate over subjects
# --- DO OVERALL ALPHA POWER FOR ALL SUBJECTS ---
for subject in subject_ids:
    # get TFR from single subject
    power_sub = alpha_power[f"subject_id=={subject}"]
    power_avg = power_sub.average()
    alpha_freqs_times = power_avg.get_data().mean(axis=0)  # get alpha power frequencies over all channels
    subjects_alpha[f"{subject}"] = alpha_freqs_times.mean(axis=0)  # average over all frequencies

# get the time vector for plotting purposes
times = np.linspace(alpha_power.tmin, alpha_power.tmax, alpha_power.get_data().shape[-1])

# Now, plot the single subject and grand average data on one canvas
# plot data
plt.figure()
for k, v in subjects_alpha.items():
    plt.plot(times, v, label=f"Subject {k}", alpha=0.2)

# get average of subjects alpha
grand_average = np.mean(list(subjects_alpha.values()), axis=0)

# plot grand average in black
plt.plot(times, grand_average, label="Grand Average", color="black", linewidth=5)

plt.xlabel("Time (s)")
plt.ylabel("Alpha Power")
plt.title("Alpha Time Course")
#plt.legend()

# plot topography of alpha
alpha_power.average().plot_topomap(tmin=0, tmax=0.75)

# plot spectrogram of alpha
alpha_power.average().plot(combine="mean")