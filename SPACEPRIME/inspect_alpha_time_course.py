import mne
import matplotlib.pyplot as plt
import glob
from SPACEPRIME.subjects import subject_ids
from SPACEPRIME import get_data_path
import numpy as np
plt.ion()


# load up the epochs from all subjects
epochs = mne.concatenate_epochs([mne.read_epochs(glob.glob(f"{get_data_path()}derivatives/epoching/sub-{subject}/eeg/sub-{subject}_task-spaceprime-epo.fif")[0], preload=True) for subject in subject_ids])

# define some TFR params
# define alpha frequency range
alpha_freqs = np.arange(7, 14, 1)
n_cycles = alpha_freqs / 2  # different number of cycle per frequency
#window_length = 0.5  # window lengths as in Wöstmann et al. (2019)
method = "morlet"  # wavelet
decim = 1  # keep all the samples along the time axis

# store the within-subject alpha power
subjects_alpha = dict()

# iterate over subjects
# --- DO OVERALL ALPHA POWER FOR ALL SUBJECTS ---
for subject in subject_ids:
    # get TFR from single subject
    power = epochs[f"subject_id=={subject}"].compute_tfr(method=method, freqs=alpha_freqs, n_cycles=n_cycles,
                                                         decim=decim, n_jobs=-1, return_itc=False, average=False)
    # power.apply_baseline((-0.5, 0), mode="logratio", verbose=False)
    power_avg = power.average()
    alpha_freqs_times = power_avg.get_data().mean(axis=0)  # get alpha power frequencies over all channels
    subjects_alpha[f"{subject}"] = alpha_freqs_times.mean(axis=0)  # average over all frequencies

# get the time vector for plotting purposes
times = np.linspace(epochs.tmin, epochs.tmax, epochs.get_data().shape[2])

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
plt.legend()

# plot topography of alpha
total_alpha = epochs.compute_tfr(method=method, freqs=alpha_freqs, n_cycles=n_cycles, decim=10, average=False,
                                 n_jobs=-1, return_itc=False)
total_alpha_avrg = total_alpha.average()
total_alpha_avrg.plot_topomap(tmin=0, tmax=0.75)
