import mne
import os
import numpy as np


# define root data path
data_path = "/home/max/Insync/schulz.max5@gmail.com/GoogleDrive/PhD/hannah_data/eeg/derivatives/"
# retrieve subjects from data path structure
subjects = sorted(os.listdir(data_path))
# instantiate epoch list to store single subject epochs in
alpha_power_subjects = []
# define some TFR params
alpha_freqs = np.arange(7, 14, 1)
n_cycles = alpha_freqs / 2
method = "morlet"
decim = 15
all_epochs = list()
weird_subjects = list()
# iterate over subjects
for subject in subjects:
    subject_data = os.path.join(data_path, subject, "epoching")
    # try reading epochs (cropped)
    try:
        epochs = mne.read_epochs(os.path.join(subject_data, os.listdir(subject_data)[0]), preload=True).crop(-1, 1)
        # append to all epochs list
        all_epochs.append(epochs)
        # compute alpha power
        alpha_power = epochs.compute_tfr(method=method, freqs=alpha_freqs, n_cycles=n_cycles, decim=decim, n_jobs=-1, return_itc=False,
                                   average=False)  # no average to compute absolute alpha power (not evoked)
        # average over epochs
        absolute_alpha = alpha_power.average()
        # now, average over whole frequency band and channels
        averaged_power = absolute_alpha.get_data().mean(axis=0)
        # append epochs to list
        alpha_power_subjects.append(alpha_power)
    except:
        weird_subjects.append(subject)


all_epochs = list()
# iterate over subjects
for subject in subjects:
    if subject in ["SP_EEG_P0015", "SP_EEG_P0020", "SP_EEG_P0058"]:
        continue
    subject_data = os.path.join(data_path, subject, "epoching")
    # try reading epochs (cropped)
    try:
        epochs = mne.read_epochs(os.path.join(subject_data, os.listdir(subject_data)[0]), preload=False)
        all_epochs.append(epochs)
    except ValueError as e:
        print(e)
        weird_subjects.append(subject)

# compare alpha power between schizo and hc
epochs = mne.concatenate_epochs(all_epochs, on_mismatch="ignore").crop(-0.6, 0)
epochs_vis = epochs["Stimulus/S 20"]
epochs_aud = epochs["Stimulus/S 80"]
epochs.metadata['is_hc'] = epochs.metadata["subject_id"].astype(int) < 51

epochs_hc_vis = epochs["is_hc==True"]
epochs_sc_vis = epochs["is_hc==False"]
# get alpha for hc and sc
alpha_hc = epochs_hc_vis.compute_tfr(method=method, freqs=alpha_freqs, n_cycles=n_cycles, decim=decim, n_jobs=-1, return_itc=False,
                                   average=False)  # no average to compute absolute alpha power (not evoked)
alpha_sc = epochs_sc_vis.compute_tfr(method=method, freqs=alpha_freqs, n_cycles=n_cycles, decim=decim, n_jobs=-1, return_itc=False,
                                   average=False)  # no average to compute absolute alpha power (not evoked)
