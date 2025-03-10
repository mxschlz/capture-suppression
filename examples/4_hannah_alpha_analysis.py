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
decim = 10
all_epochs = list()
weird_subjects = list()
# iterate over subjects
for subject in subjects:
    subject_data = os.path.join(data_path, subject, "epoching")
    # try reading epochs (cropped)
    try:
        epochs = mne.read_epochs(os.path.join(subject_data, os.listdir(subject_data)[0]), preload=True).crop(-0.7, 0.7)
        # append to all epochs list
        all_epochs.append(epochs)
        # get visual events only
        vis_epochs = epochs["Stimulus/S 20"]
        # compute alpha power
        alpha_power = vis_epochs.compute_tfr(method=method, freqs=alpha_freqs, n_cycles=n_cycles, decim=decim, n_jobs=-1, return_itc=False,
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
    subject_data = os.path.join(data_path, subject, "epoching")
    # try reading epochs (cropped)
    try:
        epochs = mne.read_epochs(os.path.join(subject_data, os.listdir(subject_data)[0]), preload=False)
        epochs.metadata = None
        epochs_vis = epochs["20"]
        # append to all epochs list
        all_epochs.append(epochs_vis)
    except:
        weird_subjects.append(subject)

# compare alpha power between schizo and hc
epochs = mne.concatenate_epochs(all_epochs)
epochs_hc = epochs["is_hc==True"]
epochs_sc = epochs["is_hc==False"]
