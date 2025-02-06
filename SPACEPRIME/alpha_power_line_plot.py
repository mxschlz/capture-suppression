import mne
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
plt.ion()


# some params
freqs = np.arange(8, 13, 1)  # 1 to 30 Hz
n_cycles = freqs / 2  # different number of cycle per frequency
method = "morlet"  # wavelet
decim = 5  # keep all the samples along the time axis
# define data root dir
data_root = "G:\\Meine Ablage\\PhD\\data\\SPACEPRIME\\derivatives\\preprocessing"
# get all the subject ids
subjects = os.listdir(data_root)
# load epochs
epochs = [mne.read_epochs(glob.glob(f"G:\\Meine Ablage\\PhD\\data\\SPACEPRIME\\derivatives\\epoching/{subject}/eeg/{subject}_task-spaceprime-epo.fif")[0]) for subject in subjects if int(subject.split("-")[1]) in [103, 104, 105, 106, 107]]
for subject_data in epochs:
    power = subject_data.compute_tfr(method=method, freqs=freqs, n_cycles=n_cycles, decim=decim, n_jobs=-1, return_itc=False,
                           average=False).average()
    alpha_time_series = power.get_data().mean(axis=0)
    alpha_time_series = alpha_time_series.mean(axis=0)
