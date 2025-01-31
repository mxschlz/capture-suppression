import mne
import matplotlib.pyplot as plt
import os
import glob
plt.ion()


# define data root dir
data_root = "/home/max/Insync/schulz.max5@gmail.com/GoogleDrive/PhD/data/SPACEPRIME/derivatives/preprocessing/"
# get all the subject ids
subjects = os.listdir(data_root)
# load epochs
epochs = mne.concatenate_epochs([mne.read_epochs(glob.glob(f"/home/max/Insync/schulz.max5@gmail.com/GoogleDrive/PhD/data/SPACEPRIME/derivatives/epoching/{subject}/eeg/{subject}_task-passive-epo.fif")[0]) for subject in subjects if int(subject.split("-")[1]) in [106, 107]])

