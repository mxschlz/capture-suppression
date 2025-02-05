import mne
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
plt.ion()


# define data root dir
data_root = "/home/max/Insync/schulz.max5@gmail.com/GoogleDrive/PhD/data/SPACEPRIME/derivatives/preprocessing/"
# get all the subject ids
subjects = os.listdir(data_root)
# load epochs
epochs = mne.concatenate_epochs([mne.read_epochs(glob.glob(f"/home/max/Insync/schulz.max5@gmail.com/GoogleDrive/PhD/data/SPACEPRIME/derivatives/epoching/{subject}/eeg/{subject}_task-spaceprime-epo.fif")[0]) for subject in subjects if int(subject.split("-")[1]) in [103, 104, 105, 106, 107]])
# epochs.apply_baseline()
all_conds = list(epochs.event_id.keys())
# Separate epochs based on target location
left_target_epochs = epochs[[x for x in all_conds if "Target-1-Singleton-2" in x]]
right_target_epochs = epochs[[x for x in all_conds if "Target-3-Singleton-2" in x]]
mne.epochs.equalize_epoch_counts([left_target_epochs, right_target_epochs], method="random")
# get left and right electrodes
#left_channels = mne.channels.make_1020_channel_selections(epochs.info, midline='z', return_ch_names=True)['Left']
#right_channels = mne.channels.make_1020_channel_selections(epochs.info, midline='z', return_ch_names=True)['Right']
# get the trial-wise data for targets
contra_target_epochs_data = np.mean(np.concatenate([left_target_epochs.copy().pick("C4").get_data(),
                                 right_target_epochs.copy().pick("C3").get_data()], axis=1), axis=1)
ipsi_target_epochs_data = np.mean(np.concatenate([left_target_epochs.copy().pick("C3").get_data(),
                               right_target_epochs.copy().pick("C4").get_data()], axis=1), axis=1)
# compute power density spectrum for evoked response and look for frequency tagging
# first, create epochs objects from numpy arrays computed above for ipsi and contra targets
ipsi_target_epochs = mne.EpochsArray(data=ipsi_target_epochs_data.reshape(len(left_target_epochs), 1, 376),
                                     info=mne.create_info(ch_names=["Ipsi Target"], sfreq=250, ch_types="eeg"))
contra_target_epochs = mne.EpochsArray(data=contra_target_epochs_data.reshape(len(left_target_epochs), 1, 376),
                                     info=mne.create_info(ch_names=["Contra Target"], sfreq=250, ch_types="eeg"))
# now, compute the power spectra for both ipsi and contra targets
target_diff = mne.EpochsArray(data=(contra_target_epochs.get_data() - ipsi_target_epochs.get_data()),
                              info=mne.create_info(ch_names=["Target: Contra - Ipsi"], sfreq=250, ch_types="eeg"))
# subtract the ipsi from the contralateral target power
powerdiff_target = target_diff.compute_psd(method="welch")
powerdiff_target.plot()
