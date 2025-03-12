import mne
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from SPACEPRIME import get_data_path
from SPACEPRIME.subjects import subject_ids
plt.ion()


# In this analysis, we want to see whether the brain synchronizes with the amplitude modulation of our target stimulus.
# In theory, neural entrainment should allow us to see a 30 Hz peak in the power spectrum. Additionally, we want to
# estimate how this frequency tagging is altered in our respective priming conditions (i.e., no priming, positive priming,
# negative priming).
# define data root dir
data_root = f"{get_data_path()}derivatives/preprocessing/"
# get all the subject ids
subjects = os.listdir(data_root)
# load epochs. Important consideration, only select the interval of stimuluis presentation (0 - 0.25 seconds).
epochs = mne.concatenate_epochs([mne.read_epochs(glob.glob(f"{get_data_path()}derivatives/epoching/sub-{subject}/eeg/sub-{subject}_task-spaceprime-epo.fif")[0], preload=True) for subject in subject_ids]).crop(0, 0.25)
# For maximum target entrainment, we want to exclusively look at epochs where the target has been identified correctly
# and where a salient distractor is absent.
epochs = epochs["select_target==True"]
# epochs.apply_baseline()
all_conds = list(epochs.event_id.keys())
# Separate epochs based on target location. Pick distractor absent trials only (Dont know whether that makes sense).
left_target_epochs = epochs[[x for x in all_conds if "Target-1-Singleton-0" in x]]
right_target_epochs = epochs[[x for x in all_conds if "Target-3-Singleton-0" in x]]
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
ipsi_target_epochs = mne.EpochsArray(data=ipsi_target_epochs_data.reshape(len(left_target_epochs), 1, left_target_epochs.get_data().shape[2]),
                                     info=mne.create_info(ch_names=["Ipsi Target"], sfreq=250, ch_types="eeg"))
contra_target_epochs = mne.EpochsArray(data=contra_target_epochs_data.reshape(len(left_target_epochs), 1, left_target_epochs.get_data().shape[2]),
                                     info=mne.create_info(ch_names=["Contra Target"], sfreq=250, ch_types="eeg"))
# now, compute the power spectra for both ipsi and contra targets
target_diff = mne.EpochsArray(data=(contra_target_epochs.get_data() - ipsi_target_epochs.get_data()),
                              info=mne.create_info(ch_names=["Target: Contra - Ipsi"], sfreq=250, ch_types="eeg"))
# subtract the ipsi from the contralateral target power
powerdiff_target = target_diff.compute_psd(method="welch", n_jobs=-1)
powerdiff_target.plot()
