import mne
import numpy as np
import matplotlib.pyplot as plt
plt.ion()


subject_id = 102
# load epochs
epochs = mne.read_epochs(f"/home/max/Insync/schulz.max5@gmail.com/GoogleDrive/PhD/data/SPACEPRIME/derivatives/epoching/sub-{subject_id}/eeg/sub-{subject_id}_task-spaceprime-epo.fif",
                         preload=True).crop(0, 0.5)
# epochs.apply_baseline()
all_conds = list(epochs.event_id.keys())
# Separate epochs based on distractor location
left_singleton_epochs = epochs[[x for x in all_conds if "Target-2-Singleton-1" in x]]
right_singleton_epochs = epochs[[x for x in all_conds if "Target-2-Singleton-3" in x]]
mne.epochs.equalize_epoch_counts([left_singleton_epochs, right_singleton_epochs], method="random")
# get the contralateral evoked response and average
contra_singleton_data = np.mean([left_singleton_epochs.copy().average(picks=["C4"]).get_data(),
                                    right_singleton_epochs.copy().average(picks=["C3"]).get_data()], axis=0)
# now, do the same for the lateral targets
# Separate epochs based on target location
left_target_epochs = epochs[[x for x in all_conds if "Target-1-Singleton-2" in x]]
right_target_epochs = epochs[[x for x in all_conds if "Target-3-Singleton-2" in x]]
mne.epochs.equalize_epoch_counts([left_target_epochs, right_target_epochs], method="random")
# get the contralateral evoked response and average
contra_target_data = np.mean([left_target_epochs.copy().average(picks=["C4"]).get_data(),
                                 right_target_epochs.copy().average(picks=["C3"]).get_data()], axis=0)
# get the trial-wise data for controls
left_control_epochs = epochs[[x for x in all_conds if not "Target-1" in x and not "Singleton-1" in x and not "Singleton-0" in x]]
right_control_epochs = epochs[[x for x in all_conds if not "Target-3" in x and not "Singleton-3" in x and not "Singleton-0" in x]]
mne.epochs.equalize_epoch_counts([left_control_epochs, right_control_epochs], method="random")
# get the contralateral evoked response and average
contra_control_data = np.mean([left_control_epochs.copy().average(picks=["C4"]).get_data(),
                                 right_control_epochs.copy().average(picks=["C3"]).get_data()], axis=0)

times = epochs.average().times
# first plot
plt.plot(times, contra_target_data[0], color="forestgreen")
plt.plot(times, contra_singleton_data[0], color="red")
plt.plot(times, contra_control_data[0], color="grey")
plt.legend(["Target", "Distractor", "Control"])
plt.ylabel("Amplitude [µV]")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude [µV]")
plt.savefig("/home/max/Insync/schulz.max5@gmail.com/GoogleDrive/PhD/Conferences/DGPs24/Fig9.svg")