import mne
import numpy as np
import matplotlib.pyplot as plt
plt.ion()


subject_id = 102
if subject_id == 101:
    # load epochs
    epochs = mne.read_epochs(
        f"/home/max/Insync/schulz.max5@gmail.com/GoogleDrive/PhD/data/SPACEPRIME/derivatives/epoching/sub-{subject_id}/eeg/sub-{subject_id}_task-spaceprime-epo.fif",
        preload=True)
elif subject_id == 102:
    # load epochs
    epochs = mne.read_epochs(
        f"/home/max/Insync/schulz.max5@gmail.com/GoogleDrive/PhD/data/SPACEPRIME/derivatives/epoching/sub-{subject_id}/eeg/sub-{subject_id}_task-spaceprime-epo.fif",
        preload=True)
epochs.apply_baseline()
if subject_id == 101:
    epochs.crop(0.35, 0.85)
elif subject_id == 102:
    epochs.crop(0, 0.5)
all_conds = list(epochs.event_id.keys())
# Separate epochs based on distractor location
left_singleton_epochs = epochs[[x for x in all_conds if "Target-2-Singleton-1" in x]]
right_singleton_epochs = epochs[[x for x in all_conds if "Target-2-Singleton-3" in x]]
mne.epochs.equalize_epoch_counts([left_singleton_epochs, right_singleton_epochs], method="random")
# get the contralateral evoked response and average
contra_singleton_data = np.mean([left_singleton_epochs.copy().average(picks=["C4"]).get_data(),
                                    right_singleton_epochs.copy().average(picks=["C3"]).get_data()], axis=0)
# get the ipsilateral evoked response and average
ipsi_singleton_data = np.mean([left_singleton_epochs.copy().average(picks=["C3"]).get_data(),
                                  right_singleton_epochs.copy().average(picks=["C4"]).get_data()], axis=0)
# now, do the same for the lateral targets
# Separate epochs based on target location
left_target_epochs = epochs[[x for x in all_conds if "Target-1-Singleton-2" in x]]
right_target_epochs = epochs[[x for x in all_conds if "Target-3-Singleton-2" in x]]
mne.epochs.equalize_epoch_counts([left_target_epochs, right_target_epochs], method="random")
# get the contralateral evoked response and average
contra_target_data = np.mean([left_target_epochs.copy().average(picks=["C4"]).get_data(),
                                 right_target_epochs.copy().average(picks=["C3"]).get_data()], axis=0)
# get the ipsilateral evoked response and average
ipsi_target_data = np.mean([left_target_epochs.copy().average(picks=["C3"]).get_data(),
                               right_target_epochs.copy().average(picks=["C4"]).get_data()], axis=0)
# get the trial-wise data for controls
left_control_epochs = epochs[[x for x in all_conds if not "Target-1" in x and not "Singleton-1" in x and not "Singleton-0" in x]]
right_control_epochs = epochs[[x for x in all_conds if not "Target-3" in x and not "Singleton-3" in x and not "Singleton-0" in x]]
mne.epochs.equalize_epoch_counts([left_control_epochs, right_control_epochs], method="random")
# get the contralateral evoked response and average
contra_control_data = np.mean([left_control_epochs.copy().average(picks=["C4"]).get_data(),
                                 right_control_epochs.copy().average(picks=["C3"]).get_data()], axis=0)
ipsi_control_data = np.mean([left_control_epochs.copy().average(picks=["C3"]).get_data(),
                                 right_control_epochs.copy().average(picks=["C4"]).get_data()], axis=0)
diff_singleton = contra_singleton_data - ipsi_singleton_data
diff_target = contra_target_data - ipsi_target_data
diff_control = contra_control_data - ipsi_control_data
times = epochs.average().times
# first plot
fig, ax = plt.subplots(2, 1, sharex=True)
ax[0].plot(times, contra_target_data[0], color="forestgreen")
ax[0].plot(times, contra_singleton_data[0], color="red")
ax[0].plot(times, contra_control_data[0], color="grey")
ax[1].plot(times, diff_target[0], color="forestgreen")
ax[1].plot(times, diff_singleton[0], color="red")
ax[1].plot(times, diff_control[0], color="grey")
plt.legend(["Target", "Distractor", "Control"])
plt.ylabel("Amplitude [µV]")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude [µV]")
#plt.savefig("/home/max/Insync/schulz.max5@gmail.com/GoogleDrive/PhD/Conferences/DGPs24/Fig9.svg")
