import mne
import numpy as np
import matplotlib.pyplot as plt
import glob
from scipy.signal import savgol_filter
from SPACEPRIME import get_data_path
from SPACEPRIME.subjects import subject_ids
plt.ion()


epochs = mne.concatenate_epochs([mne.read_epochs(glob.glob(f"{get_data_path()}derivatives/epoching/sub-{subject}/eeg/sub-{subject}_task-spaceprime-epo.fif")[0]) for subject in subject_ids])
# epochs = epochs["select_target==True"]
epochs.crop(0)  # crop for better visabillity
all_conds = list(epochs.event_id.keys())

# now, do N2ac analysis
# Separate epochs based on distractor location
left_singleton_epochs = epochs[[x for x in all_conds if "Target-2-Singleton-1" in x]]
right_singleton_epochs = epochs[[x for x in all_conds if "Target-2-Singleton-3" in x]]
mne.epochs.equalize_epoch_counts([left_singleton_epochs, right_singleton_epochs], method="random")
# get the contralateral evoked response and average
contra_singleton_data = np.mean([left_singleton_epochs.copy().average(picks="FC6").get_data(),
                                    right_singleton_epochs.copy().average(picks="FC5").get_data()], axis=0)
# get the ipsilateral evoked response and average
ipsi_singleton_data = np.mean([left_singleton_epochs.copy().average(picks="FC5").get_data(),
                                  right_singleton_epochs.copy().average(picks="FC6").get_data()], axis=0)

# now, do the same for the lateral targets
# Separate epochs based on target location
left_target_epochs = epochs[[x for x in all_conds if "Target-1-Singleton-2" in x]]
right_target_epochs = epochs[[x for x in all_conds if "Target-3-Singleton-2" in x]]
mne.epochs.equalize_epoch_counts([left_target_epochs, right_target_epochs], method="random")
# get the contralateral evoked response and average
contra_target_data = np.mean([left_target_epochs.copy().average(picks="FC6").get_data(),
                                 right_target_epochs.copy().average(picks=["FC5"]).get_data()], axis=0)
# get the ipsilateral evoked response and average
ipsi_target_data = np.mean([left_target_epochs.copy().average(picks="FC5").get_data(),
                               right_target_epochs.copy().average(picks="FC6").get_data()], axis=0)

# we can also isolate target effects without a distractor presence
left_target_epochs_distractor_absent = epochs[[x for x in all_conds if "Target-1-Singleton-0" in x]]
right_target_epochs_distractor_absent = epochs[[x for x in all_conds if "Target-3-Singleton-0" in x]]
mne.epochs.equalize_epoch_counts([left_target_epochs_distractor_absent, right_target_epochs_distractor_absent], method="random")
# get the contralateral evoked response and average
contra_target_distractor_absent_data = np.mean([left_target_epochs_distractor_absent.copy().average(picks="FC6").get_data(),
                                 right_target_epochs_distractor_absent.copy().average(picks="FC5").get_data()], axis=0)
# get the ipsilateral evoked response and average
ipsi_target_distractor_absent_data = np.mean([left_target_epochs_distractor_absent.copy().average(picks="FC5").get_data(),
                               right_target_epochs_distractor_absent.copy().average(picks="FC6").get_data()], axis=0)

# compute the difference waves (contra - ipsi) for lateral targets without and with distractor presence and
# lateral distractors
diff_wave_target_distractor_present = contra_target_data - ipsi_target_data
diff_wave_target_distractor_absent = contra_target_distractor_absent_data - ipsi_target_distractor_absent_data
diff_wave_distractor = contra_singleton_data - ipsi_singleton_data
# plot the data
window_length = 51
poly_order = 3
times = epochs.average().times
plt.plot(times, savgol_filter(diff_wave_distractor[0]*10e5, window_length=window_length, polyorder=poly_order), color="darkorange")
plt.plot(times, savgol_filter(diff_wave_target_distractor_present[0]*10e5, window_length=window_length, polyorder=poly_order), color="blue")
plt.plot(times, savgol_filter(diff_wave_target_distractor_absent[0]*10e5, window_length=window_length, polyorder=poly_order), color="green")
plt.hlines(y=0, xmin=times[0], xmax=times[-1])
plt.legend(["Distractor lateral", "Target lateral (Distractor present)", "Target lateral (Distractor absent)"])
plt.title("Difference Contra - Ipsi")
plt.ylabel("Amplitude [µV]")
plt.xlabel("Time [s]")
