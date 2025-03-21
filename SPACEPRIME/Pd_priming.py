import mne
import matplotlib.pyplot as plt
import os
import glob
from SPACEPRIME.subjects import subject_ids
from SPACEPRIME import get_data_path
import numpy as np
from scipy.signal import savgol_filter
plt.ion()


# define data root dir
data_root = f"{get_data_path()}derivatives/preprocessing/"
# get all the subject ids
subjects = os.listdir(data_root)
# load epochs
epochs = mne.concatenate_epochs([mne.read_epochs(glob.glob(f"{get_data_path()}derivatives/epoching/{subject}/eeg/{subject}_task-spaceprime-epo.fif")[0], preload=False) for subject in subjects if int(subject.split("-")[1]) in subject_ids])
# crop
epochs.crop(0, 0.7)
# epochs.apply_baseline()
all_conds = list(epochs.event_id.keys())
# separate epochs based on priming conditions
c_epochs = epochs["Priming==0"]
np_epochs = epochs["Priming==-1"]
pp_epochs = epochs["Priming==1"]
# randomly pick epochs equivalent to the minimum epoch condition count
# mne.epochs.equalize_epoch_counts([np_epochs, pp_epochs, c_epochs], method="random")

# Separate epochs based on distractor location NO PRIMING
left_singleton_epochs_c = c_epochs[[x for x in all_conds if "Target-2-Singleton-1" in x]]
right_singleton_epochs_c = c_epochs[[x for x in all_conds if "Target-2-Singleton-3" in x]]
#mne.epochs.equalize_epoch_counts([left_singleton_epochs, right_singleton_epochs], method="random")
# get the contralateral evoked response and average
contra_singleton_data_c = np.mean([left_singleton_epochs_c.copy().average(picks="C4").get_data(),
                                    right_singleton_epochs_c.copy().average(picks="C3").get_data()], axis=0)
# get the ipsilateral evoked response and average
ipsi_singleton_data_c = np.mean([left_singleton_epochs_c.copy().average(picks="C3").get_data(),
                                  right_singleton_epochs_c.copy().average(picks="C4").get_data()], axis=0)
# compute the difference waves (contra - ipsi) for lateral targets without and with distractor presence and
# lateral distractors
diff_wave_distractor_c = contra_singleton_data_c - ipsi_singleton_data_c

# Separate epochs based on distractor location NEGATIVE PRIMING
left_singleton_epochs_np = np_epochs[[x for x in all_conds if "Target-2-Singleton-1" in x]]
right_singleton_epochs_np = np_epochs[[x for x in all_conds if "Target-2-Singleton-3" in x]]
#mne.epochs.equalize_epoch_counts([left_singleton_epochs, right_singleton_epochs], method="random")
# get the contralateral evoked response and average
contra_singleton_data_np = np.mean([left_singleton_epochs_np.copy().average(picks="C4").get_data(),
                                    right_singleton_epochs_np.copy().average(picks="C3").get_data()], axis=0)
# get the ipsilateral evoked response and average
ipsi_singleton_data_np = np.mean([left_singleton_epochs_np.copy().average(picks="C3").get_data(),
                                  right_singleton_epochs_np.copy().average(picks="C4").get_data()], axis=0)
# compute the difference waves (contra - ipsi) for lateral targets without and with distractor presence and
# lateral distractors
diff_wave_distractor_np = contra_singleton_data_np - ipsi_singleton_data_np

# Separate epochs based on distractor location POSITIVE PRIMING
left_singleton_epochs_pp = pp_epochs[[x for x in all_conds if "Target-2-Singleton-1" in x]]
right_singleton_epochs_pp = pp_epochs[[x for x in all_conds if "Target-2-Singleton-3" in x]]
#mne.epochs.equalize_epoch_counts([left_singleton_epochs, right_singleton_epochs], method="random")
# get the contralateral evoked response and average
contra_singleton_data_pp = np.mean([left_singleton_epochs_pp.copy().average(picks="C4").get_data(),
                                    right_singleton_epochs_pp.copy().average(picks="C3").get_data()], axis=0)
# get the ipsilateral evoked response and average
ipsi_singleton_data_pp = np.mean([left_singleton_epochs_pp.copy().average(picks="C3").get_data(),
                                  right_singleton_epochs_pp.copy().average(picks="C4").get_data()], axis=0)
# compute the difference waves (contra - ipsi) for lateral targets without and with distractor presence and
# lateral distractors
diff_wave_distractor_pp = contra_singleton_data_pp - ipsi_singleton_data_pp

# plot the data
window_length = 51
poly_order = 3
times = epochs.average().times
plt.figure()
plt.plot(times, savgol_filter(diff_wave_distractor_pp[0]*10e5, window_length=window_length, polyorder=poly_order),
         color="darkgreen", label="positive priming")
plt.plot(times, savgol_filter(diff_wave_distractor_np[0]*10e5, window_length=window_length, polyorder=poly_order),
         color="darkred", label="negative priming")
plt.plot(times, savgol_filter(diff_wave_distractor_c[0]*10e5, window_length=window_length, polyorder=poly_order),
         color="grey", label="no priming")
plt.hlines(y=0, xmin=times[0], xmax=times[-1], color="black")
plt.legend()
plt.title("Diff wave (contra-ipsi) priming for lateral targets on electrodes FC5/6")
plt.ylabel("Amplitude [µV]")
plt.xlabel("Time [s]")
