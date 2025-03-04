import mne
import numpy as np
import matplotlib.pyplot as plt
import glob
from scipy.signal import savgol_filter
from SPACEPRIME import get_data_path
from SPACEPRIME.subjects import subject_ids
from SPACEPRIME.plotting import difference_topos
plt.ion()


epochs = mne.concatenate_epochs([mne.read_epochs(glob.glob(f"{get_data_path()}derivatives/epoching/sub-{subject}/eeg/sub-{subject}_task-spaceprime-epo.fif")[0]) for subject in subject_ids])
#epochs.apply_baseline((None, 0))
# epochs = epochs["select_target==True"]
epochs.crop(0, 0.7)  # crop for better comparability with Mandal et al. (2024)
all_conds = list(epochs.event_id.keys())
# get all channels from epochs
all_chs = epochs.ch_names
# get montage for difference topo plots
settings_path = f"{get_data_path()}settings/"
montage = mne.channels.read_custom_montage(settings_path + "CACS-64_NO_REF.bvef")

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

# compute difference waves for each channel
diff_waves_target, diff_waves_distractor = difference_topos(epochs=epochs, montage=montage)

# --- Topomap Sequence Figure ---
info = epochs.info
# 1. Define Time Range and Step
start_time = 0  # Example: Start 100ms *before* stimulus onset
end_time = 0.6  # Example: End 500ms after stimulus onset
time_step = 0.05  # 50ms step

# 2. Create Time Points
times_to_plot = np.arange(start_time, end_time + time_step, time_step)

# 3. Choose Difference Wave (Target or Distractor)
plot_title_prefix = "Target"  # if diff_waves is diff_waves_distractor
if plot_title_prefix == "Target":
    diff_waves = diff_waves_target  # For TARGET difference waves
elif plot_title_prefix == "Distractor":
    diff_waves = diff_waves_distractor  # For DISTRACTOR difference waves

# 4. Determine Figure Layout (Rows and Columns)
n_plots = len(times_to_plot)
n_cols = int(np.ceil(np.sqrt(n_plots)))
n_rows = int(np.ceil(n_plots / n_cols))

# 5. Create Figure and Subplots
fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 8))
axes = axes.flatten()

# 6.  Find Consistent vmin and vmax
all_values = []
for ch in diff_waves:
    for time_idx in [epochs.time_as_index(t)[0] for t in times_to_plot]:
        all_values.append(diff_waves[ch][time_idx])
vmin = np.min(all_values)
vmax = np.max(all_values)
print(f'vmin = {vmin}, vmax = {vmax}')

# 7. Choose the colormap *explicitly*
cmap = 'RdBu_r'  # Or any other colormap you prefer

# 8. Loop and Plot
for i, time_point in enumerate(times_to_plot):
    time_idx = epochs.time_as_index(time_point)[0]
    data_for_plot = np.array([diff_waves[ch][time_idx] for ch in all_chs if ch in diff_waves])

    # Now pass the `cmap` to plot_topomap
    mne.viz.plot_topomap(data_for_plot, info, axes=axes[i], cmap=cmap)
    axes[i].set_title(f"{time_point * 1000:.0f} ms")

# 9. Remove Unused Subplots
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

# 10. Create ScalarMappable with the correct colormap
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))

# 11. Add Colorbar and Title (using the ScalarMappable)
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])
plt.colorbar(sm, cax=cbar_ax)  # Use the ScalarMappable here

fig.suptitle(f"{plot_title_prefix} Difference Wave Topomaps (50ms Steps)", fontsize=16)
