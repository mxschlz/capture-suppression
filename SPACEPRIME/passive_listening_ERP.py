import mne
import matplotlib.pyplot as plt
import glob
from SPACEPRIME import get_data_path
from SPACEPRIME.subjects import subject_ids
from utils import get_passive_listening_ERPs
import pandas as pd
from mne.stats import permutation_cluster_test
from scipy.stats import t

plt.ion()


epochs, contra_distractor_epochs_data, ipsi_distractor_epochs_data, contra_target_epochs_data, ipsi_target_epochs_data, diff_target, diff_distractor = get_passive_listening_ERPs()


all_evokeds = dict()
evks_avrgd = dict()

for subject in subject_ids[2:]:
    epochs_single_sub = mne.read_epochs(glob.glob(f"{get_data_path()}derivatives/epoching/sub-{subject}/eeg/sub-{subject}_task-passive-epo.fif")[0])
    n_epochs = epochs_single_sub.events.__len__()
    metadata = pd.DataFrame({"subject": [subject] * n_epochs})
    epochs_single_sub.metadata = metadata
    evoked = epochs_single_sub.average(by_event_type=True)
    for cond in evoked:
        if cond.comment not in all_evokeds.keys():
            all_evokeds[cond.comment] = [cond]
        else:
            all_evokeds[cond.comment].append(cond)
    for key in all_evokeds:
        evks_avrgd[key] = mne.grand_average(all_evokeds[key])

left_target_evks = list()
right_target_evks = list()
left_distractor_evks = list()
right_distractor_evks = list()

for k, v in evks_avrgd.items():
    if "target-location-1" in k:
        contra_target_data = v.pick("C4")
        ipsi_target_data = v.pick("C3")
        left_target_evks.append(v)
    elif "target-location-3" in k:
        right_target_evks.append(v)
    elif "distractor-location-1" in k:
        left_distractor_evks.append(v)
    elif "distractor-location-3" in k:
        right_distractor_evks.append(v)

# get time points from epochs
times =  epochs.times

from scipy.stats import ttest_ind
result_target = ttest_ind(contra_target_epochs_data, ipsi_target_epochs_data, axis=0)
result_distractor = ttest_ind(contra_distractor_epochs_data, ipsi_distractor_epochs_data, axis=0)
# plot the data
fig, ax = plt.subplots(2, 2)
# first plot
ax[0][0].plot(times, contra_target_epochs_data.mean(axis=0), color="r")
ax[0][0].plot(times, ipsi_target_epochs_data.mean(axis=0), color="b")
ax[0][0].plot(times, diff_target[0], color="g")
ax[0][0].axvspan(0.2, 0.3, color='gray', alpha=0.3)  # Shade the area
ax[0][0].axvspan(0.05, 0.15, color='gray', alpha=0.3)  # Shade the area
ax[0][0].hlines(y=0, xmin=times[0], xmax=times[-1])
ax[0][0].legend(["Contra", "Ipsi", "Contra-Ipsi"])
ax[0][0].set_title("Target lateral")
ax[0][0].set_ylabel("Amplitude [µV]")
ax[0][0].set_xlabel("Time [s]")
# second plot
ax[0][1].plot(times, contra_distractor_epochs_data.mean(axis=0), color="r")
ax[0][1].plot(times, ipsi_distractor_epochs_data.mean(axis=0), color="b")
ax[0][1].plot(times, diff_distractor[0], color="g")
ax[0][1].axvspan(0.25, 0.50, color='gray', alpha=0.3)  # Shade the area
ax[0][1].axvspan(0.05, 0.15, color='gray', alpha=0.3)  # Shade the area
ax[0][1].hlines(y=0, xmin=times[0], xmax=times[-1])
ax[0][1].set_title("distractor lateral")
ax[0][1].set_ylabel("Amplitude [µV]")
ax[0][1].set_xlabel("Time [s]")
# third plot
ax[1][0].plot(times, result_target[0])
ax[1][0].axvspan(0.2, 0.3, color='gray', alpha=0.3)  # Shade the area
ax[1][0].axvspan(0.05, 0.15, color='gray', alpha=0.3)  # Shade the area
ax[1][0].hlines(y=0, xmin=times[0], xmax=times[-1])
# fourth plot
ax[1][1].plot(times, result_distractor[0])
ax[1][1].axvspan(0.25, 0.50, color='gray', alpha=0.3)  # Shade the area
ax[1][1].axvspan(0.05, 0.15, color='gray', alpha=0.3)  # Shade the area
ax[1][1].hlines(y=0, xmin=times[0], xmax=times[-1])
plt.tight_layout()

# --- STATISTICS ---
run_on = "Target"  # can be Target or Distractor
n_permutations = 10000  # number of permutations
# some stats
n_jobs = -1
pval = 0.05
tail = 0
# Now we need to set the threshold parameter. For this time-series data (1 electrode pair over time) which is NOT SUITED
# FOR SPATIAL COMPARISON BUT TEMPORAL COMPARISON, we should use a single t-value. A reasonable starting point would be
# a t-value corresponding to an uncorrected p-value of 0.05 for a single comparison. We can calculate this using
# scipy.stats.f.ppf.
n1 = contra_target_epochs_data.shape[0] if run_on == "Target" else contra_distractor_epochs_data.shape[0]
n2 = ipsi_target_epochs_data.shape[0] if run_on == "Target" else contra_distractor_epochs_data.shape[0]
df = n1 + n2 - 2
if tail == 0:
    threshold = t.ppf(1 - pval / 2, df)  # Two-tailed
else:  # tail == -1 or tail == 1
    threshold = t.ppf(pval, df) if tail == -1 else t.ppf(1 - pval, df)
print(f"Using threshold: {threshold}")

# mne.viz.plot_ch_adjacency(epochs.info, adjacency, epochs.info["ch_names"])
X = [contra_target_epochs_data, ipsi_target_epochs_data] if run_on == "Target" else [contra_distractor_epochs_data, ipsi_distractor_epochs_data]
t_obs, clusters, cluster_pv, h0 = permutation_cluster_test(X, threshold=threshold, n_permutations=n_permutations,
                                                           n_jobs=n_jobs, out_type="mask", tail=tail, stat_fun=mne.stats.ttest_ind_no_p)
times = epochs.times
fig, (ax, ax2) = plt.subplots(2, 1, figsize=(8, 4))
ax.set_title("Contra minus ipsi")
ax.plot(
    times,
    diff_target[0]*10e5 if run_on=="Target" else diff_distractor[0]*10e5,
    label="ERP Contrast (Contra minus ipsi)")
ax.set_ylabel("EEG (µV)")
ax.legend()

for i_c, c in enumerate(clusters):
    c = c[0]
    if cluster_pv[i_c] <= pval:
        h = ax2.axvspan(times[c.start], times[c.stop - 1], color="r", alpha=0.3)
    else:
        h = 0
        ax2.axvspan(times[c.start], times[c.stop - 1], color=(0.3, 0.3, 0.3), alpha=0.3)

hf = plt.plot(times, t_obs, "g")
ax2.legend((h,), ("cluster p-value < 0.05",))
ax2.set_xlabel("time (ms)")
ax2.set_ylabel("statistic value")  # which statistic?
