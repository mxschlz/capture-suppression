import mne
import matplotlib.pyplot as plt
from utils import get_passive_listening_ERPs
from mne.stats import permutation_cluster_test
from scipy.stats import t
plt.ion()


# get all the epochs
epochs, contra_distractor_epochs_data, ipsi_distractor_epochs_data, contra_target_epochs_data, ipsi_target_epochs_data, contra_control_epochs_data, ipsi_control_epochs_data, diff_target, diff_distractor, diff_control = get_passive_listening_ERPs()

# get time points from epochs
times =  epochs.times

from scipy.stats import ttest_ind
result_target = ttest_ind(contra_target_epochs_data, ipsi_target_epochs_data, axis=0)
result_distractor = ttest_ind(contra_distractor_epochs_data, ipsi_distractor_epochs_data, axis=0)
result_control = ttest_ind(contra_control_epochs_data, ipsi_control_epochs_data, axis=0)
# plot the data
fig, ax = plt.subplots(2, 3, sharex=True, sharey=False)
# target plot
ax[0][0].plot(times, contra_target_epochs_data.mean(axis=0), color="r")
ax[0][0].plot(times, ipsi_target_epochs_data.mean(axis=0), color="b")
ax[0][0].plot(times, diff_target[0], color="g")
ax[0][0].hlines(y=0, xmin=times[0], xmax=times[-1])
ax[0][0].legend(["Contra", "Ipsi", "Contra-Ipsi"])
ax[0][0].set_title("Target lateral")
ax[0][0].set_ylabel("Amplitude [µV]")
ax[0][0].set_xlabel("Time [s]")
# distractor plot
ax[0][1].plot(times, contra_distractor_epochs_data.mean(axis=0), color="r")
ax[0][1].plot(times, ipsi_distractor_epochs_data.mean(axis=0), color="b")
ax[0][1].plot(times, diff_distractor[0], color="g")
ax[0][1].hlines(y=0, xmin=times[0], xmax=times[-1])
ax[0][1].set_title("Distractor lateral")
ax[0][1].set_ylabel("Amplitude [µV]")
ax[0][1].set_xlabel("Time [s]")
# control plot
ax[0][2].plot(times, contra_control_epochs_data.mean(axis=0), color="r")
ax[0][2].plot(times, ipsi_control_epochs_data.mean(axis=0), color="b")
ax[0][2].plot(times, diff_control[0], color="g")
ax[0][2].hlines(y=0, xmin=times[0], xmax=times[-1])
ax[0][2].set_title("Control lateral")
ax[0][2].set_ylabel("Amplitude [µV]")
ax[0][2].set_xlabel("Time [s]")
# stats target plot
ax[1][0].plot(times, result_target[0])
ax[1][0].hlines(y=0, xmin=times[0], xmax=times[-1])
# stats distractor plot
ax[1][1].plot(times, result_distractor[0])
ax[1][1].hlines(y=0, xmin=times[0], xmax=times[-1])
# stats control plot
ax[1][2].plot(times, result_control[0])
ax[1][2].hlines(y=0, xmin=times[0], xmax=times[-1])
# let stats plot share y axes
ax[1][0].set_ylim((-7, 7))
ax[1][1].set_ylim((-7, 7))
ax[1][2].set_ylim((-7, 7))
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
