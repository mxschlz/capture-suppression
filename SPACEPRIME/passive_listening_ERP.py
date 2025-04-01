import mne
import matplotlib.pyplot as plt
from utils import get_passive_listening_ERPs
from mne.stats import permutation_cluster_test
from scipy.stats import t
import seaborn as sns
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
times = epochs.times
fig, ax = plt.subplots(1, 3, figsize=(12, 6), sharey=True)
ax = ax.flatten()
# first plot
ax[0].plot(times, contra_target_epochs_data.mean(axis=0)*10e5, color="r")
ax[0].plot(times, ipsi_target_epochs_data.mean(axis=0)*10e5, color="b")
ax[0].plot(times, diff_target[0]*10e5, color="g")
ax[0].hlines(y=0, xmin=times[0], xmax=times[-1])
#ax[0].legend(["Contra", "Ipsi", "Contra-Ipsi"])
ax[0].set_title("Target lateral")
ax[0].set_ylabel("Amplitude [µV]")
ax[0].set_xlabel("Time [s]")
# second plot
ax[1].plot(times, contra_distractor_epochs_data.mean(axis=0)*10e5, color="r")
ax[1].plot(times, ipsi_distractor_epochs_data.mean(axis=0)*10e5, color="b")
ax[1].plot(times, diff_distractor[0]*10e5, color="g")
ax[1].hlines(y=0, xmin=times[0], xmax=times[-1])
#ax[1].legend(["Contra", "Ipsi", "Contra-Ipsi"])
ax[1].set_title("Distractor lateral")
ax[1].set_xlabel("Time [s]")
# control
ax[2].plot(times, contra_control_epochs_data.mean(axis=0)*10e5, color="r")
ax[2].plot(times, ipsi_control_epochs_data.mean(axis=0)*10e5, color="b")
ax[2].plot(times, diff_control[0]*10e5, color="g")
ax[2].hlines(y=0, xmin=times[0], xmax=times[-1])
#ax[2].legend(["Contra", "Ipsi", "Contra-Ipsi"])
ax[2].set_title("Control lateral")
ax[2].set_xlabel("Time [s]")# add t stats on same plot with different axis
twin1 = ax[0].twinx()
twin1.tick_params(axis='y', labelcolor="brown")
twin1.plot(times, result_target[0], color="brown", linestyle="dashed", alpha=0.5)
# fourth plot
twin2 = ax[1].twinx()
twin2.tick_params(axis='y', labelcolor="brown")
twin1.sharey(twin2)
twin2.plot(times, result_distractor[0], color="brown", linestyle="dashed", alpha=0.5)
# control
twin3 = ax[2].twinx()
twin3.tick_params(axis='y', labelcolor="brown")
twin2.sharey(twin3)
twin3.plot(times, result_control[0], color="brown", linestyle="dashed", alpha=0.5)
# set axis label to right plot
twin3.set_ylabel("T-Value", color="brown")
# despine
sns.despine(fig=fig, right=False)

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
