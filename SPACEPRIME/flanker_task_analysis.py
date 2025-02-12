import pandas as pd
import seaborn as sns
from scipy.stats import ttest_rel
from stats import remove_outliers
import mne
import os
import glob
import numpy as np
from SPACEPRIME.plotting import plot_individual_lines
from SPACEPRIME import get_data_path
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.multitest import multipletests
from mne.stats import permutation_cluster_test
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt


# define data root dir
data_root = get_data_path()+ "derivatives/preprocessing/"
# get all the subject ids
subjects = os.listdir(data_root)
# load data from children
df = pd.concat([pd.read_csv(glob.glob(f"{get_data_path()}sourcedata/raw/{subject}/beh/flanker_data_{subject.split("-")[1]}*.csv")[0]) for subject in subjects if int(subject.split("-")[1]) in [105, 107, 108, 110]])
# clean rt data
df = remove_outliers(df, column_name="rt", threshold=2)
# plot reaction time distribution
sns.displot(data=df["rt"])
# plot reaction time
plot = sns.barplot(data=df, x="congruency", y="rt")
plot_individual_lines(ax=plot, data=df, x_col="congruency", y_col="rt")
# plot performance accuracy
plot = sns.barplot(data=df, x="congruency", y="correct")
plot_individual_lines(ax=plot, data=df, x_col="congruency", y_col="correct")
# transform categories into integers for statistics
mapping = dict(congruent=1, incongruent=0, neutral=2)
df["congruency_int"] = df["congruency"].map(mapping)
# do t test
# ad hoc stats
# some stats on behavior
# Perform repeated measures ANOVA for 'correct'
anova_correct = AnovaRM(df, depvar='rt', subject='subject_id', within=['congruency'], aggregate_func="mean").fit()
print(anova_correct.summary())
# Perform paired t-tests
t_stat_1, p_value_1 = ttest_rel(df.query("congruency=='neutral'")["rt"], df.query("congruency=='congruent'")["rt"],
                                  nan_policy="omit")
t_stat_2, p_value_2 = ttest_rel(df.query("congruency=='neutral'")["rt"], df.query("congruency=='incongruent'")["rt"],
                                  nan_policy="omit")
t_stat_3, p_value_3 = ttest_rel(df.query("congruency=='incongruent'")["rt"], df.query("congruency=='congruent'")["rt"],
                                  nan_policy="omit")
# Combine p-values
p_values = [p_value_1, p_value_2, p_value_3]
# Bonferroni correction
reject, p_values_corrected, _, _ = multipletests(p_values, alpha=0.05, method='bonferroni')
print("Corrected p-values (Bonferroni):", p_values_corrected)
print("Reject null hypothesis:", reject)

# get all the congruent and incongruent epochs
epochs = mne.concatenate_epochs([mne.read_epochs(glob.glob(f"{get_data_path()}derivatives/epoching/{subject}/eeg/{subject}_task-flanker-epo.fif")[0]) for subject in subjects if int(subject.split("-")[1]) in [105, 107, 108]])
# epochs.average().plot("Oz")
# compute time frequency bins
# some params
freqs = np.arange(1, 31, 1)  # 1 to 30 Hz
n_cycles = freqs / 2  # different number of cycle per frequency
method = "morlet"  # wavelet
decim = 2  # get
# transform into TFR
power = epochs.compute_tfr(method=method, freqs=freqs, n_cycles=n_cycles, decim=decim, n_jobs=-1, return_itc=False,
                           average=False)
power.average().plot(baseline=(None, 0), combine="mean", mode="logratio")
power.average().plot_topo(baseline=(None, 0), mode="logratio")
# get congruent and incongruent stimulus groups
congruent_cond = ["congruent_left", "congruent_right"]
incongruent_cond = ["incongruent_left", "incongruent_right"]
# divide epochs into congruent and incongruent
congruent_epochs = epochs[congruent_cond]
incongruent_epochs = epochs[incongruent_cond]
# equalize epoch count
mne.epochs.equalize_epoch_counts([congruent_epochs, incongruent_epochs], method="random")
#epochs.apply_baseline(baseline=(None, 0))
# compute TFR from respective conditions
congruent_power = congruent_epochs.compute_tfr(method=method, freqs=freqs, n_cycles=n_cycles, decim=decim, n_jobs=-1,
                                               return_itc=False, average=False)
incongruent_power = incongruent_epochs.compute_tfr(method=method, freqs=freqs, n_cycles=n_cycles, decim=decim, n_jobs=-1,
                                               return_itc=False, average=False)
# apply baseline to the data
congruent_power.apply_baseline(mode="percent", baseline=(None, 0))
incongruent_power.apply_baseline(mode="percent", baseline=(None, 0))
# average
congruent_power_avrg = congruent_power.average()
incongruent_power_avrg = incongruent_power.average()
# compute contrast
diff = incongruent_power_avrg - congruent_power_avrg
diff.plot_topo()
# pick channel of interest
picks = mne.pick_channels(epochs.ch_names, include=["CPz"])
# extract the alpha band activity and average over the frequency dimension
incongruent_alpha_data_avrg = incongruent_power_avrg.get_data(picks=picks, fmin=8, fmax=12).mean(axis=1)
congruent_alpha_data_avrg = congruent_power_avrg.get_data(picks=picks, fmin=8, fmax=12).mean(axis=1)
# do independent t test
incongruent_alpha_data = incongruent_power.get_data(picks=picks, fmin=8, fmax=12).mean(axis=2)
congruent_alpha_data = congruent_power.get_data(picks=picks, fmin=8, fmax=12).mean(axis=2)
ttest = ttest_ind(incongruent_alpha_data, congruent_alpha_data, axis=0)
# plot the stuff
times = incongruent_power.times
fig, ax = plt.subplots(2, 1)
ax[0].plot(times, incongruent_alpha_data_avrg[0], color="red", label="incongruent alpha")
ax[0].plot(times, congruent_alpha_data_avrg[0], color="blue", label="congruent alpha")
ax[0].plot(times, (incongruent_alpha_data_avrg[0] - congruent_alpha_data_avrg[0]), color="green", label="incongruent - congruent")
ax[1].plot(times, ttest[0][0], color="lightgrey", label="t values")
ax[0].legend()
ax[1].legend()
# do cluster-based permutation test
n_permutations = 10000
n_jobs = -1
pval = 0.05
threshold = dict(start=0, step=0.2)  # the smaller the step and the closer the start to 0, the better the approximation
adjacency, _ = mne.channels.find_ch_adjacency(
    diff.info, "eeg")
# plt.matshow(adjacency.toarray())  # take a look at the matrix

# mne.viz.plot_ch_adjacency(epochs.info, adjacency, epochs.info["ch_names"])
# get the data points for respective TFR conditions
# transpose the data because the permutation test expects the channels to be last
incongruent_power_data = incongruent_power.get_data(picks=picks).transpose(0, 2, 3, 1)
congruent_power_data = congruent_power.get_data(picks=picks).transpose(0, 2, 3, 1)
X = [incongruent_power_data, congruent_power_data]
t_obs, clusters, cluster_pv, h0 = permutation_cluster_test(X, threshold=threshold,n_permutations=n_permutations,
                                                           n_jobs=n_jobs, out_type="mask")
# get rid of channel dimension
# t_obs = t_obs.mean(axis=2)
# plot the results
times = epochs.times
fig, (ax, ax2) = plt.subplots(2, 1, figsize=(6, 4), layout="constrained")

# Compute the difference in evoked to determine which was greater since
# we used a 1-way ANOVA which tested for a difference in population means
evoked_power_contrast = diff.get_data(picks=picks)
signs = np.sign(evoked_power_contrast)

# Create new stats image with only significant clusters
F_obs_plot = np.nan * np.ones_like(t_obs)
for c, p_val in zip(clusters, cluster_pv):
    if p_val <= 0.05:
        F_obs_plot[c] = t_obs[c] * signs.mean(axis=0)[c]

ax.imshow(
    t_obs,
    extent=[times[0], times[-1], freqs[0], freqs[-1]],
    aspect="auto",
    origin="lower",
    cmap="gray",
)
max_F = np.nanmax(abs(F_obs_plot))
ax.imshow(
    F_obs_plot,
    extent=[times[0], times[-1], freqs[0], freqs[-1]],
    aspect="auto",
    origin="lower",
    cmap="RdBu_r",
    vmin=-max_F,
    vmax=max_F,
)

ax.set_xlabel("Time (s)")
ax.set_ylabel("Frequency (Hz)")
ax.set_title(f"Induced power")

# plot evoked
evoked_condition_1 = incongruent_epochs.average()
evoked_condition_2 = congruent_epochs.average()
evoked_contrast = mne.combine_evoked(
    [evoked_condition_1, evoked_condition_2], weights=[1, -1]
)
evoked_contrast.plot(axes=ax2, time_unit="s", picks=picks)
