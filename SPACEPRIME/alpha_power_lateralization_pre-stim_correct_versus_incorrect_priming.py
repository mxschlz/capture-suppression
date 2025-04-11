import mne
import matplotlib.pyplot as plt
import glob
from SPACEPRIME.subjects import subject_ids
from SPACEPRIME import get_data_path
import numpy
import numpy as np
import seaborn as sns
from scipy.signal import savgol_filter
plt.ion()


# --- ALPHA POWER PRE-STIMULUS ---
# We first retrieve our epochs
epochs = mne.concatenate_epochs([mne.read_epochs(glob.glob(f"{get_data_path()}derivatives/epoching/sub-{subject}/eeg/sub-{subject}_task-spaceprime-epo.fif")[0], preload=False) for subject in subject_ids])
# Control for priming
#epochs = epochs["Priming==0"]
# crop to reduce runtime
#epochs.crop(-0.5, 0.5)
# define freqs of interest
alpha_fmin = 8
alpha_fmax = 12
# epochs.resample(128)  # downsample from 250 to 128 to reduce RAM cost
# Get the sampling frequency because we need it later
sfreq = epochs.info["sfreq"]
# Now, we need to define some parameters for time-frequency analysis. This is pretty standard, we use morlet wavelet
# convolution (sine wave multiplied by a gaussian distribution to flatten the edges of the filter), we define a number
# of cycles of this wavelet that changes according to the frequency (smaller frequencies get smaller cycles, whereas
# larger frequencies have larger cycles, but all have a cycle of half the frequency value). We also set decim = 1 to
# keep the full amount of data
freqs = numpy.arange(alpha_fmin, alpha_fmax+1, 1)  # 8 to 12 Hz
#window_length = 0.5  # window lengths as in Wöstmann et al. (2019)
n_cycles = freqs / 3  # different number of cycle per frequency
method = "morlet"  # wavelet
decim = 5  # keep only every fifth of the samples along the time axis
mode = "mean"  # normalization
baseline = (None, None)  # Do not use baseline interval
n_jobs = -1  # number of parallel jobs. -1 uses all cores
average = False  # get total oscillatory power, opposed to evoked oscillatory power (get power from ERP)
# apply baseline to epochs
# epochs.apply_baseline(baseline=baseline)
# Compute time-frequency analysis
# First, define ROI electrode picks to reduce computation time.
left_roi = ["TP9", "TP7", "CP5", "CP3", "CP1", "P7", "P5", "P3", "P1", "PO7", "PO3", "O1"]
right_roi = ["TP10", "TP8", "CP6", "CP4", "CP2", "P8", "P6", "P4", "P2", "PO8", "PO4", "O2"]
epochs = epochs.pick(picks=left_roi+right_roi)
power_total = epochs.compute_tfr(method=method, freqs=freqs, n_cycles=n_cycles, average=average, n_jobs=n_jobs, decim=decim)
# Furthermore, since we are interested in the induced alpha power only, we get the evoked alpha by averagring epochs and
# then conducting TFR analysis. We subtract the evoked power from the total power to get induced oscillatory power.
power_evoked = epochs.average().compute_tfr(method=method, freqs=freqs, decim=decim, n_cycles=n_cycles, n_jobs=n_jobs)
# apply baseline
#power_evoked.apply_baseline(baseline=baseline, mode=mode)
# Subtract the evoked power, trial by trial.
power_induced = power_total.copy()
for trial in range(len(power_total)):
    power_induced.data[trial] -= power_evoked.data[0]  # subtract the evoked power from total power
# apply baseline
# power_total.apply_baseline(baseline, mode=mode)
# Okay, so now that we have computed the overall alpha power time courses and made a simple comparison between correct
# and incorrect trials, we can further divide the alpha power into ipsi- and contralateral ROIs. This might be more
# sensitive than looking at overall alpha power.
# equalize epoch count in all conditions
#mne.epochs.equalize_epoch_counts([left_target_epochs_correct, right_target_epochs_correct,
                                  #left_target_epochs_incorrect, right_target_epochs_incorrect], method="random")
# Now, we basically run the same analysis, with the difference of not averaging over time, so that we get a time course
# for our pre-stimulus alpha power. Sounds good, right! Let's do it.
left_target_power = power_total[[x for x in power_total.event_id if "Target-1-Singleton-2" in x]]
right_target_power = power_total[[x for x in power_total.event_id if "Target-3-Singleton-2" in x]]
# Store the computed data in a dictionary
alpha_lateralization_subjects = dict(np=dict(),
                                     pp=dict(),
                                     no_p=dict())  # Split into left and right lateral targets
for subject in subject_ids:
    print(f"Processing subject {subject}")
    # NEGATIVE PRIMING
    left_target_power_np = left_target_power["Priming==-1"]
    right_target_power_np = right_target_power["Priming==-1"]
    left_target_power_correct_np = left_target_power_np[f"select_target==True&subject_id=={subject}"]
    right_target_power_correct_np = right_target_power_np[f"select_target==True&subject_id=={subject}"]
    left_target_power_incorrect_np = left_target_power_np[f"select_target==False&subject_id=={subject}"]
    right_target_power_incorrect_np = right_target_power_np[f"select_target==False&subject_id=={subject}"]
    # Now, divide all power spectra into contra and ipsi target presentation
    # get the trial-wise data for targets contra and ipsilateral to the stimulus, concatenate and average over stimulus.
    # Also, define the alpha frequency range in the get_data() method.
    contra_target_power_correct_data_np = np.concatenate([left_target_power_correct_np.copy().get_data(picks=right_roi,
                                                                                                 fmin=alpha_fmin,
                                                                                                 fmax=alpha_fmax),
                                                       right_target_power_correct_np.copy().get_data(picks=left_roi,
                                                                                                  fmin=alpha_fmin,
                                                                                                  fmax=alpha_fmax)],
                                                      axis=0).mean(axis=(0, 1, 2))
    ipsi_target_power_correct_data_np = np.concatenate([left_target_power_correct_np.copy().get_data(picks=left_roi,
                                                                                               fmin=alpha_fmin,
                                                                                               fmax=alpha_fmax),
                                                     right_target_power_correct_np.copy().get_data(picks=right_roi,
                                                                                                fmin=alpha_fmin,
                                                                                                fmax=alpha_fmax)],
                                                    axis=0).mean(axis=(0, 1, 2))
    # Do the same for incorrect trials
    contra_target_power_incorrect_data_np = np.concatenate([left_target_power_incorrect_np.copy().get_data(picks=right_roi,
                                                                                                     fmin=alpha_fmin,
                                                                                                     fmax=alpha_fmax),
                                                         right_target_power_incorrect_np.copy().get_data(picks=left_roi,
                                                                                                      fmin=alpha_fmin,
                                                                                                      fmax=alpha_fmax)],
                                                        axis=0).mean(axis=(0, 1, 2))
    ipsi_target_power_incorrect_data_np = np.concatenate([left_target_power_incorrect_np.copy().get_data(picks=left_roi,
                                                                                                   fmin=alpha_fmin,
                                                                                                   fmax=alpha_fmax),
                                                       right_target_power_incorrect_np.copy().get_data(picks=right_roi,
                                                                                                    fmin=alpha_fmin,
                                                                                                    fmax=alpha_fmax)],
                                                      axis=0).mean(axis=(0, 1, 2))
    correct_diff_np = contra_target_power_correct_data_np - ipsi_target_power_correct_data_np
    incorrect_diff_np = contra_target_power_incorrect_data_np - ipsi_target_power_incorrect_data_np
    # store all the computed data in a dataframe
    alpha_lateralization_subjects["np"][f"{subject}"] = correct_diff_np - incorrect_diff_np
    # POSITIVE PRIMING
    left_target_power_pp = left_target_power["Priming==1"]
    right_target_power_pp = right_target_power["Priming==1"]
    left_target_power_correct_pp = left_target_power_pp[f"select_target==True&subject_id=={subject}"]
    right_target_power_correct_pp = right_target_power_pp[f"select_target==True&subject_id=={subject}"]
    left_target_power_incorrect_pp = left_target_power_pp[f"select_target==False&subject_id=={subject}"]
    right_target_power_incorrect_pp = right_target_power_pp[f"select_target==False&subject_id=={subject}"]
    # Now, divide all power spectra into contra and ipsi target presentation
    # get the trial-wise data for targets contra and ipsilateral to the stimulus, concatenate and average over stimulus.
    # Also, define the alpha frequency range in the get_data() method.
    contra_target_power_correct_data_pp = np.concatenate([left_target_power_correct_pp.copy().get_data(picks=right_roi,
                                                                                                 fmin=alpha_fmin,
                                                                                                 fmax=alpha_fmax),
                                                       right_target_power_correct_pp.copy().get_data(picks=left_roi,
                                                                                                  fmin=alpha_fmin,
                                                                                                  fmax=alpha_fmax)],
                                                      axis=0).mean(axis=(0, 1, 2))
    ipsi_target_power_correct_data_pp = np.concatenate([left_target_power_correct_pp.copy().get_data(picks=left_roi,
                                                                                               fmin=alpha_fmin,
                                                                                               fmax=alpha_fmax),
                                                     right_target_power_correct_pp.copy().get_data(picks=right_roi,
                                                                                                fmin=alpha_fmin,
                                                                                                fmax=alpha_fmax)],
                                                    axis=0).mean(axis=(0, 1, 2))
    # Do the same for incorrect trials
    contra_target_power_incorrect_data_pp = np.concatenate([left_target_power_incorrect_pp.copy().get_data(picks=right_roi,
                                                                                                     fmin=alpha_fmin,
                                                                                                     fmax=alpha_fmax),
                                                         right_target_power_incorrect_pp.copy().get_data(picks=left_roi,
                                                                                                      fmin=alpha_fmin,
                                                                                                      fmax=alpha_fmax)],
                                                        axis=0).mean(axis=(0, 1, 2))
    ipsi_target_power_incorrect_data_pp = np.concatenate([left_target_power_incorrect_pp.copy().get_data(picks=left_roi,
                                                                                                   fmin=alpha_fmin,
                                                                                                   fmax=alpha_fmax),
                                                       right_target_power_incorrect_pp.copy().get_data(picks=right_roi,
                                                                                                    fmin=alpha_fmin,
                                                                                                    fmax=alpha_fmax)],
                                                      axis=0).mean(axis=(0, 1, 2))
    # store all the computed data in a dataframe
    correct_diff_pp = contra_target_power_correct_data_pp - ipsi_target_power_correct_data_pp
    incorrect_diff_pp = contra_target_power_incorrect_data_pp - ipsi_target_power_incorrect_data_pp
    # store all the computed data in a dataframe
    alpha_lateralization_subjects["pp"][f"{subject}"] = correct_diff_pp - incorrect_diff_pp
    # NO PRIMING
    left_target_power_no_p = left_target_power["Priming==0"]
    right_target_power_no_p = right_target_power["Priming==0"]
    left_target_power_correct_no_p = left_target_power_no_p[f"select_target==True&subject_id=={subject}"]
    right_target_power_correct_no_p = right_target_power_no_p[f"select_target==True&subject_id=={subject}"]
    left_target_power_incorrect_no_p = left_target_power_no_p[f"select_target==False&subject_id=={subject}"]
    right_target_power_incorrect_no_p = right_target_power_no_p[f"select_target==False&subject_id=={subject}"]
    # Now, divide all power spectra into contra and ipsi target presentation
    # get the trial-wise data for targets contra and ipsilateral to the stimulus, concatenate and average over stimulus.
    # Also, define the alpha frequency range in the get_data() method.
    contra_target_power_correct_data_no_p = np.concatenate([left_target_power_correct_no_p.copy().get_data(picks=right_roi,
                                                                                                 fmin=alpha_fmin,
                                                                                                 fmax=alpha_fmax),
                                                       right_target_power_correct_no_p.copy().get_data(picks=left_roi,
                                                                                                  fmin=alpha_fmin,
                                                                                                  fmax=alpha_fmax)],
                                                      axis=0).mean(axis=(0, 1, 2))
    ipsi_target_power_correct_data_no_p = np.concatenate([left_target_power_correct_no_p.copy().get_data(picks=left_roi,
                                                                                               fmin=alpha_fmin,
                                                                                               fmax=alpha_fmax),
                                                     right_target_power_correct_no_p.copy().get_data(picks=right_roi,
                                                                                                fmin=alpha_fmin,
                                                                                                fmax=alpha_fmax)],
                                                    axis=0).mean(axis=(0, 1, 2))
    # Do the same for incorrect trials
    contra_target_power_incorrect_data_no_p = np.concatenate([left_target_power_incorrect_no_p.copy().get_data(picks=right_roi,
                                                                                                     fmin=alpha_fmin,
                                                                                                     fmax=alpha_fmax),
                                                         right_target_power_incorrect_no_p.copy().get_data(picks=left_roi,
                                                                                                      fmin=alpha_fmin,
                                                                                                      fmax=alpha_fmax)],
                                                        axis=0).mean(axis=(0, 1, 2))
    ipsi_target_power_incorrect_data_no_p = np.concatenate([left_target_power_incorrect_no_p.copy().get_data(picks=left_roi,
                                                                                                   fmin=alpha_fmin,
                                                                                                   fmax=alpha_fmax),
                                                       right_target_power_incorrect_no_p.copy().get_data(picks=right_roi,
                                                                                                    fmin=alpha_fmin,
                                                                                                    fmax=alpha_fmax)],
                                                      axis=0).mean(axis=(0, 1, 2))
    # store all the computed data in a dataframe
    correct_diff_no_p = contra_target_power_correct_data_no_p - ipsi_target_power_correct_data_no_p
    incorrect_diff_no_p = contra_target_power_incorrect_data_no_p - ipsi_target_power_incorrect_data_no_p
    # store all the computed data in a dataframe
    alpha_lateralization_subjects["no_p"][f"{subject}"] = correct_diff_no_p - incorrect_diff_no_p


# Plot single subject and grand average data
times = power_total.times
# Now, plot the single subject and grand average data on one canvas
# plot data
layout = """
aa
"""
fig, ax = plt.subplot_mosaic(layout, sharey=True, sharex=True)
for priming, d in alpha_lateralization_subjects.items():
    for sub, diff_time_series in d.items():
        if priming == "np":
            color = "darkred"
        elif priming == "pp":
            color = "darkgreen"
        elif priming == "no_p":
            color = "grey"
        else:
            raise ValueError(f"Unknown key {priming}")
        ax["a"].plot(times, diff_time_series, label=f"Priming: {priming}, Subject {sub}", alpha=0.2, color=color)

# get average of subjects alpha
grand_average_diff_np = np.mean(list(alpha_lateralization_subjects["np"].values()), axis=0)
grand_average_diff_pp = np.mean(list(alpha_lateralization_subjects["pp"].values()), axis=0)
grand_average_diff_no_p = np.mean(list(alpha_lateralization_subjects["no_p"].values()), axis=0)

# plot grand average in black
window_length = 51
poly_order = 3
plt.plot(times, savgol_filter(grand_average_diff_np, window_length=window_length, polyorder=poly_order),
         label="Negative priming", color="darkred")
plt.plot(times, savgol_filter(grand_average_diff_pp, window_length=window_length, polyorder=poly_order),
         label="Positive priming", color="darkgreen")
plt.plot(times, savgol_filter(grand_average_diff_no_p, window_length=window_length, polyorder=poly_order),
         label="No priming", color="grey")

# Add some labeling info
plt.xlabel("Time (s)")
plt.ylabel("Alpha Power")
plt.title("Alpha lateralization (contra-ipsi) time course difference (correct-incorrect)")
plt.hlines(y=0, xmin=times[0], xmax=times[-1], color="black")
plt.vlines(x=0, ymin=plt.ylim()[0], ymax=plt.ylim()[1], linestyle="--", color="black")
plt.legend()
sns.despine()

# --- STATISTICS ---
