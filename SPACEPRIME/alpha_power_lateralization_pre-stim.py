import mne
import matplotlib.pyplot as plt
import glob
from SPACEPRIME.subjects import subject_ids
from SPACEPRIME import get_data_path
import numpy
import seaborn as sns
import pandas as pd
import numpy as np
from scipy import stats
from SPACEPRIME.plotting import plot_individual_lines
plt.ion()


# --- ALPHA POWER PRE-STIMULUS ---
# We first retrieve our epochs
epochs = mne.concatenate_epochs([mne.read_epochs(glob.glob(f"{get_data_path()}derivatives/epoching/sub-{subject}/eeg/sub-{subject}_task-spaceprime-epo.fif")[0], preload=False) for subject in subject_ids])
# Control for priming
# epochs = epochs["Priming==0"]
# crop to reduce runtime
epochs.crop(None, 0.2)
# First, define ROI electrode picks to reduce computation time.
left_roi = ["TP9", "TP7", "CP5", "CP3", "CP1", "P7", "P5", "P3", "P1", "PO7", "PO3", "O1"]
right_roi = ["TP10", "TP8", "CP6", "CP4", "CP2", "P8", "P6", "P4", "P2", "PO8", "PO4", "O2"]
epochs = epochs.pick(picks=left_roi+right_roi)
# Get the sampling frequency because we need it later
sfreq = epochs.info["sfreq"]
# Now, we need to define some parameters for time-frequency analysis. This is pretty standard, we use morlet wavelet
# convolution (sine wave multiplied by a gaussian distribution to flatten the edges of the filter), we define a number
# of cycles of this wavelet that changes according to the frequency (smaller frequencies get smaller cycles, whereas
# larger frequencies have larger cycles, but all have a cycle of half the frequency value). We also set decim = 1 to
# keep the full amount of data
freqs = numpy.arange(1, 31, 1)  # 1 to 30 Hz
#window_length = 0.5  # window lengths as in Wöstmann et al. (2019)
n_cycles = freqs / 2  # different number of cycle per frequency
method = "morlet"  # wavelet
decim = 7  # keep only every fifth of the samples along the time axis
mode = "zscore"  # z-score normalization
baseline = (-0.9, -0.5)  # baseline from 1000 to 500 ms pre-stimulus
n_jobs = -1  # number of parallel jobs. -1 uses all cores
average = False  # get total oscillatory power, opposed to evoked oscillatory power (get power from ERP)
# Compute time-frequency analysis
power = epochs.compute_tfr(method=method, freqs=freqs, n_cycles=n_cycles, average=average, n_jobs=n_jobs, decim=decim)
# apply baseline
# power.apply_baseline(baseline=baseline, mode=mode)
# Furthermore, since we are interested in the induced alpha power only, we get the evoked alpha by averagring epochs and
# then conducting TFR analysis. We subtract the evoked power from the total power to get induced oscillatory power.
power_evoked = epochs.average().compute_tfr(method=method, freqs=freqs, decim=decim, n_cycles=n_cycles, n_jobs=n_jobs)
# apply baseline
# power_evoked.apply_baseline(baseline=baseline, mode=mode)
# Subtract the evoked power, trial by trial.
power_induced = power.copy()
for trial in range(len(power)):
    power_induced.data[trial] = power.data[trial] - power_evoked.data[0]
# apply baseline
# power_induced.apply_baseline(baseline, mode=mode)
# Now, we devide the epochs into our respective priming conditions. In order to do so, we make use of the metadata
# which was appended to the epochs of every subject during preprocessing. We can access the metadata the same way as
# we would by calling the event_ids in the experiment.
# Crop the epochs into narrower interval preceding stimulus onset
# power.crop(-0.6, 0.1)
power_corrects = power_induced["select_target==True"]
power_incorrects = power_induced["select_target==False"]
# calculate difference spectrum
power_diff = power_corrects.average() - power_incorrects.average()
power_diff.plot(combine="mean")

# define some params for the upcoming analysis
alpha_fmin = 8
alpha_fmax = 12
alpha_tmin = -0.3
alpha_tmax = 0.0
# Store subject-level results
subject_results = {}
for subject in subject_ids:
    print(f"Processing subject: {subject}")
    # Load epochs for the current subject
    power_sub = power_induced[f"subject_id=={subject}"]
    # Divide epochs into correct and incorrect trials
    power_corrects = power_sub["select_target==True"]
    power_incorrects = power_sub["select_target==False"]
    # Average alpha power (7-14 Hz) over all dimensions (epochs, channels, frequencies and time) to get one value per subject
    alpha_roi_corrects = power_corrects.get_data(fmin=alpha_fmin, fmax=alpha_fmax, tmin=alpha_tmin, tmax=alpha_tmax).mean(axis=(0, 1, 2, 3))
    alpha_roi_incorrects = power_incorrects.get_data(fmin=alpha_fmin, fmax=alpha_fmax, tmin=alpha_tmin, tmax=alpha_tmax).mean(axis=(0, 1, 2, 3))
    # Store results
    subject_results[subject] = {
        "alpha_corrects": alpha_roi_corrects,
        "alpha_incorrects": alpha_roi_incorrects}

# We store the mean alpha power value for every subject in a dataframe, so that every subject has one alpha value.
alpha_data = []
subjects = []
conditions = []
# iterate over subjects and append data
for subject, values in subject_results.items():
    alpha_data.append(values["alpha_corrects"])
    alpha_data.append(values["alpha_incorrects"])  # Append incorrect data too
    subjects.append(subject)
    subjects.append(subject) # Append the subject twice, once for each condition.
    conditions.append("Correct")
    conditions.append("Incorrect")
# store everything in a pandas dataframe for further plotting
df = pd.DataFrame({
    "subject_id": subjects,
    "alpha": alpha_data, # combine the correct and incorrect data into one column.
    "condition": conditions,
})
# plot the stuff
plot = sns.barplot(x="condition", y="alpha", data=df)
plot_individual_lines(plot, data=df, x_col="condition", y_col="alpha")
plt.title("Subject-Level Alpha Power (7-14 Hz)")
plt.ylabel("Alpha Power z-score")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
# Subtract correct from incorrect alpha
diff = df.query("condition=='Correct'")["alpha"].reset_index(drop=True) - df.query("condition=='Incorrect'")["alpha"].reset_index(drop=True)
# Compute t test
t, p = stats.ttest_1samp(diff, popmean=0)

# Okay, so now that we have computed the overall alpha power time courses and made a simple comparison between correct
# and incorrect trials, we can further divide the alpha power into ipsi- and contralateral ROIs. This might be more
# sensitive than looking at overall alpha power.
all_conds = list(power.event_id.keys())
# equalize epoch count in all conditions
#mne.epochs.equalize_epoch_counts([left_target_epochs_correct, right_target_epochs_correct,
                                  #left_target_epochs_incorrect, right_target_epochs_incorrect], method="random")
# Store the computed data in a dataframe
alpha_lateralization_subjects_mean = dict(subject_id=[],
                                          correct_ipsi=[],
                                          incorrect_ipsi=[],
                                          correct_contra=[],
                                          incorrect_contra=[])
# Iterate over subjects and get all the ipsi and contra alpha power values
for subject in subject_ids:
    print(f"Processing subject: {subject}")
    sub_power = power_induced[f"subject_id=={subject}"]
    # Split into left and right lateral targets
    left_target_power = sub_power[[x for x in sub_power.event_id if "Target-1-Singleton-2" in x]]
    right_target_power = sub_power[[x for x in sub_power.event_id if "Target-3-Singleton-2" in x]]
    # Now, divide into correct and incorrect trials
    left_target_power_correct = left_target_power["select_target==True"]
    right_target_power_correct = right_target_power["select_target==True"]
    left_target_power_incorrect = left_target_power["select_target==False"]
    right_target_power_incorrect = right_target_power["select_target==False"]
    # Now, divide all power spectra into contra and ipsi target presentation
    # get the trial-wise data for targets contra and ipsilateral to the stimulus, concatenate and average over stimulus.
    # Also, define the alpha frequency range in the get_data() method.
    contra_target_power_correct_data = np.concatenate([left_target_power_correct.copy().get_data(picks=right_roi,
                                                                                                 fmin=alpha_fmin,
                                                                                                 fmax=alpha_fmax,
                                                                                                 tmin=alpha_tmin,
                                                                                                 tmax=alpha_tmax),
                                                        right_target_power_correct.copy().get_data(picks=left_roi,
                                                                                                   fmin=alpha_fmin,
                                                                                                   fmax=alpha_fmax,
                                                                                                   tmin=alpha_tmin,
                                                                                                   tmax=alpha_tmax)],
                                                       axis=0).mean(axis=(0, 1, 2, 3))
    ipsi_target_power_correct_data = np.concatenate([left_target_power_correct.copy().get_data(picks=left_roi,
                                                                                               fmin=alpha_fmin,
                                                                                               fmax=alpha_fmax,
                                                                                               tmin=alpha_tmin,
                                                                                               tmax=alpha_tmax),
                                                      right_target_power_correct.copy().get_data(picks=right_roi,
                                                                                                 fmin=alpha_fmin,
                                                                                                 fmax=alpha_fmax,
                                                                                                 tmin=alpha_tmin,
                                                                                                 tmax=alpha_tmax)],
                                                     axis=0).mean(axis=(0, 1, 2, 3))
    # Do the same for incorrect trials
    contra_target_power_incorrect_data = np.concatenate([left_target_power_incorrect.copy().get_data(picks=right_roi,
                                                                                                      fmin=alpha_fmin,
                                                                                                     fmax=alpha_fmax,
                                                                                                     tmin=alpha_tmin,
                                                                                                     tmax=alpha_tmax),
                                                          right_target_power_incorrect.copy().get_data(picks=left_roi,
                                                                                                       fmin=alpha_fmin,
                                                                                                       fmax=alpha_fmax,
                                                                                                       tmin=alpha_tmin,
                                                                                                       tmax=alpha_tmax)],
                                                         axis=0).mean(axis=(0, 1, 2, 3))
    ipsi_target_power_incorrect_data = np.concatenate([left_target_power_incorrect.copy().get_data(picks=left_roi,
                                                                                                   fmin=alpha_fmin,
                                                                                                   fmax=alpha_fmax,
                                                                                                   tmin=alpha_tmin,
                                                                                                   tmax=alpha_tmax),
                                                        right_target_power_incorrect.copy().get_data(picks=right_roi,
                                                                                                     fmin=alpha_fmin,
                                                                                                     fmax=alpha_fmax,
                                                                                                     tmin=alpha_tmin,
                                                                                                     tmax=alpha_tmax)],
                                                       axis=0).mean(axis=(0, 1, 2, 3))
    # store all the computed data in a dataframe
    alpha_lateralization_subjects_mean["subject_id"].append(subject)
    alpha_lateralization_subjects_mean["incorrect_ipsi"].append(ipsi_target_power_incorrect_data)
    alpha_lateralization_subjects_mean["correct_ipsi"].append(ipsi_target_power_correct_data)
    alpha_lateralization_subjects_mean["incorrect_contra"].append(contra_target_power_incorrect_data)
    alpha_lateralization_subjects_mean["correct_contra"].append(contra_target_power_correct_data)

# Transform into dataframe
df_alpha_lateralization_mean = pd.DataFrame(alpha_lateralization_subjects_mean)
# Melt the DataFrame into a long format
df_melted = df_alpha_lateralization_mean.melt(id_vars='subject_id',
                                              value_vars=['correct_ipsi', 'incorrect_ipsi', 'correct_contra', 'incorrect_contra'],
                                              var_name='condition_side',
                                              value_name='value')

# Split the 'condition_side' column into 'condition' and 'side'
df_melted[['condition', 'side']] = df_melted['condition_side'].str.split('_', expand=True)
# Create the boxplots
plt.figure()
sns.boxplot(x='condition', y='value', hue='side', data=df_melted)
plt.hlines(y=0, xmin=plt.xlim()[0], xmax=plt.xlim()[1], linestyles="dashed", color="black")
plt.title('Boxplots by Condition and Side')
plt.xlabel('Condition')
plt.ylabel('Value')

# Calculate difference in ipsi- versus contralateral correct and incorrect responses
# ATTENTION: here, we calculate the difference as ipsi - contra (usually, I do contra - ipsi for everything)
diff_correct = df_melted.query("condition=='correct'&side=='contra'")["value"].reset_index(drop=True) - df_melted.query("condition=='correct'&side=='ipsi'")["value"].reset_index(drop=True)
diff_incorrect = df_melted.query("condition=='incorrect'&side=='contra'")["value"].reset_index(drop=True) - df_melted.query("condition=='incorrect'&side=='ipsi'")["value"].reset_index(drop=True)
concat_df = pd.concat([diff_correct, diff_incorrect], axis=1, keys=["correct", "incorrect"])
# Melt the DataFrame into long format
df_melted_diff = pd.melt(df, var_name='condition', value_name='value')
# Create the boxplot
plt.figure(figsize=(8, 6))
sns.boxplot(x='condition', y='value', data=df_melted)
plt.title('Alpha lateralization of Correct vs. Incorrect')
plt.xlabel('Condition')
plt.ylabel('Value')
# Do dependent t-test
t, p = stats.ttest_rel(diff_correct, diff_incorrect)

# Now, we basically run the same analysis, with the difference of not averaging over time, so that we get a time course
# for our pre-stimulus alpha power. Sounds good, right! Let's do it.
# Store the computed data in a dataframe
alpha_lateralization_subjects = dict()
# Split into left and right lateral targets
left_target_power = power_induced[[x for x in power.event_id if "Target-1-Singleton-2" in x]]
right_target_power = power_induced[[x for x in power.event_id if "Target-3-Singleton-2" in x]]
# Now, divide into correct and incorrect trials
left_target_power_correct = left_target_power["select_target==True"]
right_target_power_correct = right_target_power["select_target==True"]
left_target_power_incorrect = left_target_power["select_target==False"]
right_target_power_incorrect = right_target_power["select_target==False"]
# Now, divide all power spectra into contra and ipsi target presentation
# get the trial-wise data for targets contra and ipsilateral to the stimulus, concatenate and average over stimulus.
# Also, define the alpha frequency range in the get_data() method.
contra_target_power_correct_data = np.concatenate([left_target_power_correct.copy().get_data(picks=right_roi,
                                                                                             fmin=alpha_fmin,
                                                                                             fmax=alpha_fmax),
                                                    right_target_power_correct.copy().get_data(picks=left_roi,
                                                                                               fmin=alpha_fmin,
                                                                                               fmax=alpha_fmax)],
                                                   axis=0).mean(axis=(0, 1, 2))
ipsi_target_power_correct_data = np.concatenate([left_target_power_correct.copy().get_data(picks=left_roi,
                                                                                           fmin=alpha_fmin,
                                                                                           fmax=alpha_fmax),
                                                  right_target_power_correct.copy().get_data(picks=right_roi,
                                                                                             fmin=alpha_fmin,
                                                                                             fmax=alpha_fmax)],
                                                 axis=0).mean(axis=(0, 1, 2))
# Do the same for incorrect trials
contra_target_power_incorrect_data = np.concatenate([left_target_power_incorrect.copy().get_data(picks=right_roi,
                                                                                                  fmin=alpha_fmin,
                                                                                                 fmax=alpha_fmax),
                                                      right_target_power_incorrect.copy().get_data(picks=left_roi,
                                                                                                   fmin=alpha_fmin,
                                                                                                   fmax=alpha_fmax)],
                                                     axis=0).mean(axis=(0, 1, 2))
ipsi_target_power_incorrect_data = np.concatenate([left_target_power_incorrect.copy().get_data(picks=left_roi,
                                                                                               fmin=alpha_fmin,
                                                                                               fmax=alpha_fmax),
                                                    right_target_power_incorrect.copy().get_data(picks=right_roi,
                                                                                                 fmin=alpha_fmin,
                                                                                                 fmax=alpha_fmax)],
                                                   axis=0).mean(axis=(0, 1, 2))
# store all the computed data in a dataframe
alpha_lateralization_subjects["incorrect_ipsi"] = ipsi_target_power_incorrect_data
alpha_lateralization_subjects["correct_ipsi"] = ipsi_target_power_correct_data
alpha_lateralization_subjects["incorrect_contra"] = contra_target_power_incorrect_data
alpha_lateralization_subjects["correct_contra"] = contra_target_power_correct_data

# Plot single subject and grand average data
times = power.times
incorrect_diff_sub = alpha_lateralization_subjects["incorrect_contra"] - alpha_lateralization_subjects["incorrect_ipsi"]
correct_diff_sub = alpha_lateralization_subjects["correct_contra"] - alpha_lateralization_subjects["correct_ipsi"]
plt.plot(times, incorrect_diff_sub,
         label=f"Incorrect Average", color="red")
plt.plot(times, correct_diff_sub,
         label=f"Correct Average", color="green")
plt.plot(times, (correct_diff_sub-incorrect_diff_sub), color="black",
         label=f"Total alpha lateralization")
plt.legend()
















# iterate over subjects
# --- DO ALPHA POWER LATERALIZATION CALCULATION FOR ALL PRIMING CONDITIONS AND SUBJECTS ---
for subject in subject_ids:
    # no priming
    # compute within-subject alpha power
    power_no_p_ipsi = no_p[f"subject_id=={subject}&TargetLoc==1"].compute_tfr(method=method, freqs=freqs, n_cycles=n_cycles, decim=decim, n_jobs=-1, return_itc=False,
                           average=False).get_data(picks=)
    power_avg_no_p = power_no_p.average()
    alpha_freqs_times_no_p = power_avg_no_p.get_data(fmin=7, fmax=14).mean(axis=0)  # get alpha power frequencies over all channels
    subjects_alpha["no_p"].append(alpha_freqs_times_no_p.mean(axis=0))  # average over all frequencies

    # positive priming
    power_pp = pp[f"subject_id=={subject}"].compute_tfr(method=method, freqs=freqs, n_cycles=n_cycles, decim=decim, n_jobs=-1, return_itc=False,
                           average=False)
    power_avg_pp = power_pp.average()
    alpha_freqs_times_pp = power_avg_pp.get_data(fmin=7, fmax=14).mean(axis=0)  # get alpha power frequencies over all channels
    subjects_alpha["pp"].append(alpha_freqs_times_pp.mean(axis=0))  # average over all frequencies

    # negative priming
    power_np = np[f"subject_id=={subject}"].compute_tfr(method=method, freqs=freqs, n_cycles=n_cycles, decim=decim, n_jobs=-1, return_itc=False,
                           average=False)
    power_avg_np = power_np.average()
    alpha_freqs_times_np = power_avg_np.get_data(fmin=7, fmax=14).mean(axis=0)  # get alpha power frequencies over all channels
    subjects_alpha["np"].append(alpha_freqs_times_np.mean(axis=0))  # average over all frequencies

# get the time vector for plotting purposes
times = numpy.linspace(epochs.tmin, epochs.tmax, epochs.get_data().shape[2])

# plot data
plt.figure()
for k, v in subjects_alpha.items():
    for i, subject_alpha in enumerate(v):
        plt.plot(times, subject_alpha, label=f"Condition: {k}, Subject {subject_ids[i]}", alpha=0.2)

# get average of subjects alpha
grand_average_no_p = numpy.mean(subjects_alpha["no_p"], axis=0)
grand_average_pp = numpy.mean(subjects_alpha["pp"], axis=0)
grand_average_np = numpy.mean(subjects_alpha["np"], axis=0)

plt.plot(times, grand_average_no_p, label="Grand Average no priming", color="grey", linewidth=5)
plt.plot(times, grand_average_pp, label="Grand Average positive priming", color="darkgreen", linewidth=5)
plt.plot(times, grand_average_np, label="Grand Average negative priming", color="darkred", linewidth=5)

plt.xlabel("Time (s)")
plt.ylabel("Alpha Power")
plt.title("Time Course of Alpha Power")
plt.legend()
plt.show()

# Since we cannot observe anything in total alpha power, we might look at how the lateralization differs between
# priming conditions. To do this, we define ROIs and specifically look at how alpha lateralizes in no priming, positive
# priming and negative priming conditions.