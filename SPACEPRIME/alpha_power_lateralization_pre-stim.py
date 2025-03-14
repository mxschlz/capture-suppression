import mne
import matplotlib.pyplot as plt
import glob
from SPACEPRIME.subjects import subject_ids
from SPACEPRIME import get_data_path
import numpy
import seaborn as sns
import pandas as pd
import numpy as np
plt.ion()


# --- ALPHA POWER PRE-STIMULUS ---
# We first retrieve our epochs
epochs = mne.concatenate_epochs([mne.read_epochs(glob.glob(f"{get_data_path()}derivatives/epoching/sub-{subject}/eeg/sub-{subject}_task-spaceprime-epo.fif")[0], preload=False) for subject in subject_ids])
# crop to reduce runtime
epochs.crop(None, 0.2)
# First, define ROI electrode picks to reduce computation time.
left_roi = ["TP9", "TP7", "CP5", "CP3", "CP1", "P7", "P5", "P3", "P1", "PO7", "PO3", "O1"]
right_roi = ["TP10", "TP8", "CP6", "CP4", "CP2", "P8", "P6", "P4", "P2", "PO8", "PO4", "O2"]
epochs = epochs.pick(picks=left_roi+right_roi)
# Now, we need to define some parameters for time-frequency analysis. This is pretty standard, we use morlet wavelet
# convolution (sine wave multiplied by a gaussian distribution to flatten the edges of the filter), we define a number
# of cycles of this wavelet that changes according to the frequency (smaller frequencies get smaller cycles, whereas
# larger frequencies have larger cycles, but all have a cycle of half the frequency value). We also set decim = 1 to
# keep the full amount of data
freqs = numpy.arange(1, 31, 1)  # 1 to 30 Hz
#window_length = 0.5  # window lengths as in Wöstmann et al. (2019)
n_cycles = freqs / 2  # different number of cycle per frequency
method = "morlet"  # wavelet
decim = 1  # keep all the samples along the time axis
mode = "zscore"  # z-score normalization
baseline = (-0.9, -0.5)  # baseline from 1000 to 500 ms pre-stimulus
n_jobs = 20  # number of parallel jobs. -1 uses all cores
average = False  # get total oscillatory power, opposed to evoked oscillatory power (get power from ERP)
spec = epochs.compute_tfr(method=method, freqs=freqs, n_cycles=n_cycles, average=average, n_jobs=n_jobs, decim=decim)
# Now, we devide the epochs into our respective priming conditions. In order to do so, we make use of the metadata
# which was appended to the epochs of every subject during preprocessing. We can access the metadata the same way as
# we would by calling the event_ids in the experiment.
spec_corrects = spec["select_target==True"]
spec_incorrects = spec["select_target==False"]
# apply baseline
spec_corrects.apply_baseline(baseline=baseline, mode=mode)
spec_incorrects.apply_baseline(baseline=baseline, mode=mode)
# calculate difference spectrum
spec_diff = spec_corrects.average() - spec_incorrects.average()
spec_diff.plot(combine="mean")

# Store subject-level results
subject_results = {}
for subject in subject_ids:
    print(f"Processing subject: {subject}")
    # Load epochs for the current subject
    epochs_sub = epochs[f"subject_id=={subject}"]
    # Divide epochs into correct and incorrect trials
    corrects = epochs_sub["select_target==True"]
    incorrects = epochs_sub["select_target==False"]
    # Compute time-frequency analysis
    spectrum_corrects = corrects.compute_tfr(method=method, decim=decim, freqs=freqs, n_cycles=n_cycles, n_jobs=n_jobs, average=average)
    spectrum_corrects.apply_baseline(mode=mode, baseline=baseline)
    spectrum_incorrects = incorrects.compute_tfr(method=method, decim=decim, freqs=freqs, n_cycles=n_cycles, n_jobs=n_jobs, average=average)
    spectrum_incorrects.apply_baseline(mode=mode, baseline=baseline)
    # Extract alpha power from ROI
    roi_corrects = spectrum_corrects.pick(picks=left_roi + right_roi)
    roi_incorrects = spectrum_incorrects.pick(picks=left_roi + right_roi)
    # Average alpha power (7-14 Hz)
    alpha_roi_corrects = roi_corrects.get_data(fmin=7, fmax=14).mean(axis=2).mean(axis=1).mean(axis=1)
    alpha_roi_incorrects = roi_incorrects.get_data(fmin=7, fmax=14).mean(axis=2).mean(axis=1).mean(axis=1)
    # Store results
    subject_results[subject] = {
        "alpha_corrects": alpha_roi_corrects.mean(),
        "alpha_incorrects": alpha_roi_incorrects.mean()}

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
    "subject": subjects,
    "alpha": alpha_data, # combine the correct and incorrect data into one column.
    "condition": conditions,
})
# plot the stuff
sns.boxplot(x="condition", y="alpha", data=df)
plt.title("Subject-Level Alpha Power (7-14 Hz)")
plt.ylabel("Alpha Power z-score")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
# Okay, so now that we have computed the overall alpha power time courses and made a simple comparison between correct
# and incorrect trials, we can further divide the alpha power into ipsi- and contralateral ROIs. This might be more
# sensitive than looking at overall alpha power.
all_conds = list(epochs.event_id.keys())
# Split into left and right lateral targets
left_target_epochs = epochs[[x for x in epochs.event_id if "Target-1-Singleton-2" in x]]
right_target_epochs = epochs[[x for x in epochs.event_id if "Target-3-Singleton-2" in x]]
# Now, divide into correct and incorrect trials
left_target_epochs_correct = left_target_epochs["select_target==True"]
right_target_epochs_correct = right_target_epochs["select_target==True"]
left_target_epochs_incorrect = left_target_epochs["select_target==False"]
right_target_epochs_incorrect = right_target_epochs["select_target==False"]
# equalize epoch count in all conditions
mne.epochs.equalize_epoch_counts([left_target_epochs_correct, right_target_epochs_correct,
                                  left_target_epochs_incorrect, right_target_epochs_incorrect], method="random")
# Now, divide all epochs into contra and ipsi
# get the trial-wise data for targets
contra_target_epochs_correct_data = np.concatenate([left_target_epochs_correct.copy().get_data(picks=right_roi),
                                 right_target_epochs_correct.copy().get_data(picks=left_roi)], axis=0).mean(axis=0)
ipsi_target_epochs_correct_data = np.concatenate([left_target_epochs_correct.copy().get_data(picks=left_roi),
                               right_target_epochs_correct.copy().get_data(picks=right_roi)], axis=0).mean(axis=0)























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