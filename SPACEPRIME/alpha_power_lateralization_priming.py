import mne
import matplotlib.pyplot as plt
import glob
from SPACEPRIME.subjects import subject_ids
from SPACEPRIME import get_data_path
import numpy
plt.ion()


# --- ALPHA POWER IN PRIMING CONDITIONS ---
# We first retrieve our epochs
epochs = mne.concatenate_epochs([mne.read_epochs(glob.glob(f"{get_data_path()}derivatives/epoching/sub-{subject}/eeg/sub-{subject}_task-spaceprime-epo.fif")[0], preload=False) for subject in subject_ids])
# Now, we devide the epochs into our respective priming conditions. In order to do so, we make use of the metadata
# which was appended to the epochs of every subject during preprocessing. We can access the metadata the same way as
# we would by calling the event_ids in the experiment.
corrects = epochs["select_target==True"]
incorrects = epochs["select_target==False"]
# Now, we need to define some parameters for time-frequency analysis. This is pretty standard, we use morlet wavelet
# convolution (sine wave multiplied by a gaussian distribution to flatten the edges of the filter), we define a number
# of cycles of this wavelet that changes according to the frequency (smaller frequencies get smaller cycles, whereas
# larger frequencies have larger cycles, but all have a cycle of half the frequency value). We also set decim = 1 to
# keep the full amount of data
freqs = numpy.arange(7, 14, 1)  # 7 to 14 Hertz (alpha range)
#window_length = 0.5  # window lengths as in Wöstmann et al. (2019)
n_cycles = freqs / 2  # different number of cycle per frequency
method = "morlet"  # wavelet
decim = 1  # keep all the samples along the time axis
# store all the lateralization indices for selection and suppression, respectively
alpha_lateralization_scores = dict(selection_no_p=[],
                                   selection_pp=[],
                                   selection_np=[])
alpha_corrects = corrects.compute_tfr(method=method, decim=decim, freqs=freqs, n_cycles=n_cycles)
alpha_incorrects = incorrects.compute_tfr(method=method, decim=decim, freqs=freqs, n_cycles=n_cycles)

# We want to quantify lateralization effects in terms of lateralization indices. Ultimately, we want to look at pre-stimulus
# alpha lateralization
# define contra- and ipsilateral electrode picks to calculate alpha lateralization scores
left_elecs = ["TP9", "TP7", "CP5", "CP3", "CP1", "P7", "P5", "P3", "P1", "PO7", "PO3", "O1"]
right_elecs = ["TP10", "TP8", "CP6", "CP4", "CP2", "P8", "P6", "P4", "P2", "PO8", "PO3", "O2"]
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