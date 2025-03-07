import mne
import matplotlib.pyplot as plt
import glob
from SPACEPRIME.subjects import subject_ids
from SPACEPRIME import get_data_path
import numpy
plt.ion()


# load epochs
epochs = mne.concatenate_epochs([mne.read_epochs(glob.glob(f"{get_data_path()}derivatives/epoching/sub-{subject}/eeg/sub-{subject}_task-spaceprime-epo.fif")[0], preload=False) for subject in subject_ids])
no_p = epochs["Priming==0"]
pp = epochs["Priming==1"]
np = epochs["Priming==-1"]
# some params
freqs = numpy.arange(1, 31, 1)  # 1 to 30 Hz
window_length = 0.5  # window lengths as in Wöstmann et al. (2019)
n_cycles = freqs * window_length / 2  # different number of cycle per frequency
method = "morlet"  # wavelet
decim = 1  # keep all the samples along the time axis
subjects_alpha = dict(no_p=[],
                      pp=[],
                      np=[])  # store the within-subject alpha power
# get the time vector for plotting purposes
times = numpy.linspace(epochs.tmin, epochs.tmax, epochs.get_data().shape[2])
# iterate over subjects
# --- DO ALPHA POWER CALCULATIONS FOR ALL PRIMING CONDITIONS AND SUBJECTS ---
for subject in subject_ids:
    # no priming
    power_no_p = no_p[f"subject_id=={subject}"].compute_tfr(method=method, freqs=freqs, n_cycles=n_cycles, decim=decim, n_jobs=-1, return_itc=False,
                           average=False)
    # power_no_p.apply_baseline((None, 0), mode="logratio", verbose=False)
    power_avg_no_p = power_no_p.average()
    alpha_freqs_times_no_p = power_avg_no_p.get_data(fmin=7, fmax=14).mean(axis=0)  # get alpha power frequencies over all channels
    subjects_alpha["no_p"].append(alpha_freqs_times_no_p.mean(axis=0))  # average over all frequencies

    # positive priming
    power_pp = pp[f"subject_id=={subject}"].compute_tfr(method=method, freqs=freqs, n_cycles=n_cycles, decim=decim, n_jobs=-1, return_itc=False,
                           average=False)
    # power_pp.apply_baseline((None, 0), mode="logratio", verbose=False)
    power_avg_pp = power_pp.average()
    alpha_freqs_times_pp = power_avg_pp.get_data(fmin=7, fmax=14).mean(axis=0)  # get alpha power frequencies over all channels
    subjects_alpha["pp"].append(alpha_freqs_times_pp.mean(axis=0))  # average over all frequencies

    # negative priming
    power_np = np[f"subject_id=={subject}"].compute_tfr(method=method, freqs=freqs, n_cycles=n_cycles, decim=decim, n_jobs=-1, return_itc=False,
                           average=False)
    # power_np.apply_baseline((None, 0), mode="logratio", verbose=False)
    power_avg_np = power_np.average()
    alpha_freqs_times_np = power_avg_np.get_data(fmin=7, fmax=14).mean(axis=0)  # get alpha power frequencies over all channels
    subjects_alpha["np"].append(alpha_freqs_times_np.mean(axis=0))  # average over all frequencies

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
