import mne
import numpy as np
import matplotlib.pyplot as plt


def reject_based_on_snr(raw, signal_interval, epoch_interval, event_dict, picks=None,
                        thresholds=np.linspace(200, 60, 50), plot=True):
	snr = np.zeros(len(thresholds))
	events, _ = mne.events_from_annotations(raw)
	for i, thresh in enumerate(thresholds):
		epochs = mne.Epochs(raw, events, event_id=event_dict, tmin=epoch_interval[0], tmax=epoch_interval[1], baseline=None,
		                    reject=dict(eeg=thresh * 1e-6), proj=False, picks=picks,
		                    preload=True)

		signal = epochs.copy().crop(signal_interval[0], signal_interval[1]).average()
		noise = epochs.copy().crop(None, 0.0).average()
		signal_rms = np.sqrt(np.mean(signal._data ** 2))*1e-6
		noise_rms = np.sqrt(np.mean(noise._data ** 2))*1e-6
		snr[i] = signal_rms / noise_rms
	if plot:
		plt.plot(thresholds, snr)
		plt.xlim(200, 50)
		plt.xlabel("treshold in microvolts")
		plt.ylabel("signal to noise ratio")
		plt.show()
	return dict(eeg=(thresholds[snr == max(snr)][0] * 1e-6))