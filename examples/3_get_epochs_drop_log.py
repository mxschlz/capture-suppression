import mne
import os
from SPACEPRIME import get_data_path
import matplotlib.pyplot as plt
import seaborn as sns
plt.ion()


mne.set_log_level("INFO")
settings_path = f"{get_data_path()}settings/"
data_path = "/home/max/Insync/schulz.max5@gmail.com/GoogleDrive/PhD/hannah_data/eeg/derivatives/"
sub_ids = sorted(os.listdir(data_path))

# loop over subjects, retrieve number of dropped epochs and append them to a list
epochs_dropped = dict()
for subject in sub_ids:
	try:
		epochs = mne.read_epochs(f"{data_path}/{subject}/epoching/{subject}_task-supratyp-epo.fif",
		                         preload=False)
		drop_log = epochs.drop_log
		print(f"Drop log subject {subject}: \n"
		      f"{drop_log}\n")
		dropped_indices = [i for i, reason in enumerate(drop_log) if reason != ()]
		print(f"Indices of Dropped Epochs: {dropped_indices}\n")
		num_dropped = len(dropped_indices)
		print(f"Number of dropped Epochs: \n"
		      f" {num_dropped}\n")
		epochs_dropped[f"{subject}"] = num_dropped
	except ValueError:
		continue

# save dropped epochs dictionary
with open("/home/max/Insync/schulz.max5@gmail.com/GoogleDrive/PhD/hannah_data/eeg/dropped_epochs.txt", "w") as file:
	for k, v in epochs_dropped.items():
		file.write(f"{k}: {v}\n")

sns.displot(epochs_dropped.values())
