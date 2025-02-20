import mne
import os


subject_id = input("Enter the subject name (without extension): ")
raws = []
flanker_raws = []
passive_raws = []
data_path = f"/home/max/Insync/schulz.max5@gmail.com/GoogleDrive/PhD/data/SPACEPRIME/sourcedata/raw/sub-{subject_id}/eeg/"
raw_filenames = sorted([x for x in os.listdir(data_path) if ".vhdr" in x])
for filename in raw_filenames:
	if "passive_listening" in filename:
		passive_raws.append(mne.io.read_raw_brainvision(os.path.join(data_path, filename), preload=True))
	elif "flanker_data" in filename:
		flanker_raws.append(mne.io.read_raw_brainvision(data_path + filename, preload=True))
	else:
		raws.append(mne.io.read_raw_brainvision(data_path + filename, preload=True))
raw = mne.concatenate_raws(raws)
raw_flanker = mne.concatenate_raws(flanker_raws)
raw.save(data_path + f"sub-{subject_id}_task-spaceprime_raw.fif", overwrite=True)
raw_flanker.save(data_path + f"sub-{subject_id}_task-flanker_raw.fif", overwrite=True)
raw_passive = mne.concatenate_raws(passive_raws)
raw_passive.save(data_path + f"sub-{subject_id}_task-passive_raw.fif", overwrite=True)
