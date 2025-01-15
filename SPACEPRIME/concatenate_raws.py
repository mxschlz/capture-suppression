import mne
import os


subject_id = 104
raws = []
flanker_raws = []
data_path = f"/home/max/Insync/schulz.max5@gmail.com/GoogleDrive/PhD/data/SPACEPRIME/sourcedata/raw/sub-{subject_id}/eeg/"
raw_filenames = sorted([x for x in os.listdir(data_path) if ".vhdr" in x])
for filename in raw_filenames:
	if not "flanker_data" in filename:
		raws.append(mne.io.read_raw_brainvision(data_path + filename, preload=True))
	elif "flanker_data" in filename:
		flanker_raws.append(mne.io.read_raw_brainvision(data_path + filename, preload=True))
raw = mne.concatenate_raws(raws)
raw_flanker = mne.concatenate_raws(flanker_raws)
raw.save(data_path + f"sub-{subject_id}_task-spaceprime_raw.fif", overwrite=True)
raw_flanker.save(data_path + f"sub-{subject_id}_task-flanker_raw.fif", overwrite=True)
