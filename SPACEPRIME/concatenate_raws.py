import mne
import os


subject_id = 104
raws = []
data_path = f"/home/max/Insync/schulz.max5@gmail.com/GoogleDrive/PhD/data/SPACEPRIME/sourcedata/raw/sub-{subject_id}/eeg/"
raw_filenames = sorted([x for x in os.listdir(data_path) if ".vhdr" in x])
for filename in raw_filenames:
	if not "flanker_data" in filename:
		raws.append(mne.io.read_raw_brainvision(data_path + filename, preload=True))

raw = mne.concatenate_raws(raws)
raw.save(data_path + f"sub-{subject_id}_task-spaceprime_raw.fif", overwrite=True)
