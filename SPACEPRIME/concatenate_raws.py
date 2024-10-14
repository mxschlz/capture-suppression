import mne
import os


subject_id = 101
raws = []
data_path = f"/home/max/data/SPACEPRIME/sub-{subject_id}/eeg/"
raw_filenames = sorted([x for x in os.listdir(data_path) if ".vhdr" in x])
for filename in raw_filenames:
	raws.append(mne.io.read_raw_brainvision(data_path + filename, preload=True))

raw = mne.concatenate_raws(raws)
raw.save(data_path + "concatenated_raw.fif")