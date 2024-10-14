import matplotlib.pyplot as plt
import mne
import numpy as np
import scipy.io as sio

from pyprep.prep_pipeline import PrepPipeline

mne.set_log_level("INFO")
# get subject id and settings path
subject_id = 101
data_path = f"/home/max/data/SPACEPRIME/sub-{subject_id}/eeg/"
settings_path = "/home/max/data/setting/"
# read raw fif
raw = mne.io.read_raw_fif(data_path + "concatenated_raw.fif", preload=True)
# Add a montage to the data
montage = mne.channels.read_custom_montage(settings_path + "AS-96_REF.bvef")
# Extract some info
sample_rate = raw.info["sfreq"]
# Make a copy of the data
raw_copy = raw.copy()
# Fit prep
prep_params = {
    "ref_chs": "eeg",
    "reref_chs": "eeg",
    "line_freqs": np.arange(50, sample_rate / 2, 50),
}
# instantiate
prep = PrepPipeline(raw_copy, prep_params, montage)
# fit pipeline
prep.fit()
# check bad channels
print(f"Bad channels: {prep.interpolated_channels}")
print(f"Bad channels original: {prep.noisy_channels_original['bad_all']}")
print(f"Bad channels after interpolation: {prep.still_noisy_channels}")
