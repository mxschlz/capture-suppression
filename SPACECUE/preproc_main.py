import mne
import mne_icalabel
import autoreject
import os
import re
import pandas as pd
from SPACECUE.encoding import *
from SPACECUE import get_data_path
from SPACECUE.bad_chs import bad_chs
from SPACECUE.subjects import subject_ids
import glob


n_jobs = -1  # Number of parallel processes. On my laptop, 12 is max.
mne.set_log_level("INFO")
params = dict(
    resampling_freq=250,
    add_ref_channel="Fz",
    ica_reject_threshold=0.9,
    highpass=1,
    lowpass=40,
    epoch_tmin=-1.0,
    epoch_tmax=1.0
)
# get subject id and settings path
for subject_id in subject_ids:
    data_path = f"{get_data_path()}sourcedata/raw/sci-{subject_id}/eeg/"
    # read raw fif
    raw = mne.io.read_raw_fif(data_path + f"sci-{subject_id}_task-spacecue_raw.fif", preload=True)
    # Downsample because the computer crashes if sampled with 1000 Hz :-(
    raw.resample(params['resampling_freq'])
    
    # Extract integer trigger values from annotation descriptions (e.g. '121' or 'S 121' -> 121)
    trigger_mapping = {}
    for desc in set(raw.annotations.description):
        match = re.search(r'\d+', desc)
        if match:
            trigger_mapping[desc] = int(match.group())
            
    # get events from annotations, using the mapping to get the correct integer trigger values
    events, _ = mne.events_from_annotations(raw, event_id=trigger_mapping)
    # add reference channel back after setting the montage
    raw.add_reference_channels([params["add_ref_channel"]])
    # Add a montage to the data
    montage = mne.channels.make_standard_montage("easycap-M1")
    raw.set_montage(montage)
    # interpolate bad channels
    bads = bad_chs[subject_id]
    if bads:
        raw.info["bads"] = bads
        raw.interpolate_bads()
    # average reference
    raw.set_eeg_reference(ref_channels="average")
    
    # 1. Prepare data strictly for ICA fitting & ICLabel (Must be 1 - 100 Hz)
    raw_ica = raw.copy().filter(l_freq=1.0, h_freq=100.0)
    # apply ICA
    ica = mne.preprocessing.ICA(method="infomax", fit_params=dict(extended=True), random_state=42)
    ica.fit(raw_ica)
    ic_labels = mne_icalabel.label_components(raw_ica, ica, method="iclabel")
    exclude_idx = [idx for idx, (label, prob) in enumerate(zip(ic_labels["labels"], ic_labels["y_pred_proba"])) if label not in ["brain", "other"] and prob > params["ica_reject_threshold"]]
    print(f"Excluding these ICA components: {exclude_idx}")
    del raw_ica  # Free memory
    # ica.plot_properties(raw_filt, picks=exclude_idx)  # inspect the identified IC

    # 2. Filter the MAIN data for your actual analysis BEFORE applying ICA
    raw.filter(l_freq=params["highpass"], h_freq=params["lowpass"])
    
    # 3. Apply the ICA weights to your correctly filtered analysis data
    reconst_raw_filt = raw.copy()
    ica.apply(reconst_raw_filt, exclude=exclude_idx)
    os.makedirs(f"{get_data_path()}derivatives/preprocessing/sci-{subject_id}/eeg/", exist_ok=True)
    # save the preprocessed raw file
    reconst_raw_filt.save(f"{get_data_path()}derivatives/preprocessing/sci-{subject_id}/eeg/sci-{subject_id}_task-spacecue_raw.fif",
                          overwrite=True)
    # save the ica fit
    ica.save(f"{get_data_path()}derivatives/preprocessing/sci-{subject_id}/eeg/sci-{subject_id}_task-spacecue_ica.fif",
             overwrite=True)
    del ica, raw  # delete as an interim step prior to autoreject (that is quite RAM intensive)
    # save the indices that were excluded
    with open(f"{get_data_path()}derivatives/preprocessing/sci-{subject_id}/eeg/sci-{subject_id}_task-spacecue_ica_labels.txt", "w") as file:
        for item in exclude_idx:
            file.write(f"{item}\n")
            
    # Filter EEG_TRIGGER_MAP to only include events actually present in this recording
    present_triggers = set(events[:, 2])
    subject_event_id = {k: v for k, v in EEG_TRIGGER_MAP.items() if v in present_triggers}

    # cut epochs
    epochs = mne.Epochs(reconst_raw_filt, events=events, event_id=subject_event_id, preload=True, tmin=params["epoch_tmin"], tmax=params["epoch_tmax"],
                        baseline=None)
    del reconst_raw_filt
    beh = pd.read_csv(glob.glob(f"{get_data_path()}derivatives/preprocessing/sci-{subject_id}/beh/sci-{subject_id}_clean*.csv")[0])
    # append metadata to epochs
    epochs.metadata = beh
    
    # apply RANSAC for channel interpolation
    ransac = autoreject.Ransac(n_jobs=n_jobs)
    epochs_clean = ransac.fit_transform(epochs)

    # run AutoReject
    ar = autoreject.AutoReject(n_jobs=n_jobs, random_state=42)
    epochs_ar, log = ar.fit_transform(epochs_clean, return_log=True)
    # save epochs
    os.makedirs(f"{get_data_path()}derivatives/epoching/sci-{subject_id}/eeg/", exist_ok=True)
    epochs_ar.save(f"{get_data_path()}derivatives/epoching/sci-{subject_id}/eeg/sci-{subject_id}_task-spacecue-epo.fif",
                overwrite=True)
    # save the drop log
    log.save(f"{get_data_path()}derivatives/epoching/sci-{subject_id}/eeg/sci-{subject_id}_task-spacecue-epo_log.npz",
             overwrite=True)
    del epochs, epochs_clean, epochs_ar, log, beh, ar, ransac