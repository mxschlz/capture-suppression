import mne
import mne_icalabel
import os
from SPACEPRIME import get_data_path
import matplotlib.pyplot as plt
import shutil
plt.ion()


mne.set_log_level("INFO")
params = dict(
    resampling_freq=250,
    add_ref_channel="TP9",
    ica_reject_threshold=0.9,
    highpass=0.1,
    lowpass=30,
    epoch_tmin=-2,
    epoch_tmax=2)
settings_path = f"{get_data_path()}settings/"
data_path = "/home/max/Insync/schulz.max5@gmail.com/GoogleDrive/PhD/hannah_data/eeg/raw/"
sub_ids = sorted(os.listdir(data_path))
for subject_id in sub_ids[45:]:
    if subject_id in ["SP_EEG_P0020"]:  # weird subjects
        continue
    print(f"Preprocessing subject {subject_id} ... ")
    raw_fp = os.path.join(data_path, subject_id)
    raw_filenames = sorted([x for x in os.listdir(raw_fp) if ".vhdr" in x])
    for raw_filename in raw_filenames:
        raw_brainvision = mne.io.read_raw_brainvision(raw_fp + "/" + raw_filename, preload=True)
    # read raw fif
    raw_orig = raw_brainvision
    # Downsample because the computer crashes if sampled with 1000 Hz :-(
    raw = raw_orig.copy().resample(params['resampling_freq'])
    # get events from annotations
    events, event_id = mne.events_from_annotations(raw)
    # add reference channel
    raw.add_reference_channels([params["add_ref_channel"]])
    # Add a montage to the data
    montage = mne.channels.read_custom_montage(settings_path + "CACS-64_NO_REF.bvef")
    raw.set_montage(montage)
    # raw.compute_psd().plot(exclude=[params["add_ref_channel"]])
    # plt.pause(10)
    # plt.close()
    # interpolate bad channels
    # bads = input("Input bad channel: ")
    # if len(bads) > 1:
    #     raw.info["bads"] = [bads]
    #     raw.interpolate_bads()
    # average reference
    raw.set_eeg_reference(ref_channels=["TP9", "TP10"])
    # Filter the data. These values are needed for the CNN to label the ICs effectively
    raw_filt = raw.copy().filter(1, 100)
    # apply ICA
    ica = mne.preprocessing.ICA(method="infomax", fit_params=dict(extended=True))
    ica.fit(raw_filt)
    ic_labels = mne_icalabel.label_components(raw_filt, ica, method="iclabel")
    exclude_idx = [idx for idx, (label, prob) in enumerate(zip(ic_labels["labels"], ic_labels["y_pred_proba"])) if label not in ["brain", "other"] and prob > params["ica_reject_threshold"]]
    print(f"Excluding these ICA components: {exclude_idx}")
    # ica.plot_properties(raw_filt, picks=exclude_idx)  # inspect the identified IC
    # ica.apply() changes the Raw object in-place, so let's make a copy first:
    reconst_raw = raw.copy()
    ica.apply(reconst_raw, exclude=exclude_idx)
    # band pass filter
    reconst_raw_filt = reconst_raw.copy().filter(params["highpass"], params["lowpass"])
    try:
        os.makedirs(f"{data_path}{subject_id}/derivatives/preprocessing")
    except FileExistsError:
        print("EEG derivatives preprocessing directory already exists")
    # save the preprocessed raw file
    reconst_raw_filt.save(f"{data_path}{subject_id}/derivatives/preprocessing/{subject_id}_task-supratyp_raw.fif",
                          overwrite=True)
    # save the ica fit
    ica.save(f"{data_path}{subject_id}/derivatives/preprocessing/{subject_id}_task-supratyp_ica.fif",
             overwrite=True)
    # save the indices that were excluded
    with open(f"{data_path}{subject_id}/derivatives/preprocessing/{subject_id}_task-supratyp_ica_labels.txt", "w") as file:
        for item in exclude_idx:
            file.write(f"{item}\n")
    # cut epochs
    flat = dict(eeg=1e-6)
    reject=dict(eeg=250e-6)
    if subject_id in ["SP_EEG_P0054", 'SP_EEG_P0085']:
        event_id = {'1': 1, '20': 20, '21': 21, '22': 22, '50': 50, '51': 51, '52': 52, '53': 53, '54': 54, '80': 80}
    else:
        event_id = {'1': 1, '20': 20, '21': 21, '22': 22, '50': 50, '51': 51, '52': 52, '53': 53, '54': 54, '80': 80,
                         '81': 81, '82': 82}
    epochs = mne.Epochs(reconst_raw_filt, events=events, event_id=event_id, preload=True, tmin=params["epoch_tmin"],
                        tmax=params["epoch_tmax"], baseline=None, event_repeated="merge", reject=reject, flat=flat)
    try:
        os.makedirs(f"{data_path}{subject_id}/derivatives/epoching")
    except FileExistsError:
        print("EEG derivatives epoching directory already exists")
    epochs.save(f"{data_path}{subject_id}/derivatives/epoching/{subject_id}_task-supratyp-epo.fif",
                overwrite=True)
    del raw_orig, raw, raw_filt, reconst_raw_filt, reconst_raw, epochs, ica


def move_data(eeg_dir):
    try:
        os.makedirs(f"{eeg_dir}/derivatives")
    except FileExistsError:
        print("EEG derivatives directory already exists")
    raw_dir = os.path.join(eeg_dir, "raw")
    derivates_dir_target = os.path.join(eeg_dir, "derivatives")
    subjects = sorted(os.listdir(raw_dir))
    for sub in subjects:
        sub_dir = f"{raw_dir}/{sub}"
        derivates_dir_source = f"{sub_dir}/derivatives"
        derivatives_dir_source_data = os.listdir(derivates_dir_source)
        sub_derivatives_dir_target = os.path.join(derivates_dir_target, sub)
        try:
            os.makedirs(sub_derivatives_dir_target)
        except FileExistsError:
            print("derivatives directory already exists")
        for d in derivatives_dir_source_data:
            shutil.move(src=os.path.join(derivates_dir_source, d), dst=os.path.join(sub_derivatives_dir_target, d))
        os.rmdir(derivates_dir_source)
move_data(eeg_dir='/home/max/Insync/schulz.max5@gmail.com/GoogleDrive/PhD/hannah_data/eeg/')
