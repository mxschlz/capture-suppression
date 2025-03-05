import mne
import os
from SPACEPRIME import get_data_path
import matplotlib.pyplot as plt
import pandas as pd
import scipy
import numpy as np
plt.ion()


def append_behavior(subject, epochs):
    # get behavioral data visual
    mat_data_path_vis = f"G:\\Meine Ablage\\PhD\\hannah_data\\eeg\\behav\\P{subject}_vis_sorted_file.mat"
    mat_data_vis = scipy.io.loadmat(mat_data_path_vis)
    clean_mat_vis = dict()
    for k, v in mat_data_vis.items():
        if "__" not in k:
            clean_mat_vis[k] = v.flatten().tolist()
    clean_mat_vis.pop("blockOrder_vis", None)
    clean_mat_vis.pop("corrAnswersTrain", None)
    # now load auditory behavior
    mat_data_path_aud = f"G:\\Meine Ablage\\PhD\\hannah_data\\eeg\\behav\\P{subject}_aud_sorted_file.mat"
    mat_data_aud = scipy.io.loadmat(mat_data_path_aud)
    clean_mat_aud = dict()
    for k, v in mat_data_aud.items():
        if "__" not in k:
            clean_mat_aud[k] = v.flatten().tolist()
    clean_mat_aud.pop("blockOrder_aud", None)
    clean_mat_aud.pop("corrAnswersTrain", None)
    # load dataframe
    df_vis = pd.DataFrame(clean_mat_vis)
    df_vis = df_vis.reset_index()  # Create 'index' column
    df_vis = df_vis.rename(columns={'index': 'trial_number'})  # Rename for clarity
    # do all the same for auditory events
    df_aud = pd.DataFrame(clean_mat_aud)
    df_aud = df_aud.reset_index()  # Create 'index' column
    df_aud = df_aud.rename(columns={'index': 'trial_number'})  # Rename for clarity

    merged_df = pd.concat([df_vis, df_aud], axis=1)
    visual_event_indices = np.where(events[:, 2] == 20)[0]  # Correctly get indices from visual events
    auditory_event_indices = np.where(events[:, 2] == 80)[0]
    # IMPORTANT CHECK: Ensure enough rows in both dataframes!
    if len(visual_event_indices) > len(df_vis):
        raise ValueError("Not enough rows in df_vis for visual events.")
    if len(auditory_event_indices) > len(df_aud):
        raise ValueError("Not enough rows in df_aud for auditory events.")
    event_codes = np.zeros(len(epochs), dtype=int)
    event_codes[visual_event_indices] = 20
    event_codes[auditory_event_indices] = 80
    metadata = pd.DataFrame({'event_code': event_codes}).reset_index()

    # --- Add columns from BOTH df_vis and df_aud to metadata ---
    # We do this separately to avoid KeyError

    for col in df_vis.columns:
        if col not in metadata.columns:  # Avoid duplicates
            metadata[col] = np.nan

    for col in df_aud.columns:
        if col not in metadata.columns:  # Avoid duplicates
            metadata[col] = np.nan

    # --- Insert data, handling potential KeyErrors gracefully ---

    # Insert visual data
    for col in df_vis.columns:
        if col in metadata.columns:  # added check
            metadata.loc[visual_event_indices, col] = df_vis[col].values[:len(visual_event_indices)]

    # Insert auditory data
    for col in df_aud.columns:
        if col in metadata.columns:  # added check
            metadata.loc[auditory_event_indices, col] = df_aud[col].values[:len(auditory_event_indices)]

    # --- Display and Check ---
    print(metadata)
    print(metadata.info())
    print(metadata.shape)

    metadata["subject_id"] = subject
    epochs.metadata = metadata
    return epochs

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
data_path = "G:\\Meine Ablage\\PhD\\hannah_data\\eeg\\derivatives\\"
sub_ids = sorted(os.listdir(data_path))
bad_subjects = []
for subject_id in sub_ids:
    try:
        components = subject_id.split("_")[2]
        if components[-2] == "0":
            subject = components[-1]
        else:
            subject = components[-2:]
        raw = mne.io.read_raw_fif(f"{data_path}{subject_id}/preprocessing/{subject_id}_task-supratyp_raw.fif",
                                  preload=True)
        events, event_id = mne.events_from_annotations(raw)
        epochs = mne.Epochs(raw, events=events, event_id=event_id, preload=True, tmin=params["epoch_tmin"],
                            tmax=params["epoch_tmax"], baseline=None, event_repeated="merge")
        epochs_appended = append_behavior(subject=subject, epochs=epochs)
        # cut epochs
        flat = dict(eeg=1e-6)
        reject=dict(eeg=250e-6)
        #if subject_id in ["SP_EEG_P0054", 'SP_EEG_P0085']:
            #event_id = {'1': 1, '20': 20, '21': 21, '22': 22, '50': 50, '51': 51, '52': 52, '53': 53, '54': 54, '80': 80}
        #else:
            #event_id = {'1': 1, '20': 20, '21': 21, '22': 22, '50': 50, '51': 51, '52': 52, '53': 53, '54': 54, '80': 80,
                             #'81': 81, '82': 82}
        #epochs_appended.event_id = event_id
        epochs_final = epochs_appended.copy().drop_bad(reject=reject, flat=flat)
        try:
            os.makedirs(f"{data_path}{subject_id}/")
        except FileExistsError:
            print("EEG derivatives epoching directory already exists")
        epochs.save(f"{data_path}{subject_id}\\epoching\\{subject_id}_task-supratyp-epo.fif",
                    overwrite=True)
        del raw, epochs, epochs_appended, epochs_final
    except:
        bad_subjects.append(subject_id)

with open("G:\\Meine Ablage\\PhD\\hannah_data\\eeg\\bad_subjects.txt", "w") as file:
    for item in bad_subjects:
        file.write(f"{item}\n")
