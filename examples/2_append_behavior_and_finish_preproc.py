import mne
import os
from SPACEPRIME import get_data_path
import matplotlib.pyplot as plt
import pandas as pd
import scipy
plt.ion()


params = dict(epoch_tmin=-2, epoch_tmax=2)

settings_path = f"{get_data_path()}settings/"
data_path = "/home/max/Insync/schulz.max5@gmail.com/GoogleDrive/PhD/hannah_data/eeg/derivatives"
sub_ids = sorted(os.listdir(data_path))
bad_subjects = []
# iterate over every subject
for subject_id in sub_ids:
    try:
        components = subject_id.split("_")[2]
        if components[-2] == "0":
            subject = components[-1]
        else:
            subject = components[-2:]
        raw = mne.io.read_raw_fif(f"{data_path}/{subject_id}/preprocessing/{subject_id}_task-supratyp_raw.fif",
                                  preload=True)
        events, event_id = mne.events_from_annotations(raw)
        epochs = mne.Epochs(raw, events=events, event_id=event_id, preload=True, tmin=params["epoch_tmin"],
                            tmax=params["epoch_tmax"], baseline=None, event_repeated="drop")
        epochs = epochs["Stimulus/S 20"]
        # get behavioral data visual
        mat_data_path_vis = f"/home/max/Insync/schulz.max5@gmail.com/GoogleDrive/PhD/hannah_data/eeg/behav/P{subject}_vis_sorted_file.mat"
        mat_data_vis = scipy.io.loadmat(mat_data_path_vis)
        clean_mat_vis = dict()
        for k, v in mat_data_vis.items():
            if "__" not in k:
                clean_mat_vis[k] = v.flatten().tolist()
        clean_mat_vis.pop("blockOrder_vis", None)
        clean_mat_vis.pop("corrAnswersTrain", None)
        # load dataframe
        df_vis = pd.DataFrame(clean_mat_vis)
        df_vis = df_vis.reset_index()  # Create 'index' column
        df_vis = df_vis.rename(columns={'index': 'trial_number'})  # Rename for clarity
        epochs.metadata = df_vis
        # cut epochs
        flat = dict(eeg=1e-6)
        reject = dict(eeg=250e-6)
        if subject == "8":
            epochs.info["bads"] = ["FC4"]
            epochs.interpolate_bads()
        epochs_final = epochs.copy().drop_bad(reject=reject, flat=flat)
        try:
            os.makedirs(f"{data_path}/{subject_id}/epoching/")
        except FileExistsError:
            print("EEG derivatives epoching directory already exists")
        epochs_final.save(f"{data_path}/{subject_id}/epoching/{subject_id}_task-supratyp-epo.fif",
                          overwrite=True)
        del raw, epochs, epochs_final
    except:
        bad_subjects.append(subject_id)


with open("/home/max/Insync/schulz.max5@gmail.com/GoogleDrive/PhD/hannah_data/eeg/bad_subjects.txt", "w") as file:
    for item in bad_subjects:
        file.write(f"{item}\n")

# load bad subjects
try:
    with open("/home/max/Insync/schulz.max5@gmail.com/GoogleDrive/PhD/hannah_data/eeg/bad_subjects.txt", "r") as file:  # "r" for read mode
        bad_subjects = [line.strip() for line in file]
        print(bad_subjects)
except FileNotFoundError:
    print("File not found.")


# handle bad subjects
for subject_id in bad_subjects:
    components = subject_id.split("_")[2]
    if components[-2] == "0":
        subject = components[-1]
    else:
        subject = components[-2:]
    raw = mne.io.read_raw_fif(f"{data_path}/{subject_id}/preprocessing/{subject_id}_task-supratyp_raw.fif",
                              preload=True)
    events, event_id = mne.events_from_annotations(raw)
    epochs = mne.Epochs(raw, events=events, event_id=event_id, preload=True, tmin=params["epoch_tmin"],
                        tmax=params["epoch_tmax"], baseline=None, event_repeated="drop")
    epochs = epochs["Stimulus/S 20"]
    # get behavioral data visual
    mat_data_path_vis = f"/home/max/Insync/schulz.max5@gmail.com/GoogleDrive/PhD/hannah_data/eeg/behav/P{subject}_vis_sorted_file.mat"
    mat_data_vis = scipy.io.loadmat(mat_data_path_vis)
    clean_mat_vis = dict()
    for k, v in mat_data_vis.items():
        if "__" not in k:
            clean_mat_vis[k] = v.flatten().tolist()
    clean_mat_vis.pop("blockOrder_vis", None)
    clean_mat_vis.pop("corrAnswersTrain", None)
    # load dataframe
    df_vis = pd.DataFrame(clean_mat_vis)
    df_vis = df_vis.reset_index()  # Create 'index' column
    df_vis = df_vis.rename(columns={'index': 'trial_number'})  # Rename for clarity
    epochs.metadata = df_vis

    # cut epochs
    flat = dict(eeg=1e-6)
    reject=dict(eeg=250e-6)
    if subject == "8":
        epochs.info["bads"] = ["FC4"]
        epochs.interpolate_bads()
    epochs_final = epochs.copy().drop_bad(reject=reject, flat=flat)
    try:
        os.makedirs(f"{data_path}/{subject_id}/epoching/")
    except FileExistsError:
        print("EEG derivatives epoching directory already exists")
    epochs_final.save(f"{data_path}/{subject_id}/epoching/{subject_id}_task-supratyp-epo.fif",
                overwrite=True)
    del raw, epochs, epochs_final
