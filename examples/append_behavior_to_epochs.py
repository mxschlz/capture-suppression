import mne
import scipy
import pandas as pd


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


if __name__ == "__main__":
    import numpy as np
    # get raw data
    raw = mne.io.read_raw_fif("G:\\Meine Ablage\\PhD\\hannah_data\\eeg\\derivatives\\SP_EEG_P0002\\preprocessing\\SP_EEG_P0002_task-supratyp_raw.fif")
    # get events from raws
    events, event_id = mne.events_from_annotations(raw)
    # get all epochs from raw
    epochs = mne.Epochs(raw, events=events, event_id=event_id, preload=True, tmin=-2, tmax=2, baseline=None, event_repeated="merge")
    # rename events in data
    event_id_adapted = {'1': 1, '20': 20, '21': 21, '22': 22, '50': 50, '51': 51, '52': 52, '53': 53, '54': 54, '80': 80,
                        '81': 81, '82': 82}
    epochs.event_id = event_id_adapted