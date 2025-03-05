import mne
import scipy
import pandas as pd


def append_behavior(subject, epochs, behavior):
    mat_data_path = "G:\\Meine Ablage\\PhD\\hannah_data\\eeg\\behav\\P1_vis_sorted_file.mat"
    mat_data = scipy.io.loadmat(mat_data_path)
    clean_mat = dict()
    for k, v in mat_data.items():
        if "__" not in k:
            clean_mat[k] = v.flatten().tolist()
    clean_mat.pop("blockOrder_vis", None)
    clean_mat.pop("corrAnswersTrain", None)
    df = pd.DataFrame(clean_mat)

    epochs = mne.read_epochs("G:\\Meine Ablage\\PhD\\hannah_data\\eeg\\derivatives\\SP_EEG_P0001\\epoching\\SP_EEG_P0001_task-spaceprime-epo.fif")
    epochs_vis = epochs["20"]
    epochs_aud = epochs["80"]
    epochs_vis.metadata = df


if __name__ == "__main__":
    raw = mne.io.read_raw_fif("/home/max/Insync/schulz.max5@gmail.com/GoogleDrive/PhD/hannah_data/eeg/derivatives/SP_EEG_P0001/preprocessing/SP_EEG_P0001_task-supratyp_raw.fif")
    events, event_id = mne.events_from_annotations(raw)
    epochs = mne.Epochs(raw, events=events, event_id=event_id, preload=True, tmin=-2, tmax=2, baseline=None, event_repeated="merge")
