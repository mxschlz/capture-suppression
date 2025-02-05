import mne


subject = 104
epochs = mne.read_epochs(f"/home/max/Insync/schulz.max5@gmail.com/GoogleDrive/PhD/data/SPACEPRIME/derivatives/epoching/sub-{subject}/eeg/sub-{subject}_task-flanker-epo.fif",
                                   preload=True)
aligned_epochs = epochs.copy().shift_time(tshift=-0.05)
