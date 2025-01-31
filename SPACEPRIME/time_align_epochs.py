import mne


template_epochs = mne.read_epochs("/home/max/Insync/schulz.max5@gmail.com/GoogleDrive/PhD/data/SPACEPRIME/derivatives/epoching/sub-106/eeg/sub-106_task-spaceprime-epo.fif",
                                  preload=True)
unaligned_epochs = mne.read_epochs("/home/max/Insync/schulz.max5@gmail.com/GoogleDrive/PhD/data/SPACEPRIME/derivatives/epoching/sub-101/eeg/sub-101_task-spaceprime-epo.fif",
                                   preload=True)
aligned_epochs = unaligned_epochs.copy().shift_time(tshift=-0.35)
aligned_epochs.save("sub-101_task-spaceprime-epo.fif")
