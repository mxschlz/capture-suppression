import mne
from SPACEPRIME.encoding import *
import numpy as np

# Assuming 'epochs' is your Epochs object
epochs = mne.read_epochs("/home/max/Insync/schulz.max5@gmail.com/GoogleDrive/PhD/data/SPACEPRIME/derivatives/epoching/sub-106/eeg/sub-106_task-spaceprime_wrong_events-epo.fif",
                                  preload=True)
# Get a copy of the events array
events = epochs.events.copy()
event_ids = np.unique(events[:, 2])

# Subtract 1 from the event codes
events[:, 2] += 1

# Create a new Epochs object with the modified events
epochs_renamed = mne.EpochsArray(epochs.get_data(), epochs.info, events=events, event_id=encoding, tmin=epochs.tmin)

# Optionally, update the event_id dictionary if you have one
if epochs.event_id:
    epochs_renamed.event_id = {k: v+1 for k, v in epochs.event_id.items()}
epochs_renamed.metadata = epochs.metadata
epochs_renamed.save("/home/max/Insync/schulz.max5@gmail.com/GoogleDrive/PhD/data/SPACEPRIME/derivatives/epoching/sub-106/eeg/sub-106_task-spaceprime-epo.fif", overwrite=True)
