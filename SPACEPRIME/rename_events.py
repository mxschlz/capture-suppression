import mne
import numpy as np


def add_to_events(epochs_orig, new_encoding, change_by=1):
    events = epochs_orig.events.copy()
    events[:, 2] += change_by
    epochs_renamed = mne.EpochsArray(epochs_orig.get_data(), epochs_orig.info, events=events, event_id=new_encoding,
                                     tmin=epochs_orig.tmin)
    if change_by > 0:
        if epochs_orig.event_id:
            epochs_renamed.event_id = {k: v + np.abs(change_by) for k, v in epochs_orig.event_id.items()}
    elif change_by < 0:
        if epochs_orig.event_id:
            epochs_renamed.event_id = {k: v - np.abs(change_by) for k, v in epochs_orig.event_id.items()}
    return epochs_renamed
