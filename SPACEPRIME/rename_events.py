import mne


def add_to_events(epochs_orig, new_encoding):
    events = epochs_orig.events.copy()
    events[:, 2] += 1
    epochs_renamed = mne.EpochsArray(epochs_orig.get_data(), epochs_orig.info, events=events, event_id=new_encoding, tmin=epochs_orig.tmin)
    if epochs_orig.event_id:
        epochs_renamed.event_id = {k: v + 1 for k, v in epochs_orig.event_id.items()}
    return epochs_renamed
