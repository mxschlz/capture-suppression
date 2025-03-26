import mne
import numpy as np
import glob
from SPACEPRIME import get_data_path
from SPACEPRIME.subjects import subject_ids

def get_passive_listening_ERPs():
    # load epochs
    epochs = mne.concatenate_epochs([mne.read_epochs(glob.glob(f"{get_data_path()}derivatives/epoching/sub-{subject}/eeg/sub-{subject}_task-passive-epo.fif")[0]) for subject in subject_ids[2:]])
    # epochs.apply_baseline((None, 0))
    # get all conditions
    # all_conds = list(epochs.event_id.keys())
    # get target, distractor and control epochs
    # target_epochs = epochs[[x for x in all_conds if "target" in x]]
    # distractor_epochs = epochs[[x for x in all_conds if "distractor" in x]]
    # control_epochs = epochs[[x for x in all_conds if "control" in x]]
    # plot ERP
    # mne.viz.plot_compare_evokeds([target_epochs.average(), distractor_epochs.average(), control_epochs.average()], picks="FCz")

    # plot diff waves
    all_conds = list(epochs.event_id.keys())
    # Separate epochs based on distractor location
    left_singleton_epochs = epochs[[x for x in all_conds if "distractor-location-1" in x]]
    right_singleton_epochs = epochs[[x for x in all_conds if "distractor-location-3" in x]]
    mne.epochs.equalize_epoch_counts([left_singleton_epochs, right_singleton_epochs], method="random")
    # get the contralateral evoked response and average
    contra_singleton_data = np.mean([left_singleton_epochs.copy().average(picks=["C4"]).get_data(),
                                        right_singleton_epochs.copy().average(picks=["C3"]).get_data()], axis=0)
    # get the ipsilateral evoked response and average
    ipsi_singleton_data = np.mean([left_singleton_epochs.copy().average(picks=["C3"]).get_data(),
                                      right_singleton_epochs.copy().average(picks=["C4"]).get_data()], axis=0)

    # now, do the same for the lateral targets
    # Separate epochs based on target location
    left_target_epochs = epochs[[x for x in all_conds if "target-location-1" in x]]
    right_target_epochs = epochs[[x for x in all_conds if "target-location-3" in x]]
    mne.epochs.equalize_epoch_counts([left_target_epochs, right_target_epochs], method="random")
    # get the contralateral evoked response and average
    contra_target_data = np.mean([left_target_epochs.copy().average(picks=["C4"]).get_data(),
                                     right_target_epochs.copy().average(picks=["C3"]).get_data()], axis=0)
    # get the ipsilateral evoked response and average
    ipsi_target_data = np.mean([left_target_epochs.copy().average(picks=["C3"]).get_data(),
                                   right_target_epochs.copy().average(picks=["C4"]).get_data()], axis=0)

    # now, do the same for the lateral controls
    # Separate epochs based on controls location
    left_control_epochs = epochs[[x for x in all_conds if "control-location-1" in x]]
    right_control_epochs = epochs[[x for x in all_conds if "control-location-3" in x]]
    mne.epochs.equalize_epoch_counts([left_control_epochs, right_control_epochs], method="random")
    # get the contralateral evoked response and average
    contra_control_data = np.mean([left_control_epochs.copy().average(picks=["C4"]).get_data(),
                                     right_control_epochs.copy().average(picks=["C3"]).get_data()], axis=0)
    # get the ipsilateral evoked response and average
    ipsi_control_data = np.mean([left_control_epochs.copy().average(picks=["C3"]).get_data(),
                                   right_control_epochs.copy().average(picks=["C4"]).get_data()], axis=0)


    # get the trial-wise data for targets
    contra_target_epochs_data = np.mean(np.concatenate([left_target_epochs.copy().get_data(picks="C4"),
                                     right_target_epochs.copy().get_data(picks="C3")], axis=1), axis=1)
    ipsi_target_epochs_data = np.mean(np.concatenate([left_target_epochs.copy().get_data(picks="C3"),
                                   right_target_epochs.copy().get_data(picks="C4")], axis=1), axis=1)
    # get the trial-wise data for singletons
    contra_singleton_epochs_data = np.mean(np.concatenate([left_singleton_epochs.copy().get_data(picks="C4"),
                                     right_singleton_epochs.copy().get_data(picks="C3")], axis=1), axis=1)
    ipsi_singleton_epochs_data = np.mean(np.concatenate([left_singleton_epochs.copy().get_data(picks="C3"),
                                   right_singleton_epochs.copy().get_data(picks="C4")], axis=1), axis=1)
    # get the trial-wise data for controls
    contra_control_epochs_data = np.mean(np.concatenate([left_control_epochs.copy().get_data(picks="C4"),
                                     right_control_epochs.copy().get_data(picks="C3")], axis=1), axis=1)
    ipsi_control_epochs_data = np.mean(np.concatenate([left_control_epochs.copy().get_data(picks="C3"),
                                   right_control_epochs.copy().get_data(picks="C4")], axis=1), axis=1)
    diff_singleton = contra_singleton_data - ipsi_singleton_data
    diff_target = contra_target_data - ipsi_target_data
    diff_control = contra_control_data - ipsi_control_data

    return (epochs, contra_singleton_epochs_data, ipsi_singleton_epochs_data, contra_target_epochs_data,
            ipsi_target_epochs_data, contra_control_epochs_data, ipsi_control_epochs_data, diff_target, diff_singleton,
            diff_control)
