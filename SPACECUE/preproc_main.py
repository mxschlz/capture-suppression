import mne
import mne_icalabel
import autoreject
import os
import re
import pandas as pd
from SPACECUE.encoding import *
from SPACECUE import get_data_path
from SPACECUE.bad_chs import bad_chs
from SPACECUE.subjects import subject_ids
import glob
from SPACECUE.concatenate_raws import concatenate_raws
from SPACECUE.move_video_data import move_video_data


n_jobs = -1  # Number of parallel processes. On my laptop, 12 is max.
mne.set_log_level("INFO")
params = dict(
    resampling_freq=250,
    add_ref_channel="Fz",
    ica_reject_threshold=0.9,
    highpass=1,
    lowpass=40,
    epoch_tmin=-1.0,
    epoch_tmax=1.0
)
START_FROM_INTERIM = True  # Set to True to skip raw filtering and ICA and load the saved interim raw file

def ensure_subject_registered(subject_id):
    """Ensures subject_id is present in SPACECUE/subjects.py so downstream concatenation includes it."""
    import SPACECUE.subjects as subs
    if subject_id not in subs.subject_ids:
        subjects_file = os.path.join(os.path.dirname(__file__), 'subjects.py')
        new_ids = sorted(list(set(subs.subject_ids + [subject_id])))
        formatted_list = "subject_ids = [\n" + "".join([f"    {sid},\n" for sid in new_ids]) + "]\n"
        with open(subjects_file, 'w') as f:
            f.write(formatted_list)
        subs.subject_ids = new_ids
        print(f"[INFO] sci-{subject_id}: Automatically registered in SPACECUE/subjects.py")

# Subject IDs to preprocess in this run
subjects_to_process = [16]

for subject_id in subjects_to_process:
    ensure_subject_registered(subject_id)
    print(f"\n=========================================================")
    print(f"   PREPARING SUBJECT: {subject_id}")
    print(f"=========================================================")
    
    interim_path = f"{get_data_path()}derivatives/preprocessing/sci-{subject_id}/eeg/sci-{subject_id}_task-spacecue_raw.fif"

    if START_FROM_INTERIM and os.path.exists(interim_path):
        print(f"Loading interim step for subject {subject_id} (skipping ICA)...")
        reconst_raw_filt = mne.io.read_raw_fif(interim_path, preload=True)
        # Extract integer trigger values from annotation descriptions
        trigger_mapping = {}
        for desc in set(reconst_raw_filt.annotations.description):
            match = re.search(r'\d+', desc)
            if match:
                trigger_mapping[desc] = int(match.group())
        events, _ = mne.events_from_annotations(reconst_raw_filt, event_id=trigger_mapping)
    else:
        move_video_data(subject_id)
        concatenate_raws(subject_id)
    
        data_path = f"{get_data_path()}sourcedata/raw/sci-{subject_id}/eeg/"
        # read raw fif
        raw = mne.io.read_raw_fif(data_path + f"sci-{subject_id}_task-spacecue_raw.fif", preload=True)
        # Downsample because the computer crashes if sampled with 1000 Hz :-(
        raw.resample(params['resampling_freq'])
        
        # Extract integer trigger values from annotation descriptions (e.g. '121' or 'S 121' -> 121)
        trigger_mapping = {}
        for desc in set(raw.annotations.description):
            match = re.search(r'\d+', desc)
            if match:
                trigger_mapping[desc] = int(match.group())
                
        # get events from annotations, using the mapping to get the correct integer trigger values
        events, _ = mne.events_from_annotations(raw, event_id=trigger_mapping)
        # add reference channel back after setting the montage
        raw.add_reference_channels([params["add_ref_channel"]])
        # Add a montage to the data
        montage = mne.channels.make_standard_montage("easycap-M1")
        raw.set_montage(montage)
        # interpolate bad channels
        bads = bad_chs.get(subject_id, None)
        if bads:
            raw.info["bads"] = bads
            raw.interpolate_bads()
        # average reference
        raw.set_eeg_reference(ref_channels="average")
        
        # 1. Prepare data strictly for ICA fitting & ICLabel (Must be 1 - 100 Hz)
        raw_ica = raw.copy().filter(l_freq=1.0, h_freq=100.0)
        # apply ICA
        ica = mne.preprocessing.ICA(method="infomax", fit_params=dict(extended=True), random_state=42)
        ica.fit(raw_ica)
        ic_labels = mne_icalabel.label_components(raw_ica, ica, method="iclabel")
        exclude_idx = [idx for idx, (label, prob) in enumerate(zip(ic_labels["labels"], ic_labels["y_pred_proba"])) if label not in ["brain", "other"] and prob > params["ica_reject_threshold"]]
        print(f"Excluding these ICA components: {exclude_idx}")
        del raw_ica  # Free memory
        # ica.plot_properties(raw_filt, picks=exclude_idx)  # inspect the identified IC
    
        # 2. Filter the MAIN data for your actual analysis BEFORE applying ICA
        raw.filter(l_freq=params["highpass"], h_freq=params["lowpass"])
        
        # 3. Apply the ICA weights to your correctly filtered analysis data
        reconst_raw_filt = raw.copy()
        ica.apply(reconst_raw_filt, exclude=exclude_idx)
        os.makedirs(f"{get_data_path()}derivatives/preprocessing/sci-{subject_id}/eeg/", exist_ok=True)
        # save the preprocessed raw file
        reconst_raw_filt.save(interim_path, overwrite=True)
        # save the ica fit
        ica.save(f"{get_data_path()}derivatives/preprocessing/sci-{subject_id}/eeg/sci-{subject_id}_task-spacecue_ica.fif",
                 overwrite=True)
        del ica, raw  # delete as an interim step prior to autoreject (that is quite RAM intensive)
        # save the indices that were excluded
        with open(f"{get_data_path()}derivatives/preprocessing/sci-{subject_id}/eeg/sci-{subject_id}_task-spacecue_ica_labels.txt", "w") as file:
            for item in exclude_idx:
                file.write(f"{item}\n")
            
    # Filter EEG_TRIGGER_MAP to only include events actually present in this recording
    present_triggers = set(events[:, 2])
    subject_event_id = {k: v for k, v in EEG_TRIGGER_MAP.items() if v in present_triggers}

    # cut epochs
    epochs = mne.Epochs(reconst_raw_filt, events=events, event_id=subject_event_id, preload=True, tmin=params["epoch_tmin"], tmax=params["epoch_tmax"],
                        baseline=None)
    del reconst_raw_filt
    # Format and load clean behavioral data (stimulus-centric to retain all 1800 trials)
    from SPACECUE.clean_behavioral_data import clean_subject_behavioral_data
    beh, beh_path = clean_subject_behavioral_data(subject_id, overwrite=False)
    
    # Check if number of epochs matches metadata rows, and align if a trigger was missed
    if len(epochs) != len(beh):
        print(f"\n[WARNING] Metadata length ({len(beh)}) does not match epochs count ({len(epochs)}) for subject {subject_id}!")
        loc_map = {'Left': 1, 'Front': 2, 'Right': 3, 1: 1, 2: 2, 3: 3, 1.0: 1, 2.0: 2, 3.0: 3}
        exp_triggers = []
        for r in beh.itertuples():
            t_loc = int(loc_map.get(getattr(r, 'TargetLoc'), 0))
            s_loc = int(loc_map.get(getattr(r, 'SingletonLoc'), 0))
            hp_loc = int(loc_map.get(getattr(r, 'HP_Distractor_Loc'), 0))
            hp_prob = getattr(r, 'HP_Distractor_Prob')
            key = f"Target-{t_loc}-Singleton-{s_loc}-HP-Distractor-Loc-{hp_loc}-{hp_prob}"
            exp_triggers.append(EEG_TRIGGER_MAP.get(key, -1))

        epoch_triggers = epochs.events[:, 2]
        missing_indices = []
        eeg_idx = 0
        for beh_idx in range(len(exp_triggers)):
            if eeg_idx < len(epoch_triggers) and exp_triggers[beh_idx] == epoch_triggers[eeg_idx]:
                eeg_idx += 1
            else:
                missing_indices.append(beh_idx)

        if len(missing_indices) == len(beh) - len(epochs) and eeg_idx == len(epoch_triggers):
            print(f"[INFO] Automatically identified {len(missing_indices)} dropped trigger(s) at behavioral index/indices: {missing_indices}")
            beh = beh.drop(index=missing_indices).reset_index(drop=True)
        else:
            raise ValueError(
                f"Could not automatically align metadata ({len(beh)} rows) with epochs ({len(epochs)} events). "
                f"Matched {eeg_idx}/{len(epoch_triggers)} triggers. Please inspect manually."
            )

    # append metadata to epochs
    epochs.metadata = beh
    
    # run AutoReject
    # By restricting the consensus to higher values, we force the algorithm to be more liberal
    # (a higher percentage of channels must be bad for the whole epoch to be rejected).
    # We also provide more n_interpolate options so it can opt to interpolate rather than reject.
    ar = autoreject.AutoReject(
        n_jobs=n_jobs, 
        random_state=42,
        consensus=[0.8, 0.85, 0.9, 0.95, 1.0],
        n_interpolate=[1, 4, 8, 16, 32]
    )
    epochs_ar, log = ar.fit_transform(epochs, return_log=True)
    # save epochs
    os.makedirs(f"{get_data_path()}derivatives/epoching/sci-{subject_id}/eeg/", exist_ok=True)
    epochs_ar.save(f"{get_data_path()}derivatives/epoching/sci-{subject_id}/eeg/sci-{subject_id}_task-spacecue-epo.fif",
                overwrite=True)
    # save the drop log
    log.save(f"{get_data_path()}derivatives/epoching/sci-{subject_id}/eeg/sci-{subject_id}_task-spacecue-epo_log.npz",
             overwrite=True)
             
    del epochs, epochs_ar, log, beh, ar

# --- Post-Preprocessing Data Concatenation ---
print("\n=========================================================")
print("   RUNNING POST-PREPROCESSING CONCATENATION   ")
print("=========================================================")

# Verify that processed subjects are registered in subjects.py
from SPACECUE.subjects import subject_ids as registered_subject_ids
unregistered = [s for s in subjects_to_process if s not in registered_subject_ids]
if unregistered:
    print(f"\n[WARNING] Subject(s) {unregistered} were processed but are NOT listed in SPACECUE/subjects.py!")
    print(f"          They will NOT be included in the concatenated EEG/behavioral datasets until added to subjects.py.\n")

try:
    print("\n--- Ensuring Behavioral Data is Clean & Formatted for All Subjects ---")
    from SPACECUE.clean_behavioral_data import clean_all_behavioral_data
    clean_all_behavioral_data(overwrite=False)
except Exception as e:
    print(f"Error checking behavioral data: {e}")

try:
    print("\n--- Concatenating EEG Data ---")
    from SPACECUE import concatenate_eeg_and_save
    concatenate_eeg_and_save('spacecue', overwrite=True, create_tfr_if_missing=False)
except Exception as e:
    print(f"Error concatenating EEG data: {e}")

try:
    print("\n--- Concatenating Behavioral Data ---")
    from SPACECUE.concatenate_behavioral_data import main as concat_beh_main
    concat_beh_main()
except Exception as e:
    print(f"Error concatenating behavioral data: {e}")

print("\n=========================================================")
print("   PREPROCESSING AND CONCATENATION FINISHED   ")
print("=========================================================")
