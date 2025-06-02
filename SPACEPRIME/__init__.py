import os
import socket
import mne
from SPACEPRIME.subjects import subject_ids
import glob


# --- Configuration for filenames ---
# These define the expected names for your processed data files.
CONCATENATED_EPOCHS_FILENAME = "concatenated_epochs-epo.fif"
CONCATENATED_TFR_FILENAME = "concatenated_tfr-tfr.h5"  # Adjust if using different TFR file type or naming


# --- Path Utilities ---
def get_data_path():
    """
    Determines the data path based on the operating system and hostname.
    (This function is from your existing code.)
    """
    if os.name == 'nt':
        data_path = 'G:\\Meine Ablage\\PhD\\data\\SPACEPRIME\\'
    elif os.name == 'posix':
        hostname = socket.gethostname()
        if "rechenknecht" in hostname:
            data_path = '/home/maxschulz/IPSY1-Storage/Projects/ac/Experiments/running_studies/SPACEPRIME/'
        elif 'MaxPC' in hostname:
            data_path = '/home/max/Insync/schulz.max5@gmail.com/GoogleDrive/PhD/data/SPACEPRIME/'
        else:
            raise OSError(f"Unknown Linux machine: {hostname}")
    else:
        raise OSError("Unsupported operating system.")
    return data_path


def _get_concatenated_folder_path(create_if_not_exists=True):
    """
    Gets the path to the 'concatenated' directory and optionally creates it.
    Helper function.
    """
    data_path = get_data_path()
    concatenated_folder = os.path.join(data_path, "concatenated")
    if create_if_not_exists:
        os.makedirs(concatenated_folder, exist_ok=True)
    return concatenated_folder


# --- Data Existence Check ---
def check_if_concatenated_eeg_exists():
    """
    Checks for the existence of the 'concatenated' directory and
    the pre-defined concatenated epochs and TFR data files within it.

    Returns:
        tuple: (
            concatenated_folder_exists (bool),
            epochs_file_exists (bool),
            tfr_file_exists (bool),
            epochs_file_path (str),
            tfr_file_path (str)
        )
    """
    data_path = get_data_path()
    concatenated_eeg_folder = os.path.join(data_path, "derivatives", "concatenated")

    # Define file paths regardless of folder existence for consistent return
    epochs_file_path = os.path.join(concatenated_eeg_folder, CONCATENATED_EPOCHS_FILENAME)
    tfr_file_path = os.path.join(concatenated_eeg_folder, CONCATENATED_TFR_FILENAME)

    if not os.path.isdir(concatenated_eeg_folder):
        return False, False, False, epochs_file_path, tfr_file_path

    epochs_exist = os.path.exists(epochs_file_path)
    tfr_exist = os.path.exists(tfr_file_path)

    return True, epochs_exist, tfr_exist, epochs_file_path, tfr_file_path


# --- Data Creation Logic (Placeholders) ---
def _create_and_save_concatenated_epochs(epochs_output_path):
    print(f"INFO: Attempting to create and save concatenated epochs to {epochs_output_path}...")
    # --- Subject Loop ---
    all_epochs = list()
    for subject in subject_ids:
        print(f"\n--- Loading Subject: {subject} ---")
        try:
            epoch_file_pattern = os.path.join(get_data_path(), "derivatives", "epoching", f"sub-{subject}", "eeg",
                                              f"sub-{subject}_task-spaceprime-epo.fif")
            epoch_files = glob.glob(epoch_file_pattern)
            if not epoch_files:
                print(f"  Epoch file not found for subject {subject} using pattern: {epoch_file_pattern}. Skipping.")
                continue
            epochs_sub = mne.read_epochs(epoch_files[0], preload=False)
            print(f"  Loaded {len(epochs_sub)} epochs.")
        except Exception as e:
            print(f"  Error loading data for subject {subject}: {e}. Skipping.")
            continue
        all_epochs.append(epochs_sub)
        if not all_epochs:
             raise RuntimeError(f"ERROR: No epochs data found/processed to concatenate for {epochs_output_path}.")
    if all_epochs: # Only proceed if there's something to concatenate
        concatenated_epochs = mne.concatenate_epochs(all_epochs)
        concatenated_epochs.save(epochs_output_path, overwrite=True)
        print(f"INFO: Successfully saved concatenated epochs to {epochs_output_path}.")
    else:
        print(f"INFO: No epochs to concatenate. File not created: {epochs_output_path}")


def _create_and_save_concatenated_tfr(tfr_output_path, source_epochs_path):
    import numpy as np
    print(f"INFO: Attempting to create and save concatenated TFR to {tfr_output_path}...")
    # --- BEGIN MNE-PYTHON TFR COMPUTATION AND SAVING LOGIC ---
    # Example (replace with your actual TFR parameters and processing):
    #
    if not os.path.exists(source_epochs_path):
       print(f"ERROR: Cannot create TFRs, concatenated epochs file not found: {source_epochs_path}")
       return # Or raise error

    epochs = mne.read_epochs(source_epochs_path)
    freqs = np.arange(1, 31, 1)  # Example: 8-29 Hz in 1 Hz steps
    n_cycles = freqs / 2.0       # Example: Morlet wavelet cycles

    # Compute TFR (e.g., Morlet, multitaper) - choose one and average if needed
    # For AverageTFR (if you average across epochs during computation):
    power = epochs.compute_tfr(method='morlet', freqs=freqs, n_cycles=n_cycles, return_itc=False,
                               average=False)
    power.save(tfr_output_path, overwrite=True, output="complex") # Saves AverageTFR object
    print(f"INFO: Successfully saved concatenated TFR to {tfr_output_path}.")

def concatenate_eeg_and_save(create_epochs_if_missing=True, create_tfr_if_missing=True):
    """
    Creates and saves concatenated EEG epochs and/or TFR data if they are missing.
    Saves them into the 'concatenated' folder.

    Args:
        create_epochs_if_missing (bool): If True, create epochs if they don't exist.
        create_tfr_if_missing (bool): If True, create TFR if it doesn't exist.

    Returns:
        tuple: (
            epochs_file_path (str),
            tfr_file_path (str),
            epochs_available_after_op (bool),
            tfr_available_after_op (bool)
        )
    """
    concatenated_folder = _get_concatenated_folder_path(create_if_not_exists=True)

    epochs_fpath = os.path.join(concatenated_folder, CONCATENATED_EPOCHS_FILENAME)
    tfr_fpath = os.path.join(concatenated_folder, CONCATENATED_TFR_FILENAME)

    epochs_available = os.path.exists(epochs_fpath)
    tfr_available = os.path.exists(tfr_fpath)

    if create_epochs_if_missing and not epochs_available:
        print(f"INFO: Concatenated epochs file ('{CONCATENATED_EPOCHS_FILENAME}') not found. Attempting creation.")
        try:
            _create_and_save_concatenated_epochs(epochs_fpath)
            epochs_available = os.path.exists(epochs_fpath)  # Re-check
            if not epochs_available:
                print(f"ERROR: Failed to create/verify concatenated epochs file at {epochs_fpath}.")
        except Exception as e:
            print(f"ERROR: An exception occurred during concatenated epochs creation: {e}")
            epochs_available = False  # Ensure it's marked as unavailable on error

    if create_tfr_if_missing and not tfr_available:
        print(f"INFO: Concatenated TFR file ('{CONCATENATED_TFR_FILENAME}') not found. Attempting creation.")
        if not epochs_available:  # Check dependency on epochs
            print(
                f"WARNING: Cannot create TFRs because the concatenated epochs file ('{CONCATENATED_EPOCHS_FILENAME}') is missing or failed to create. Skipping TFR creation.")
        else:
            try:
                _create_and_save_concatenated_tfr(tfr_fpath, epochs_fpath)
                tfr_available = os.path.exists(tfr_fpath)  # Re-check
                if not tfr_available:
                    print(f"ERROR: Failed to create/verify concatenated TFR file at {tfr_fpath}.")
            except Exception as e:
                print(f"ERROR: An exception occurred during concatenated TFR creation: {e}")
                tfr_available = False  # Ensure it's marked as unavailable on error

    return epochs_fpath, tfr_fpath, epochs_available, tfr_available


# --- Data Loading Logic (Placeholders) ---
def _load_concatenated_epochs_data(epochs_file_path):
    """
    TODO: Implement MNE-Python logic to load concatenated epochs.
    This is a placeholder.
    """
    print(f"INFO: Loading concatenated epochs from {epochs_file_path}...")
    # --- BEGIN MNE-PYTHON EPOCH LOADING LOGIC ---
    # try:
    #     # Preload true/false based on typical usage patterns for this data.
    #     epochs_data = mne.read_epochs(epochs_file_path, preload=True)
    #     print(f"INFO: Successfully loaded epochs from {epochs_file_path}.")
    #     return epochs_data
    # except Exception as e:
    #     print(f"ERROR: Failed to load epochs from {epochs_file_path}: {e}")
    #     return None # Or re-raise, depending on desired error handling
    # --- END MNE-PYTHON EPOCH LOADING LOGIC ---
    print("INFO: (Placeholder) Returning dummy epochs data.")
    return f"[[Dummy Epochs Object from {os.path.basename(epochs_file_path)}]]"


def _load_concatenated_tfr_data(tfr_file_path):
    """
    TODO: Implement MNE-Python logic to load concatenated TFR data.
    This is a placeholder.
    """
    print(f"INFO: Loading concatenated TFR from {tfr_file_path}...")
    # --- BEGIN MNE-PYTHON TFR LOADING LOGIC ---
    # try:
    #     # mne.time_frequency.read_tfrs usually returns a list of TFR objects.
    #     # If you save a single AverageTFR, it will be the first (and only) element.
    #     tfr_objects_list = mne.time_frequency.read_tfrs(tfr_file_path)
    #     if tfr_objects_list:
    #         tfr_data = tfr_objects_list[0] # Assuming one TFR object per file
    #         print(f"INFO: Successfully loaded TFR from {tfr_file_path}.")
    #         return tfr_data
    #     else:
    #         print(f"ERROR: No TFR data found in file {tfr_file_path}.")
    #         return None
    # except Exception as e:
    #     print(f"ERROR: Failed to load TFR from {tfr_file_path}: {e}")
    #     return None # Or re-raise
    # --- END MNE-PYTHON TFR LOADING LOGIC ---
    print("INFO: (Placeholder) Returning dummy TFR data.")
    return f"[[Dummy TFR Object from {os.path.basename(tfr_file_path)}]]"


# ======== Main execution block for __init__.py ========
# This code runs when the SPACEPRIME package is imported.

print(f"INFO: Initializing SPACEPRIME package...")

# 1. Determine initial state of concatenated data
_conc_folder_exists, _epochs_init_exist, _tfr_init_exist, \
    _epochs_fpath_on_check, _tfr_fpath_on_check = check_if_concatenated_eeg_exists()

# 2. Determine if creation is needed
_create_needed_for_epochs = not _epochs_init_exist
_create_needed_for_tfr = not _tfr_init_exist

# Initialize final paths and availability with initial check results
final_epochs_file_path = _epochs_fpath_on_check
final_tfr_file_path = _tfr_fpath_on_check
are_epochs_available = _epochs_init_exist
is_tfr_available = _tfr_init_exist

# 3. If data is missing, attempt to create it
if _create_needed_for_epochs or _create_needed_for_tfr:
    if _create_needed_for_epochs:
        print(f"INFO: Concatenated epochs data ('{CONCATENATED_EPOCHS_FILENAME}') needs to be created.")
    if _create_needed_for_tfr:
        print(f"INFO: Concatenated TFR data ('{CONCATENATED_TFR_FILENAME}') needs to be created.")

    _epath, _tpath, _epok, _tpok = concatenate_eeg_and_save(
        create_epochs_if_missing=_create_needed_for_epochs,
        create_tfr_if_missing=_create_needed_for_tfr
    )

    final_epochs_file_path = _epath
    final_tfr_file_path = _tpath
    are_epochs_available = _epok
    is_tfr_available = _tpok
else:
    if _conc_folder_exists:  # Only print this if the folder and files were already there
        print(f"INFO: Concatenated epochs and TFR data files already exist.")
    # If folder didn't exist, the create_needed flags would be true, so this 'else' branch isn't hit.

# 4. Load the data into package-level variables for default access
concatenated_epochs_data = None
concatenated_tfr_data = None

print(f"INFO: Attempting to load default SPACEPRIME datasets...")

if are_epochs_available:
    try:
        concatenated_epochs_data = _load_concatenated_epochs_data(final_epochs_file_path)
        if concatenated_epochs_data is None:
            print(
                f"WARNING: Loading concatenated epochs from '{final_epochs_file_path}' resulted in None (loader function might have failed).")
    except Exception as e:
        print(f"ERROR: Exception during final loading of epochs data from '{final_epochs_file_path}': {e}")
else:
    print(
        f"WARNING: Concatenated epochs data ('{CONCATENATED_EPOCHS_FILENAME}') is not available for loading. `concatenated_epochs_data` will be None.")

if is_tfr_available:
    try:
        concatenated_tfr_data = _load_concatenated_tfr_data(final_tfr_file_path)
        if concatenated_tfr_data is None:
            print(
                f"WARNING: Loading concatenated TFR from '{final_tfr_file_path}' resulted in None (loader function might have failed).")
    except Exception as e:
        print(f"ERROR: Exception during final loading of TFR data from '{final_tfr_file_path}': {e}")
else:
    print(
        f"WARNING: Concatenated TFR data ('{CONCATENATED_TFR_FILENAME}') is not available for loading. `concatenated_tfr_data` will be None.")

print(f"INFO: SPACEPRIME package data initialization complete.")
# To use the data after importing SPACEPRIME:
# import SPACEPRIME
# epochs = SPACEPRIME.concatenated_epochs_data
# tfr = SPACEPRIME.concatenated_tfr_data