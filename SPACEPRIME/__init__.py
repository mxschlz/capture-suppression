import os
import socket
import mne
from SPACEPRIME.subjects import subject_ids
import glob
import numpy as np # Needed for _create_and_save_concatenated_tfr

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
    # Assuming concatenated files go into 'concatenated' based on check_if_concatenated_eeg_exists
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
    concatenated_eeg_folder = os.path.join(data_path, "concatenated")

    # Define file paths regardless of folder existence for consistent return
    epochs_file_path = os.path.join(concatenated_eeg_folder, CONCATENATED_EPOCHS_FILENAME)
    tfr_file_path = os.path.join(concatenated_eeg_folder, CONCATENATED_TFR_FILENAME)

    folder_exists = os.path.isdir(concatenated_eeg_folder)
    epochs_exist = os.path.exists(epochs_file_path)
    tfr_exist = os.path.exists(tfr_file_path)

    return folder_exists, epochs_exist, tfr_exist, epochs_file_path, tfr_file_path


# --- Data Creation Logic ---
def _create_and_save_concatenated_epochs(epochs_output_path):
    """Internal function to create and save concatenated epochs."""
    print(f"INFO: Attempting to create and save concatenated epochs to {epochs_output_path}...")
    all_epochs = list()
    for subject in subject_ids:
        print(f"\n--- Loading Subject: {subject} ---")
        try:
            # Adjust path pattern if needed based on your processing pipeline output
            epoch_file_pattern = os.path.join(get_data_path(), "derivatives", "epoching", f"sub-{subject}", "eeg",
                                              f"sub-{subject}_task-spaceprime-epo.fif")
            epoch_files = glob.glob(epoch_file_pattern)
            if not epoch_files:
                print(f"  Epoch file not found for subject {subject} using pattern: {epoch_file_pattern}. Skipping.")
                continue
            # Assuming you want to concatenate raw epochs before any subject-specific averaging
            epochs_sub = mne.read_epochs(epoch_files[0], preload=False) # Use preload=False initially
            print(f"  Loaded {len(epochs_sub)} epochs.")
            all_epochs.append(epochs_sub)
        except Exception as e:
            print(f"  Error loading data for subject {subject}: {e}. Skipping.")
            continue

    if not all_epochs:
         print(f"ERROR: No epochs data found/processed to concatenate for {epochs_output_path}. File not created.")
         return False # Indicate failure

    try:
        # Preload data before concatenating if memory allows, or concatenate without preloading
        # and then preload the result if needed later. Concatenating without preloading is safer for large datasets.
        concatenated_epochs = mne.concatenate_epochs(all_epochs)
        concatenated_epochs.save(epochs_output_path, overwrite=True)
        print(f"INFO: Successfully saved concatenated epochs to {epochs_output_path}.")
        return True # Indicate success
    except Exception as e:
        print(f"ERROR: An exception occurred during concatenation or saving: {e}")
        return False # Indicate failure


def _create_and_save_concatenated_tfr(tfr_output_path, source_epochs_path):
    """Internal function to create and save concatenated TFR."""
    print(f"INFO: Attempting to create and save concatenated TFR to {tfr_output_path}...")

    if not os.path.exists(source_epochs_path):
       print(f"ERROR: Cannot create TFRs, concatenated epochs file not found: {source_epochs_path}")
       return False # Indicate failure

    try:
        # Load epochs (preload=True might be needed for TFR computation)
        epochs = mne.read_epochs(source_epochs_path, preload=True)

        # --- BEGIN MNE-PYTHON TFR COMPUTATION AND SAVING LOGIC ---
        # Example (replace with your actual TFR parameters and processing):
        freqs = np.arange(5, 26, 1)  # Example: 5-25 Hz in 1 Hz steps
        n_cycles = freqs / 2.0       # Example: Morlet wavelet cycles

        # Compute TFR (e.g., Morlet, multitaper)
        # If you want AverageTFR, compute with average=True
        # If you want EpochsTFR, compute with average=False
        # The saving method depends on the output type. .save() works for both.
        print("INFO: Computing TFR...")
        power = epochs.compute_tfr(method='morlet', freqs=freqs, n_cycles=n_cycles, return_itc=False,
                                   average=True, # Set to True for AverageTFR, False for EpochsTFR
                                   decim=15, n_jobs=5) # Adjust decim and n_jobs as needed
        print("INFO: TFR computation complete. Saving...")

        power.save(tfr_output_path, overwrite=True) # Saves the TFR object
        print(f"INFO: Successfully saved concatenated TFR to {tfr_output_path}.")
        return True # Indicate success
    except Exception as e:
        print(f"ERROR: An exception occurred during TFR computation or saving: {e}")
        return False # Indicate failure


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
    # Ensure the target folder exists
    _get_concatenated_folder_path(create_if_not_exists=True)

    # Get current status
    _, epochs_available_before, tfr_available_before, \
        epochs_fpath, tfr_fpath = check_if_concatenated_eeg_exists()

    epochs_available_after = epochs_available_before
    tfr_available_after = tfr_available_before

    if create_epochs_if_missing and not epochs_available_before:
        print(f"INFO: Concatenated epochs file ('{CONCATENATED_EPOCHS_FILENAME}') not found. Attempting creation.")
        epochs_available_after = _create_and_save_concatenated_epochs(epochs_fpath)
        if not epochs_available_after:
             print(f"ERROR: Concatenated epochs creation failed.")


    # Check epochs availability again before attempting TFR creation
    # This is important if epochs creation failed above
    _, epochs_available_now, _, _, _ = check_if_concatenated_eeg_exists()


    if create_tfr_if_missing and not tfr_available_before:
        print(f"INFO: Concatenated TFR file ('{CONCATENATED_TFR_FILENAME}') not found. Attempting creation.")
        if not epochs_available_now:  # Check dependency on epochs
            print(
                f"WARNING: Cannot create TFRs because the concatenated epochs file ('{CONCATENATED_EPOCHS_FILENAME}') is missing or failed to create. Skipping TFR creation.")
            tfr_available_after = False # Ensure TFR is marked unavailable
        else:
            tfr_available_after = _create_and_save_concatenated_tfr(tfr_fpath, epochs_fpath)
            if not tfr_available_after:
                 print(f"ERROR: Concatenated TFR creation failed.")


    # Final check after operations
    _, epochs_available_final, tfr_available_final, _, _ = check_if_concatenated_eeg_exists()


    return epochs_fpath, tfr_fpath, epochs_available_final, tfr_available_final


# --- Data Loading Logic ---
def _load_concatenated_epochs_data(epochs_file_path):
    """Internal function to load concatenated epochs."""
    print(f"INFO: Loading concatenated epochs from {epochs_file_path}...")
    try:
        # Preload true/false based on typical usage patterns for this data.
        # Often you need data preloaded for plotting or further processing.
        epochs_data = mne.read_epochs(epochs_file_path, preload=True)
        print(f"INFO: Successfully loaded epochs from {epochs_file_path}.")
        return epochs_data
    except Exception as e:
        print(f"ERROR: Failed to load epochs from {epochs_file_path}: {e}")
        return None # Return None on error


def _load_concatenated_tfr_data(tfr_file_path):
    """Internal function to load concatenated TFR."""
    print(f"INFO: Loading concatenated TFR from {tfr_file_path}...")
    try:
        # mne.time_frequency.read_tfrs usually returns a list of TFR objects.
        # If you save a single AverageTFR, it will be the first (and only) element.
        tfr_objects_list = mne.time_frequency.read_tfrs(tfr_file_path)
        if tfr_objects_list:
            tfr_data = tfr_objects_list[0] # Assuming one TFR object per file
            print(f"INFO: Successfully loaded TFR from {tfr_file_path}.")
            return tfr_data
        else:
            print(f"ERROR: No TFR data found in file {tfr_file_path}.")
            return None
    except Exception as e:
        print(f"ERROR: Failed to load TFR from {tfr_file_path}: {e}")
        return None # Return None on error


# --- Public Loading Functions ---
def load_concatenated_epochs():
    """
    Loads the concatenated epochs data if available.

    Checks for the file's existence before attempting to load.

    Returns:
        mne.Epochs or None: The loaded epochs object, or None if the file is not available or loading fails.
    """
    # Use the check function to get the current status and path
    _, epochs_available, _, epochs_file_path, _ = check_if_concatenated_eeg_exists()

    if not epochs_available:
        print(f"WARNING: Concatenated epochs data ('{CONCATENATED_EPOCHS_FILENAME}') is not available for loading at {epochs_file_path}. Cannot load.")
        return None

    # Reuse the internal loading logic
    return _load_concatenated_epochs_data(epochs_file_path)


def load_concatenated_tfr():
    """
    Loads the concatenated TFR data if available.

    Checks for the file's existence before attempting to load.

    Returns:
        mne.time_frequency.AverageTFR or None: The loaded TFR object, or None if the file is not available or loading fails.
    """
    # Use the check function to get the current status and path
    _, _, tfr_available, _, tfr_file_path = check_if_concatenated_eeg_exists()

    if not tfr_available:
        print(f"WARNING: Concatenated TFR data ('{CONCATENATED_TFR_FILENAME}') is not available for loading at {tfr_file_path}. Cannot load.")
        return None

    # Reuse the internal loading logic
    return _load_concatenated_tfr_data(tfr_file_path)


# ======== Main execution block for __init__.py ========
# This code runs when the SPACEPRIME package is imported.

print(f"INFO: Initializing SPACEPRIME package...")

# 1. Determine initial state of concatenated data and store paths/availability
# These variables are now package-level and accessible after import
# They reflect the state *at the time of import*.
_conc_folder_exists, are_epochs_available, is_tfr_available, \
    final_epochs_file_path, final_tfr_file_path = check_if_concatenated_eeg_exists()

# 2. Do NOT automatically create or load data upon import.
# The user will call concatenate_eeg_and_save() to create files if missing,
# and load_concatenated_epochs() or load_concatenated_tfr() to load data when needed.

print(f"INFO: SPACEPRIME package initialization complete. Data files checked.")
print(f"INFO: Epochs file availability: {are_epochs_available}")
print(f"INFO: TFR file availability: {is_tfr_available}")
print(f"INFO: Epochs file path: {final_epochs_file_path}")
print(f"INFO: TFR file path: {final_tfr_file_path}")
print("\nINFO: To load data, use:")
print("INFO:   epochs = SPACEPRIME.load_concatenated_epochs()")
print("INFO:   tfr = SPACEPRIME.load_concatenated_tfr()")
print("\nINFO: To create missing files, use:")
print("INFO:   SPACEPRIME.concatenate_eeg_and_save()")
