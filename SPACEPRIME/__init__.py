import os
import socket
import mne
from SPACEPRIME.subjects import subject_ids
import glob
import numpy as np # Needed for _create_and_save_concatenated_tfr
import pandas as pd # Added for load_concatenated_csv

SUPPORTED_PARADIGMS = ['spaceprime', 'spaceprime_desc-csd', 'passive', 'flanker']

# --- Configuration for filenames (functions to generate names based on paradigm) ---

def _get_concatenated_epochs_filename(paradigm):
    """Generates the filename for a concatenated epochs file for a given paradigm."""
    # Using a consistent naming scheme, e.g., "concatenated_spaceprime-epo.fif"
    return f"concatenated_{paradigm}-epo.fif"


def _get_concatenated_tfr_filename(paradigm):
    """Generates the filename for a concatenated TFR file for a given paradigm."""
    return f"concatenated_{paradigm}-tfr.h5"

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
def check_if_concatenated_eeg_exists(paradigm):
    """
    Checks for the existence of the 'concatenated' directory and
    the concatenated epochs and TFR data files for a given paradigm.

    Args:
        paradigm (str): The paradigm to check for (e.g., 'spaceprime', 'passive').
    Returns:
        tuple: (
            concatenated_folder_exists (bool),
            epochs_file_exists (bool),
            tfr_file_exists (bool),
            epochs_file_path (str),
            tfr_file_path (str)
        )
    """
    if paradigm not in SUPPORTED_PARADIGMS:
        raise ValueError(f"Paradigm '{paradigm}' is not supported. Choose from: {SUPPORTED_PARADIGMS}")

    data_path = get_data_path()
    concatenated_eeg_folder = os.path.join(data_path, "concatenated")

    # Define file paths regardless of folder existence for consistent return
    epochs_filename = _get_concatenated_epochs_filename(paradigm)
    tfr_filename = _get_concatenated_tfr_filename(paradigm)
    epochs_file_path = os.path.join(concatenated_eeg_folder, epochs_filename)
    tfr_file_path = os.path.join(concatenated_eeg_folder, tfr_filename)

    folder_exists = os.path.isdir(concatenated_eeg_folder)
    epochs_exist = os.path.exists(epochs_file_path)
    tfr_exist = os.path.exists(tfr_file_path)

    return folder_exists, epochs_exist, tfr_exist, epochs_file_path, tfr_file_path


# --- Data Creation Logic ---
def _create_and_save_concatenated_epochs(paradigm, epochs_output_path):
    """Internal function to create and save concatenated epochs."""
    print(f"INFO: Attempting to create and save concatenated '{paradigm}' epochs to {epochs_output_path}...")
    all_epochs = list()
    for subject in subject_ids:
        print(f"\n--- Loading Subject: {subject} for paradigm: {paradigm} ---")
        try:
            # Adjust path pattern if needed based on your processing pipeline output
            epoch_file_pattern = os.path.join(get_data_path(), "derivatives", "epoching", f"sub-{subject}", "eeg",
                                              f"sub-{subject}_task-{paradigm}-epo.fif")
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


def concatenate_eeg_and_save(paradigm, create_epochs_if_missing=True, create_tfr_if_missing=True):
    """
    Creates and saves concatenated EEG epochs and/or TFR data if they are missing.
    Saves them into the 'concatenated' folder.

    Args:
        paradigm (str): The paradigm to process (e.g., 'spaceprime', 'passive').
        create_epochs_if_missing (bool): If True, create epochs if they don't exist.
        create_tfr_if_missing (bool): If True, create TFR if it doesn't exist.

    Returns:
        tuple: (
            epochs_file_path (str),
            tfr_file_path (str),
            epochs_available (bool),
            tfr_available (bool)
        )
    """
    if paradigm not in SUPPORTED_PARADIGMS:
        raise ValueError(f"Paradigm '{paradigm}' is not supported. Choose from: {SUPPORTED_PARADIGMS}")

    # Ensure the target folder exists
    _get_concatenated_folder_path(create_if_not_exists=True)

    # Get current status
    _, epochs_available_before, tfr_available_before, \
        epochs_fpath, tfr_fpath = check_if_concatenated_eeg_exists(paradigm)

    epochs_available_after = epochs_available_before
    tfr_available_after = tfr_available_before

    if create_epochs_if_missing and not epochs_available_before:
        print(f"INFO: Concatenated epochs file for paradigm '{paradigm}' ('{os.path.basename(epochs_fpath)}') not found. Attempting creation.")
        epochs_available_after = _create_and_save_concatenated_epochs(paradigm, epochs_fpath)
        if not epochs_available_after:
             print(f"ERROR: Concatenated epochs creation for paradigm '{paradigm}' failed.")


    # Check epochs availability again before attempting TFR creation
    # This is important if epochs creation failed above
    _, epochs_available_now, _, _, _ = check_if_concatenated_eeg_exists(paradigm)

    if create_tfr_if_missing and not tfr_available_before:
        print(f"INFO: Concatenated TFR file for paradigm '{paradigm}' ('{os.path.basename(tfr_fpath)}') not found. Attempting creation.")
        if not epochs_available_now:  # Check dependency on epochs
            print(
                f"WARNING: Cannot create TFRs for paradigm '{paradigm}' because its concatenated epochs file is missing or failed to create. Skipping TFR creation.")
            tfr_available_after = False # Ensure TFR is marked unavailable
        else:
            tfr_available_after = _create_and_save_concatenated_tfr(tfr_fpath, epochs_fpath)
            if not tfr_available_after:
                 print(f"ERROR: Concatenated TFR creation for paradigm '{paradigm}' failed.")

    # Final check after operations
    _, epochs_available_final, tfr_available_final, _, _ = check_if_concatenated_eeg_exists(paradigm)

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
def load_concatenated_epochs(paradigm):
    """
    Loads the concatenated epochs data for a given paradigm, if available.

    Checks for the file's existence before attempting to load.

    Args:
        paradigm (str): The paradigm to load (e.g., 'spaceprime', 'passive').

    Returns:
        mne.Epochs or None: The loaded epochs object, or None if the file is not available or loading fails.
    """
    if paradigm not in SUPPORTED_PARADIGMS:
        raise ValueError(f"Paradigm '{paradigm}' is not supported. Choose from: {SUPPORTED_PARADIGMS}")

    # Use the check function to get the current status and path
    _, epochs_available, _, epochs_file_path, _ = check_if_concatenated_eeg_exists(paradigm)

    if not epochs_available:
        print(f"WARNING: Concatenated epochs data for paradigm '{paradigm}' is not available for loading at {epochs_file_path}. Cannot load.")
        return None

    # Reuse the internal loading logic
    return _load_concatenated_epochs_data(epochs_file_path)


def load_concatenated_tfr(paradigm):
    """
    Loads the concatenated TFR data for a given paradigm, if available.

    Checks for the file's existence before attempting to load.

    Args:
        paradigm (str): The paradigm to load (e.g., 'spaceprime', 'passive').

    Returns:
        mne.time_frequency.AverageTFR or None: The loaded TFR object, or None if the file is not available or loading fails.
    """
    if paradigm not in SUPPORTED_PARADIGMS:
        raise ValueError(f"Paradigm '{paradigm}' is not supported. Choose from: {SUPPORTED_PARADIGMS}")

    # Use the check function to get the current status and path
    _, _, tfr_available, _, tfr_file_path = check_if_concatenated_eeg_exists(paradigm)

    if not tfr_available:
        print(f"WARNING: Concatenated TFR data for paradigm '{paradigm}' is not available for loading at {tfr_file_path}. Cannot load.")
        return None

    # Reuse the internal loading logic
    return _load_concatenated_tfr_data(tfr_file_path)

def load_concatenated_csv(filename, **kwargs):
    """
    Loads a specified CSV file from the 'concatenated' folder.

    Args:
        filename (str): The name of the CSV file (e.g., "my_data.csv").

    Returns:
        pandas.DataFrame or None: The loaded DataFrame, or None if the file
                                   is not found or loading fails.
    """
    concatenated_folder = _get_concatenated_folder_path(create_if_not_exists=False)
    csv_file_path = os.path.join(concatenated_folder, filename)

    if not os.path.exists(csv_file_path):
        print(f"WARNING: CSV file ('{filename}') not found at {csv_file_path}. Cannot load.")
        return None

    print(f"INFO: Loading CSV file from {csv_file_path}...")
    try:
        df = pd.read_csv(csv_file_path, **kwargs)
        print(f"INFO: Successfully loaded CSV file '{filename}'.")
        return df
    except Exception as e:
        print(f"ERROR: Failed to load CSV file '{filename}' from {csv_file_path}: {e}")
        return None


# On import, this package does not automatically create or load any data.
# You must explicitly call the creation or loading functions.
print(f"INFO: Supported paradigms are: {SUPPORTED_PARADIGMS}")

print("\nINFO: To load data for a specific paradigm (e.g., 'passive'), use:")
print("INFO:   epochs = SPACEPRIME.load_concatenated_epochs('passive')")
print("INFO:   my_dataframe = SPACEPRIME.load_concatenated_csv('your_file.csv')")

print("\nINFO: To create missing files for a paradigm, use:")
print("INFO:   SPACEPRIME.concatenate_eeg_and_save('flanker')")
