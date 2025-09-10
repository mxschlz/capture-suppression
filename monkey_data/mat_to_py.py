import scipy.io
import h5py
import numpy as np
import pandas as pd  # For DataFrame
import mne  # For MNE Epochs object


def load_matlab_file(mat_file_path):
    """
    Loads data from a .mat file, attempting h5py for v7.3 files if scipy fails.

    Args:
        mat_file_path (str): The path to the .mat file.

    Returns:
        dict or h5py.File: A dictionary (for older .mat versions) or an h5py File object
                             (for v7.3+ .mat files) containing the data.
                             Returns None if the file cannot be loaded.
    """
    try:
        # Try with scipy.io.loadmat first
        mat_data = scipy.io.loadmat(mat_file_path)
        print(f"Successfully loaded with scipy.io.loadmat: {mat_file_path}")
        return mat_data
    except NotImplementedError:  # This is often the error for v7.3 files
        print(f"scipy.io.loadmat failed (likely v7.3+ file). Trying h5py for: {mat_file_path}")
        try:
            # h5py returns a file object, not a dict directly like loadmat.
            mat_data = h5py.File(mat_file_path, 'r')
            print(f"Successfully loaded with h5py: {mat_file_path}")
            return mat_data  # Return the h5py file object
        except Exception as e_h5:
            print(f"An error occurred while loading with h5py {mat_file_path}: {e_h5}")
            return None
    except FileNotFoundError:
        print(f"Error: File not found at {mat_file_path}")
        return None
    except Exception as e:
        print(f"An error occurred while loading {mat_file_path}: {e}")
        return None


def extract_hdf5_data(hdf5_object, file_handle):
    """
    Recursively extracts data from an HDF5 object (Group or Dataset),
    handling object references and converting to Python-friendly types.

    Args:
        hdf5_object: The current h5py Group, Dataset, or Reference to process.
        file_handle: The main h5py.File object, needed for dereferencing.

    Returns:
        dict, np.ndarray, or other Python types:
            - dict for HDF5 Groups.
            - np.ndarray for HDF5 Datasets (numeric, string, object arrays).
            - Dereferenced data for HDF5 References.
            - Python strings for HDF5 string data.
    """
    if isinstance(hdf5_object, h5py.Dataset):
        data = hdf5_object[()]  # Read the dataset's data

        # Case 1: Dataset is an array of HDF5 references (e.g., MATLAB cell array)
        if data.dtype == h5py.ref_dtype:
            if data.shape == ():  # Scalar reference
                # 'data' is the h5py.Reference object itself
                return extract_hdf5_data(file_handle[data], file_handle)
            else:  # Array of references
                extracted_array = np.empty(data.shape, dtype=object)
                for index in np.ndindex(data.shape):
                    ref_obj = data[index]  # This is an h5py.Reference object
                    dereferenced_item = file_handle[ref_obj]
                    extracted_array[index] = extract_hdf5_data(dereferenced_item, file_handle)
                return extracted_array

        # Case 2: Dataset is a structured array (e.g., MATLAB struct array)
        elif data.dtype.fields is not None:
            # Helper to process one element of a structured array (a single struct)
            def process_struct_element(struct_item):
                item_dict = {}
                for field_name in struct_item.dtype.names:
                    field_value = struct_item[field_name]

                    if isinstance(field_value, h5py.Reference):
                        item_dict[field_name] = extract_hdf5_data(file_handle[field_value], file_handle)
                    elif isinstance(field_value, np.ndarray) and field_value.dtype == h5py.ref_dtype:
                        # Field contains an array of references
                        if field_value.shape == ():  # Scalar reference in field (as 0-d array)
                            ref_in_field = field_value.item()
                            item_dict[field_name] = extract_hdf5_data(file_handle[ref_in_field], file_handle)
                        else:  # Array of references in field
                            field_extracted_array = np.empty(field_value.shape, dtype=object)
                            for idx_field in np.ndindex(field_value.shape):
                                ref_in_field = field_value[idx_field]
                                deref_item_in_field = file_handle[ref_in_field]
                                field_extracted_array[idx_field] = extract_hdf5_data(deref_item_in_field, file_handle)
                            item_dict[field_name] = field_extracted_array
                    elif isinstance(field_value, (h5py.Dataset, h5py.Group)):  # Field is an HDF5 object itself (rare)
                        item_dict[field_name] = extract_hdf5_data(field_value, file_handle)
                    else:  # Field is a simple type or another numpy array
                        item_dict[
                            field_name] = field_value  # Assume already Python-friendly or will be handled if it's an object array
                return item_dict

            if data.shape == ():  # Scalar structured array
                return process_struct_element(data)
            else:  # Array of structured arrays
                # Convert to list of dicts, then to object array
                processed_list = [process_struct_element(data[index]) for index in np.ndindex(data.shape)]
                return np.array(processed_list, dtype=object).reshape(data.shape)

        # Case 3: Dataset is a NumPy object array (might contain HDF5 references or other Python objects)
        elif data.dtype == object:
            if data.shape == ():
                item = data.item()
                if isinstance(item, h5py.Reference):
                    return extract_hdf5_data(file_handle[item], file_handle)
                elif isinstance(item, (h5py.Dataset, h5py.Group)):  # Should be rare here
                    return extract_hdf5_data(item, file_handle)
                return item  # Already a Python object
            else:  # Array of objects
                processed_array = np.empty(data.shape, dtype=object)
                for index in np.ndindex(data.shape):
                    element = data[index]
                    if isinstance(element, h5py.Reference):
                        processed_array[index] = extract_hdf5_data(file_handle[element], file_handle)
                    elif isinstance(element, (h5py.Dataset, h5py.Group)):  # Should be rare here
                        processed_array[index] = extract_hdf5_data(element, file_handle)
                    else:
                        processed_array[index] = element
                return processed_array

        # Case 4: Other dataset types (e.g., numeric, boolean, string)
        else:
            # Handle string decoding for S-type arrays (often from MATLAB char arrays)
            if data.dtype.kind == 'S':  # Byte strings
                if data.shape == ():
                    scalar_item = data.item()
                    return scalar_item.decode('utf-8').rstrip('\x00') if isinstance(scalar_item, bytes) else scalar_item
                else:
                    # For arrays of S-type, decode each element to Python string
                    str_array = np.empty(data.shape, dtype=object)
                    for index in np.ndindex(data.shape):
                        byte_string = data[index]
                        try:
                            str_array[index] = byte_string.decode('utf-8').rstrip('\x00')
                        except (AttributeError, UnicodeDecodeError):  # If not bytes or decode fails
                            str_array[index] = byte_string
                    return str_array
            # For other simple types (numbers, booleans, numpy unicode 'U'), return directly
            return data

    elif isinstance(hdf5_object, h5py.Group):
        group_data = {}
        for name, member in hdf5_object.items():
            group_data[name] = extract_hdf5_data(member, file_handle)
        return group_data

    elif isinstance(hdf5_object, h5py.Reference):
        # This handles direct calls with a reference object
        dereferenced_obj = file_handle[hdf5_object]
        return extract_hdf5_data(dereferenced_obj, file_handle)

    else:
        # Should not happen if input is from h5py object iteration
        print(f"Warning: Encountered unhandled object type: {type(hdf5_object)}")
        return hdf5_object  # Return as is, or None


# --- Main execution block ---
if __name__ == "__main__":
    input_file_path = 'G:\\Meine Ablage\\PhD\\data\\monkey_data\\DATA\\Processed_Data\\M1\\M1_20141112_1\\M1_20141112_1.mat'
    data_object = load_matlab_file(input_file_path)
    extracted_data = None

    if isinstance(data_object, h5py.File):
        extracted_data = {}
        print(f"\nProcessing HDF5 file: {input_file_path}")
        if "Data" in data_object:
            print("Extracting data from /Data...")
            extracted_data["Data"] = extract_hdf5_data(data_object["Data"], data_object)
            print("Finished extracting /Data.")
        if "LOG" in data_object and "Trial" in data_object["LOG"]:
            print("Extracting data from /LOG/Trial...")
            extracted_data["LOG_Trial"] = extract_hdf5_data(data_object["LOG"]["Trial"], data_object)
            print("Finished extracting /LOG/Trial.")
        print("\nExtraction complete.")
        # data_object.close()
        # print("HDF5 file closed.")

    elif isinstance(data_object, dict):
        print("File was loaded as a dictionary. No HDF5 extraction needed.")
        extracted_data = data_object
    else:
        print("File could not be loaded or data_object is None.")

    # --- Convert to Pandas DataFrame and MNE Epochs ---
    if extracted_data:
        # 1. Convert extracted_data["LOG_Trial"] to Pandas DataFrame
        if "LOG_Trial" in extracted_data and extracted_data["LOG_Trial"]:

            def unwrap_nested_scalar(item):
                """Recursively unwraps a scalar value potentially nested in lists/arrays."""
                # Keep unwrapping as long as the item is a list or numpy array with only one element
                while isinstance(item, (list, np.ndarray)) and len(item) == 1:
                    item = item[0]
                # Handle cases where the final item might still be a 0-dim numpy array (e.g., np.array(5))
                if isinstance(item, np.ndarray) and item.ndim == 0:
                    return item.item()  # Extract the scalar value
                return item  # Return the unwrapped item (should be a scalar or non-singleton list/array)


            processed_log_trial_for_df = {}
            for k, v_col_original in extracted_data["LOG_Trial"].items():
                v_col = v_col_original

                # Ensure it's a numpy array for easier iteration, handling potential lists of lists etc.
                if not isinstance(v_col, np.ndarray):
                    try:
                        # Try converting to object array if it's a list of potentially mixed types
                        v_col = np.array(v_col, dtype=object)
                    except Exception:
                        print(
                            f"Warning: Could not convert column '{k}' to numpy array. Skipping deep unwrap for this column.")
                        processed_log_trial_for_df[k] = v_col  # Keep original structure if conversion fails
                        continue  # Move to next column

                # Apply the unwrap function to each element in the array
                # Use .flat to iterate over all elements regardless of original dimensions
                try:
                    unwrapped_elements = [unwrap_nested_scalar(item) for item in v_col.flat]
                    # Convert the list of unwrapped elements back to a numpy array
                    # This will typically result in a 1D array suitable for a DataFrame column
                    processed_log_trial_for_df[k] = np.array(unwrapped_elements)
                except Exception as e:
                    print(
                        f"Warning: Error processing column '{k}' for DataFrame: {e}. Keeping original data structure.")
                    processed_log_trial_for_df[k] = v_col  # Fallback to original if unwrap fails

            try:
                # Create DataFrame from the processed dictionary
                log_trial_df = pd.DataFrame(processed_log_trial_for_df)
                print("\nSuccessfully created Pandas DataFrame from LOG_Trial.")
                print("DataFrame head:\n", log_trial_df.head())
                # You can now save or use log_trial_df
                # log_trial_df.to_csv("log_trial_data.csv", index=False)
            except Exception as e:
                print(f"\nError creating DataFrame from processed LOG_Trial data: {e}")
                print("Processed keys and a sample of their data for DataFrame:")
                for k_df, v_df in processed_log_trial_for_df.items():
                    sample = v_df[0] if isinstance(v_df, np.ndarray) and v_df.size > 0 else v_df
                    print(f"  {k_df}: {sample} (type: {type(sample)})")

        # 2. Convert extracted_data["Data"]["Signal"] to MNE Epochs object
        if "Data" in extracted_data and extracted_data["Data"] and "Signal" in extracted_data["Data"]:
            signal_data = extracted_data["Data"]["Signal"]  # Expected shape: (n_epochs, n_channels, n_times)

            # --- You'll likely need to find or define these parameters ---
            # Sampling frequency (Hz)
            sfreq = extracted_data["Data"].get("Sampf",  # Try to get it from data
                                               extracted_data["Data"].get("sfreq", 1000))  # Default if not found

            # Channel names (list of strings)
            # Example: ch_names = ['EEG 001', 'EEG 002', ..., 'STIM 001']
            # Try to get it from data or define manually
            num_channels = signal_data.shape[1] if signal_data.ndim == 3 else 1
            ch_names = extracted_data["Data"].get("ChannelNames",
                                                  [f"Chan{i + 1}" for i in range(num_channels)])
            if isinstance(ch_names, np.ndarray):  # Ensure ch_names is a list of strings
                ch_names = ch_names.flatten().astype(str).tolist()
            if len(ch_names) != num_channels:  # Fallback if names don't match
                ch_names = [f"Chan{i + 1}" for i in range(num_channels)]

            # Channel types (e.g., 'eeg', 'meg', 'stim')
            # Example: ch_types = ['eeg'] * num_channels
            ch_types = extracted_data["Data"].get("ChannelTypes", ['eeg'] * num_channels)
            if len(ch_types) != num_channels:  # Fallback
                ch_types = ['misc'] * num_channels

            # Events array: (n_events, 3) -> [sample_index, previous_event_id, event_id]
            # If signal_data is already epoched (n_epochs, n_channels, n_times)
            n_epochs = signal_data.shape[0]
            n_samples_per_epoch = signal_data.shape[2]

            # Create a basic events array assuming epochs start at regular intervals
            # or use actual event timings if available from LOG_Trial
            # This is a placeholder; you'll need to create meaningful events.
            event_onsets_samples = np.arange(n_epochs) * n_samples_per_epoch  # Example: if epochs are contiguous
            event_ids = np.ones(n_epochs, dtype=int)  # Example: all events are type 1

            # If you have event onset times (in samples) from log_trial_df:
            # For example, if log_trial_df has a column 'TrialStartSample':
            # if 'TrialStartSample' in log_trial_df.columns:
            #     event_onsets_samples = log_trial_df['TrialStartSample'].to_numpy().astype(int)
            #     if len(event_onsets_samples) != n_epochs:
            #         print("Warning: Number of event onsets doesn't match number of epochs in Signal.")
            #         # Fallback or error handling
            # else:
            #     print("Warning: 'TrialStartSample' not in log_trial_df. Using synthetic event onsets.")

            events = np.column_stack((event_onsets_samples,
                                      np.zeros(n_epochs, dtype=int),  # previous event ID
                                      event_ids))  # event ID

            # Time offset of the first sample in epochs relative to the event (in seconds)
            tmin = extracted_data["Data"].get("tmin", 0.0)  # Default to 0 if not specified

            # Create MNE Info object
            try:
                info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
                print(f"\nCreated MNE Info object: {info}")

                # Create MNE EpochsArray
                # Ensure signal_data is (n_epochs, n_channels, n_times)
                if signal_data.ndim == 2:  # If (n_channels, n_times), assume 1 epoch
                    signal_data = signal_data[np.newaxis, :, :]

                if signal_data.shape[0] != events.shape[0]:
                    print(f"Warning: Number of epochs in signal_data ({signal_data.shape[0]}) "
                          f"does not match number of events ({events.shape[0]}). Adjusting events.")
                    # This is a common issue if events are not perfectly aligned or if signal_data
                    # represents continuous data that needs to be epoched differently.
                    # For this example, we'll truncate/adjust events if possible, or error.
                    # A more robust solution depends on the data's nature.
                    if signal_data.shape[0] < events.shape[0]:
                        events = events[:signal_data.shape[0], :]
                    else:  # signal_data.shape[0] > events.shape[0]
                        print(
                            "Error: More epochs in signal data than events. Cannot create EpochsArray without proper event definition.")
                        epochs_object = None  # Skip creation

                if signal_data.shape[0] == events.shape[0]:  # Proceed if counts match
                    epochs_object = mne.EpochsArray(signal_data, info, events=events, tmin=tmin, baseline=None)
                    print("\nSuccessfully created MNE EpochsArray object.")
                    print(epochs_object)
                    # You can now use the epochs_object, e.g., epochs_object.plot()
                else:
                    epochs_object = None
                    print("Could not create MNE EpochsArray due to mismatched epoch/event counts.")


            except Exception as e:
                print(f"\nError creating MNE Epochs object: {e}")
                epochs_object = None
        else:
            print("\nSignal data not found in extracted_data['Data']['Signal'] for MNE Epochs.")