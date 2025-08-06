import matplotlib.pyplot as plt
import SPACEPRIME
import mne
import os


plt.ion()


# load epochs
epochs = SPACEPRIME.load_concatenated_epochs("spaceprime")

# Define the root path for your derivatives.
# I'm using the path from your example. Please adjust if it's different.
# Using a raw string (r"...") is good practice for Windows paths.
derivatives_path = f"{SPACEPRIME.get_data_path()}derivatives\\epoching"

# Get a list of all unique subject IDs from the metadata
unique_subjects = epochs.metadata['subject_id'].unique()[1:]  # first value is nan

print(f"Found {len(unique_subjects)} subjects. Saving CSD epochs...")

# Loop through each subject
for subject_id in unique_subjects:
    # float to int
    subject_id = int(subject_id)
    # Select the epochs for the current subject
    subject_epochs = epochs[epochs.metadata['subject_id'] == subject_id]
    # csd transform
    subject_epochs_csd = mne.preprocessing.compute_current_source_density(subject_epochs)
    # Create the BIDS-compliant path and filename.
    # We add `desc-csd` to distinguish this file from the original epochs.
    subject_dir = os.path.join(derivatives_path, f"sub-{subject_id}", "eeg")
    output_fname = f"sub-{subject_id}_task-spaceprime_desc-csd-epo.fif"
    output_path = os.path.join(subject_dir, output_fname)

    # Create the directory for the subject if it doesn't already exist
    os.makedirs(subject_dir, exist_ok=True)

    # Save the subject's CSD epochs file
    print(f"  Saving to: {output_path}")
    subject_epochs_csd.save(output_path, overwrite=True)

print("\nAll subject CSD epochs have been saved successfully!")

