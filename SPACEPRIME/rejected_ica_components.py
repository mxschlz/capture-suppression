import numpy as np
from SPACEPRIME import get_data_path, subjects
import mne


subs = subjects.subject_ids

ica_labels = {}
rejected_counts = []

for sub in subs:
    ica_label_path = f"{get_data_path()}derivatives\\preprocessing\\sub-{sub}\\eeg\\sub-{sub}_task-spaceprime_ica_labels.txt"
    try:
        with open(ica_label_path, 'r') as f:
            rejected = [int(line.strip()) for line in f if line.strip().isdigit()]

        ica_labels[sub] = rejected if rejected else None
        rejected_counts.append(len(rejected))
    except FileNotFoundError:
        ica_labels[sub] = None
        rejected_counts.append(0)

print(f"Mean rejected components: {np.mean(rejected_counts):.2f}")
print(f"Standard Deviation: {np.std(rejected_counts):.2f}")

ica_object = mne.preprocessing.read_ica(f"{get_data_path()}derivatives\\preprocessing\\sub-108\\eeg\\sub-108_task-spaceprime_ica.fif")