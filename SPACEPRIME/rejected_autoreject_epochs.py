from SPACEPRIME import get_data_path, subjects
import autoreject
import numpy as np

subs = subjects.subject_ids
rejected_counts = []

for sub in subs:
    log_path = f"{get_data_path()}derivatives\\epoching\\sub-{sub}\\eeg\\sub-{sub}_task-spaceprime-epo_log.npz"
    try:
        reject_log = autoreject.read_reject_log(log_path)
        rejected_counts.append(np.sum(reject_log.bad_epochs))
    except FileNotFoundError:
        rejected_counts.append(0)

print(f"Mean rejected epochs: {np.mean(rejected_counts):.2f}")
print(f"Standard Deviation: {np.std(rejected_counts):.2f}")
