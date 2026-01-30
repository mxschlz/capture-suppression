import SPACEPRIME
from SPACEPRIME import subjects
import os
import glob
import pandas as pd

# Define path to raw data
data_root = SPACEPRIME.get_data_path()
raw_data_path = os.path.join(data_root, 'sourcedata', 'raw')
subs = subjects.subject_ids

# Find all accuracy test files
file_paths = []
for sub in subs:
    # Structure: sourcedata/raw/sub-XXX/beh/accuracy_test_sub-XXX_*.csv
    search_pattern = os.path.join(raw_data_path, f'sub-{sub}', 'beh', f'accuracy_test_sub-{sub}_*.csv')
    found_files = glob.glob(search_pattern)
    if found_files:
        file_paths.extend(found_files)
    else:
        print(f"Warning: No accuracy test file found for sub-{sub}")

print(f"Found {len(file_paths)} accuracy test files.")

df_list = []
for file_path in file_paths:
    try:
        df = pd.read_csv(file_path)
        
        # Extract subject ID from filename if not in columns
        # Filename example: accuracy_test_sub-108_February_12_2025_09_32_47.csv
        if 'subject_id' not in df.columns:
            filename = os.path.basename(file_path)
            # Find the part that starts with 'sub-'
            for part in filename.split('_'):
                if part.startswith('sub-'):
                    try:
                        subject_id = int(part.split('-')[1])
                        df['subject_id'] = subject_id
                    except (ValueError, IndexError):
                        continue
        
        df_list.append(df)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")

if df_list:
    all_accuracy_data = pd.concat(df_list, ignore_index=True)
    print("Data loaded successfully.")
    print(all_accuracy_data.head())

    # Calculate attempts per subject
    # Round is 0-indexed, so max(Round) + 1 gives the number of attempts
    attempts_per_subject = all_accuracy_data.groupby('subject_id')['Round'].max() + 1

    print(f"Mean attempts per subject: {attempts_per_subject.mean():.2f}")
    print(f"Standard deviation: {attempts_per_subject.std():.2f}")

else:
    print("No data found.")
    all_accuracy_data = pd.DataFrame()