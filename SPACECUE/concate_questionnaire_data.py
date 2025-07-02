import pandas as pd
import os
import glob  # Used to find files matching a pattern
from stats import remove_outliers


# --- Define subjects to exclude ---
# Using a set is highly efficient for checking if an item is present.
subjects_to_exclude = {13, 25, 37}

# --- 1. Concatenate all subject behavioral data ---

# Define the base path where subject folders are located
# Using a raw string (r"...") is good practice for Windows paths
base_path = r"G:\Meine Ablage\PhD\data\SPACECUE_behavioral_pilot\derivatives\preprocessing"

# Create a search pattern to find all the cleaned behavioral CSV files
# The '*' is a wildcard that matches any character
search_pattern = os.path.join(base_path, "sub-*", "beh", "sub-*_clean.csv")

# Use glob to find all file paths that match the pattern
all_files = glob.glob(search_pattern)

if not all_files:
    print(f"Warning: No files found matching the pattern: {search_pattern}")
    # Exit or handle the case where no files are found
    exit()

print(f"Found {len(all_files)} subject files to concatenate.")

# This list will hold the individual DataFrames before we combine them
df_list = []

for file_path in all_files:
    # --- Extract subject ID from the file path ---
    # We get the 'sub-XX' folder name to safely identify the subject
    try:
        sub_folder_name = os.path.basename(os.path.dirname(os.path.dirname(file_path)))
        # Split 'sub-XX' at the hyphen and take the second part ('XX')
        subject_id = int(sub_folder_name.split('-')[1])
    except (IndexError, ValueError):
        print(f"Could not extract a valid subject ID from path: {file_path}. Skipping this file.")
        continue

    # --- Check if the current subject should be excluded ---
    if subject_id in subjects_to_exclude:
        print(f"Excluding subject {subject_id} as requested.")
        continue  # Skip to the next file

    # Read the subject's data into a temporary DataFrame
    temp_df = pd.read_csv(file_path)

    # Add a 'subject_id' column. This is crucial for the final merge.
    temp_df['subject_id'] = subject_id

    # Add the prepared DataFrame to our list
    df_list.append(temp_df)

# Concatenate all DataFrames in the list into a single master DataFrame
if not df_list:
    print("No data to process after exclusions. Exiting.")
    exit()
df_behavioral = pd.concat(df_list, ignore_index=True)

# --- 2. Save the concatenated DataFrame ---

# Define the output directory and filename
output_dir = r"G:\Meine Ablage\PhD\data\SPACECUE_behavioral_pilot\concatenated"
output_file = os.path.join(output_dir, "all_subjects_behavioral_clean.csv")

# Create the output directory if it doesn't already exist
os.makedirs(output_dir, exist_ok=True)

# Save the concatenated data to a new CSV file
# index=False prevents pandas from writing a new, unnamed index column
df_behavioral.to_csv(output_file, index=False)
print(f"\nConcatenated data successfully saved to: {output_file}")

# --- 3. Merge with questionnaire data ---

# Load the raw demographic/questionnaire data
questionnaire_path = r"C:\Users\Max\Downloads\df_demogr_raw.csv"
qd = pd.read_csv(questionnaire_path)

# Rename the 'subject_number' column to 'subject_id' to match our other DataFrame
if 'subject_number' in qd.columns:
    qd = qd.rename(columns={"subject_number": "subject_id"})

# Merge the two DataFrames based on the common 'subject_id' column
# 'how='left'' ensures all behavioral data is kept, even if there's no matching questionnaire data
df_merged = pd.merge(df_behavioral, qd, on="subject_id", how='left')

# --- 4. NEW: Clean up duplicate columns after merging ---
# Find and drop columns with the '_y' suffix (from the right DataFrame)
cols_to_drop = [col for col in df_merged.columns if col.endswith('_y')]
cols_to_drop.append("duration")
cols_to_drop.append("Unnamed: 0")
df_merged = df_merged.drop(columns=cols_to_drop)
print(f"\nDropped {len(cols_to_drop)} duplicate columns with '_y' suffix: {cols_to_drop}")

# Find and rename columns with the '_x' suffix to remove the suffix
cols_to_rename = {col: col.removesuffix('_x') for col in df_merged.columns if col.endswith('_x')}
df_merged = df_merged.rename(columns=cols_to_rename)
print(f"Cleaned up {len(cols_to_rename)} column names with '_x' suffix.")

# 5. remove reaction time outliers
df_merged = remove_outliers(df_merged, column_name="rt", threshold=2)

# add age of participant 43
df_merged.loc[df_merged['subject_id'] == 43, 'age'] = 24

# --- 6. Save the final merged and cleaned DataFrame ---
final_merged_file = os.path.join(output_dir, "all_subjects_merged_data.csv")
df_merged.to_csv(final_merged_file, index=False)
print(f"\nFinal merged and cleaned data saved to: {final_merged_file}")

# Display the first few rows and info of the final merged DataFrame to verify
print("\n--- Merge and Clean Complete ---")
print("Head of the final merged DataFrame:")
print(df_merged.head())
print("\nInfo of the final DataFrame:")
df_merged.info()
