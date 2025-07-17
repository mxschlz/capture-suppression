import pandas as pd
from pathlib import Path
import SPACEPRIME
from SPACEPRIME.subjects import subject_ids

def evaluate_wnss_from_csv(file_path):
    """
    Loads WNSS data, calculates a normalized noise resistance score, and
    returns a DataFrame. The score ranges from 0 to 1, where higher values
    indicate higher noise resistance (i.e., lower noise sensitivity).
    """
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: WNSS file '{file_path}' not found.")
        return None

    reverse_scored_items = ['q1', 'q3', 'q8', 'q12', 'q14', 'q15', 'q20']
    df_scored = df.copy()

    for item in reverse_scored_items:
        if item in df_scored.columns:
            # Reverse score on a 6-point scale (1-6)
            df_scored[item] = 7 - df_scored[item]
        else:
            print(f"Warning: WNSS column '{item}' not found.")

    question_columns = [f'q{i}' for i in range(1, 22) if f'q{i}' in df_scored.columns]
    if len(question_columns) != 21:
        print(f"Warning: Expected 21 WNSS question columns, but found {len(question_columns)}. The score might be inaccurate.")

    # Calculate the raw sum of scores, representing noise sensitivity
    total_sensitivity_score = df_scored[question_columns].sum(axis=1)

    # Normalize the score based on the specified method.
    # The WNSS has 21 items, each scored on a 6-point scale (1-6).
    num_items = 21
    min_possible_score = num_items * 1  # Lowest possible sensitivity score
    max_possible_score = num_items * 6  # Highest possible sensitivity score
    score_range = max_possible_score - min_possible_score

    # Normalize the sensitivity score to a 0-1 range
    # Formula: (score - min_score) / (max_score - min_score)
    normalized_sensitivity = (total_sensitivity_score - min_possible_score) / score_range

    # Calculate noise resistance by inverting the normalized sensitivity.
    # Higher score = higher resistance.
    df['wnss_noise_resistance'] = 1 - normalized_sensitivity
    print("WNSS evaluation complete.")
    return df


def evaluate_ssq_from_csv(file_path):
    """
    Loads SSQ data, calculates mean scores for subscales, and returns a new DataFrame.
    The scores are normalized to a 0-1 scale based on the possible range of 0-10,
    with higher scores reflecting better self-reported hearing ability.
    """
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: SSQ file '{file_path}' not found.")
        return None

    # The SSQ scores range from 0 to 10. We normalize by dividing by 10.
    max_score = 10.0

    # Define columns for each subscale
    speech_cols = [col for col in df.columns if 'speechcompr' in col]
    spatial_cols = [col for col in df.columns if 'spatialhearing' in col]
    quality_cols = [col for col in df.columns if 'hearingqual' in col]

    # Calculate the mean for each subscale and normalize
    if speech_cols:
        df['ssq_speech_mean'] = df[speech_cols].mean(axis=1) / max_score
    if spatial_cols:
        df['ssq_spatial_mean'] = df[spatial_cols].mean(axis=1) / max_score
    if quality_cols:
        df['ssq_quality_mean'] = df[quality_cols].mean(axis=1) / max_score

    # Calculate the overall mean score and normalize
    all_ssq_cols = speech_cols + spatial_cols + quality_cols
    if all_ssq_cols:
        df['ssq_overall_mean'] = df[all_ssq_cols].mean(axis=1) / max_score

    print("SSQ evaluation complete.")
    return df


# --- 1. Define Paths ---
# IMPORTANT: Update this to the root directory of your project.
# The structure should be:
# DATA_ROOT/
#  - sourcedata/raw/sub-001/questionnaires/...
#  - sourcedata/raw/sub-002/questionnaires/...
data_root = Path(SPACEPRIME.get_data_path())

source_data_path = data_root / "sourcedata" / "raw"
derivatives_path = data_root / "derivatives"

# Create the derivatives folder if it doesn't exist
derivatives_path.mkdir(parents=True, exist_ok=True)

# --- 2. Process each subject ---
all_subject_results = []
print(f"Processing {len(subject_ids)} subjects from the subjects module. Starting evaluation...")

for subject_id in subject_ids:
    # Construct the path to the subject's source data directory
    subject_dir = Path(source_data_path / f"sub-{subject_id}")
    print(f"\n--- Processing {subject_id} ---")

    # Define paths to the questionnaire files for the current subject
    wnss_file = subject_dir / 'questionnaires' / f'wnss_results_sub-{subject_id}.csv'
    ssq_file = subject_dir / 'questionnaires' / f'ssq_results_sub-{subject_id}.csv'

    # --- 3. Evaluate and Merge ---
    if not (wnss_file.exists() and ssq_file.exists()):
        print(f"Warning: Missing questionnaire files for {subject_id}. Skipping.")
        continue

    wnss_results = evaluate_wnss_from_csv(wnss_file)
    ssq_results = evaluate_ssq_from_csv(ssq_file)

    if wnss_results is not None and ssq_results is not None:
        # Ensure 'subject_id' column exists for merging. Add it if missing.
        if 'subject_id' not in wnss_results.columns:
            wnss_results['subject_id'] = subject_id
        if 'subject_id' not in ssq_results.columns:
            ssq_results['subject_id'] = subject_id

        # Select only the necessary columns to keep the final dataframe clean
        wnss_cols = ['subject_id', 'wnss_noise_resistance']
        ssq_cols = ['subject_id', 'ssq_speech_mean', 'ssq_spatial_mean', 'ssq_quality_mean', 'ssq_overall_mean']

        wnss_scores = wnss_results[[c for c in wnss_cols if c in wnss_results.columns]]
        ssq_scores = ssq_results[[c for c in ssq_cols if c in ssq_results.columns]]

        # Merge the results for the current subject and add to our list
        subject_df = pd.merge(wnss_scores, ssq_scores, on='subject_id')
        all_subject_results.append(subject_df)

        # --- Save individual subject results ---
        # Create the subject-specific derivatives folder structure
        output_dir_per_subject = derivatives_path / "preprocessing" / f"sub-{subject_id}" / "questionnaires"
        output_dir_per_subject.mkdir(parents=True, exist_ok=True)

        # Define the output file path and save the dataframe
        output_file_per_subject = output_dir_per_subject / f"{subject_id}_questionnaires.csv"
        subject_df.to_csv(output_file_per_subject, index=False)
        print(f"Saved individual scores to {output_file_per_subject}")

# --- 4. Combine and Save Final Results ---
if all_subject_results:
    final_dataframe = pd.concat(all_subject_results, ignore_index=True)
    print("\n--- Combined Evaluation Results ---")
    print(final_dataframe.head(10))
    # save output
    concatenated_path = data_root / "concatenated"
    output_filename = concatenated_path / "combined_questionnaire_results.csv"
    final_dataframe.to_csv(output_filename, index=False)
    print(f"\nSuccessfully saved the combined results to '{output_filename}'")
else:
    print("\nProcessing complete, but no data was successfully evaluated.")
