import os
import sys
import glob
import shutil
from datetime import datetime
import pandas as pd
import numpy as np

# Ensure project root is in python path
try:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
except NameError:
    project_root = os.path.abspath(os.getcwd())
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import SPACECUE
from SPACECUE.subjects import subject_ids

def extract_demographics_from_clean_file(subject_id, candidate_files=None):
    """
    Extracts demographics (Age, Gender, Handedness) from an existing offline-formatted *_clean*.csv file.
    Searches candidate files, derivatives/preprocessing, legacy_clean, raw folders, and psychopy logs.
    """
    data_path = SPACECUE.get_data_path()
    search_files = []
    if candidate_files:
        if isinstance(candidate_files, str):
            search_files.append(candidate_files)
        else:
            search_files.extend(candidate_files)

    prep_dir = os.path.join(data_path, 'derivatives', 'preprocessing', f'sci-{subject_id}', 'beh')
    raw_dir = os.path.join(data_path, 'sourcedata', 'raw', f'sci-{subject_id}', 'beh')
    psychopy_clean_dir = r"C:\Users\Max\PycharmProjects\psychopy-experiments\SPACECUE_implicit\logs\clean"

    patterns = [
        os.path.join(prep_dir, f'sci-{subject_id}_clean*.csv'),
        os.path.join(prep_dir, 'legacy_clean', f'sci-{subject_id}_clean*.csv'),
        os.path.join(raw_dir, f'sci-{subject_id}*clean*.csv'),
        os.path.join(raw_dir, f'*clean*.csv'),
        os.path.join(psychopy_clean_dir, f'sci-{subject_id}*clean*.csv')
    ]
    for pat in patterns:
        search_files.extend(glob.glob(pat))

    search_files = [f for f in search_files if os.path.isfile(f) and 'BROKEN' not in f]
    search_files = sorted(set(search_files), key=os.path.getmtime, reverse=True)

    for cf in search_files:
        try:
            df = pd.read_csv(cf)
            age_col = next((c for c in ['Age', 'age'] if c in df.columns), None)
            gender_col = next((c for c in ['Gender', 'gender'] if c in df.columns), None)
            hand_col = next((c for c in ['Handedness', 'handedness'] if c in df.columns), None)

            age_val = np.nan
            gender_val = 'Unknown'
            hand_val = 'Unknown'

            if age_col and df[age_col].notna().any():
                val = df[age_col].dropna().iloc[0]
                if float(val) > 0:
                    age_val = float(val)
            if gender_col and df[gender_col].notna().any():
                gender_val = str(df[gender_col].dropna().iloc[0])
            if hand_col and df[hand_col].notna().any():
                hand_val = str(df[hand_col].dropna().iloc[0])

            if pd.notna(age_val):
                print(f"[INFO] sci-{subject_id}: Successfully extracted demographics from {os.path.basename(cf)} "
                      f"(Age={age_val}, Gender={gender_val}, Handedness={hand_val})")
                return {'Age': age_val, 'Gender': gender_val, 'Handedness': hand_val}
        except Exception:
            continue

    return None


def get_subject_demographics(subject_id, candidate_files=None):
    """
    Retrieves demographics for a subject directly from existing offline clean CSV files
    (e.g., generated during post-experiment formatting in PsychoPy).
    Defaults to NaN/Unknown if no clean file with demographic info is found.
    """
    extracted = extract_demographics_from_clean_file(subject_id, candidate_files=candidate_files)
    if extracted is not None:
        return extracted

    print(f"[NOTICE] sci-{subject_id}: Demographics not found in any clean CSV. Setting Age=NaN, Gender='Unknown', Handedness='Unknown'.")
    return {'Age': np.nan, 'Gender': 'Unknown', 'Handedness': 'Unknown'}


LOCATIONS_MAP = {
    1: 'Left', 2: 'Front', 3: 'Right', 0: 'Not present',
    1.0: 'Left', 2.0: 'Front', 3.0: 'Right', 0.0: 'Not present',
    '1': 'Left', '2': 'Front', '3': 'Right', '0': 'Not present',
    '1.0': 'Left', '2.0': 'Front', '3.0': 'Right', '0.0': 'Not present'
}

PRIMING_MAP = {
    0: 'No priming', -1: 'Negative priming', 1: 'Positive priming',
    0.0: 'No priming', -1.0: 'Negative priming', 1.0: 'Positive priming',
    '0': 'No priming', '-1': 'Negative priming', '1': 'Positive priming',
    '0.0': 'No priming', '-1.0': 'Negative priming', '1.0': 'Positive priming'
}

def clean_subject_behavioral_data(subject_id, overwrite=False):
    """
    Cleans and formats raw behavioral events for a single SPACECUE subject.
    
    Uses stimulus events (event_type == 'stim') as the ground-truth base to guarantee
    that all 1,800 trials retain complete stimulus spatial metadata (TargetLoc,
    SingletonLoc, DistractorProb, Target_pos_x, etc.) even when participants did not
    complete a mouse click before the trial deadline. Mouse clicks are left-joined onto
    the stimulus presentations.
    
    Args:
        subject_id (int): Subject number (e.g. 1, 11, 114)
        overwrite (bool): If True, re-processes raw data even if a valid clean file exists.
        
    Returns:
        tuple: (df_clean, clean_file_path)
    """
    data_path = SPACECUE.get_data_path()
    raw_dir = os.path.join(data_path, 'sourcedata', 'raw', f'sci-{subject_id}', 'beh')
    prep_dir = os.path.join(data_path, 'derivatives', 'preprocessing', f'sci-{subject_id}', 'beh')
    os.makedirs(prep_dir, exist_ok=True)
    
    # Check if a fully valid clean file already exists (1800 rows with no NaNs in Subject ID)
    existing_clean_files = glob.glob(os.path.join(prep_dir, f'sci-{subject_id}_clean*.csv'))
    # Filter out broken / legacy files
    valid_existing = [f for f in existing_clean_files if 'BROKEN' not in f and 'legacy' not in f.lower()]
    
    if not overwrite and valid_existing:
        latest_file = max(valid_existing, key=os.path.getmtime)
        try:
            df_existing = pd.read_csv(latest_file)
            if len(df_existing) == 1800 and df_existing['Subject ID'].isna().sum() == 0:
                print(f"[INFO] sci-{subject_id}: Found existing valid clean file with 1800 trials ({os.path.basename(latest_file)}). Skipping.")
                return df_existing, latest_file
        except Exception:
            pass

    # 1. Discover raw event files
    raw_files = sorted(glob.glob(os.path.join(raw_dir, '*events.csv')))
    if not raw_files:
        raise FileNotFoundError(f"No raw events.csv files found for subject sci-{subject_id} in {raw_dir}")

    # 2. Extract stimulus events and clicks, accounting for session pauses/restarts
    if subject_id == 12 and len(raw_files) > 1:
        # Subject 12 was paused after block 2; file 2 restarted blocks at 0 (representing blocks 3 & 4)
        df1 = pd.read_csv(raw_files[0])
        df2 = pd.read_csv(raw_files[1])
        
        # Stimuli
        s1 = df1[df1['event_type'] == 'stim'].copy()
        s2 = df2[df2['event_type'] == 'stim'].copy()
        s1 = s1[s1['block'] < 3].copy()
        s2 = s2[s2['block'] < 2].copy()
        s2['block'] = s2['block'] + 3
        stims = pd.concat([s1, s2], ignore_index=True)
        
        # Clicks
        c1 = df1[df1['event_type'] == 'mouse_click'].copy()
        c2 = df2[df2['event_type'] == 'mouse_click'].copy()
        c1 = c1[c1['block'] < 3].copy()
        c2 = c2[c2['block'] < 2].copy()
        c2['block'] = c2['block'] + 3
        clicks = pd.concat([c1, c2], ignore_index=True)
        clicks = clicks.drop_duplicates(subset=['block', 'trial_nr'], keep='first')
    else:
        dfs = [pd.read_csv(f) for f in raw_files]
        df_raw = pd.concat(dfs, ignore_index=True)
        stims = df_raw[df_raw['event_type'] == 'stim'].copy()
        stims = stims.drop_duplicates(subset=['block', 'trial_nr'], keep='last')
        clicks = df_raw[df_raw['event_type'] == 'mouse_click'].copy()
        clicks = clicks.drop_duplicates(subset=['block', 'trial_nr'], keep='first')

    stims.sort_values(by=['block', 'trial_nr'], inplace=True)
    stims.reset_index(drop=True, inplace=True)

    # 3. Left-join mouse clicks onto all stimulus trials
    df_clean = pd.merge(
        stims,
        clicks[['block', 'trial_nr', 'response', 'rt']],
        on=['block', 'trial_nr'],
        how='left',
        suffixes=('', '_click')
    )
    df_clean['response'] = pd.to_numeric(df_clean['response_click'], errors='coerce')
    df_clean['rt'] = pd.to_numeric(df_clean['rt_click'], errors='coerce')
    df_clean.drop(columns=['response_click', 'rt_click'], inplace=True)

    # 4. Standardize locations to readable strings
    for c in ['TargetLoc', 'SingletonLoc', 'Non-Singleton1Loc', 'Non-Singleton2Loc']:
        if c in df_clean.columns:
            df_clean[c] = df_clean[c].map(LOCATIONS_MAP).fillna(df_clean[c])
            
    if 'Priming' in df_clean.columns:
        df_clean['Priming'] = df_clean['Priming'].map(PRIMING_MAP).fillna(df_clean['Priming'])

    # 5. Calculate response accuracy and selection columns
    # For non-click trials (rt is NaN), IsCorrect and selection flags are False
    df_clean['IsCorrect'] = (df_clean['response'] == df_clean['TargetDigit'])
    df_clean['SelectDistractor'] = (df_clean['response'] == df_clean['SingletonDigit'])
    df_clean['SelectControl'] = (df_clean['response'] == df_clean['Non-Singleton2Digit']) | (df_clean['response'] == df_clean['Non-Singleton1Digit'])
    df_clean['SelectOther'] = df_clean['response'].notna() & (~df_clean['IsCorrect']) & (~df_clean['SelectDistractor']) & (~df_clean['SelectControl'])

    # 6. Demographics
    demographics = get_subject_demographics(subject_id, candidate_files=existing_clean_files)
    df_clean['Gender'] = demographics['Gender']
    df_clean['Handedness'] = demographics['Handedness']
    df_clean['Age'] = demographics['Age']
    df_clean['Subject ID'] = float(subject_id)

    # 7. Standardize trial, block, phase column names
    df_clean['Trial Nr'] = df_clean['trial_nr']
    df_clean['Block'] = df_clean['block']
    df_clean['Phase'] = df_clean['phase'] if 'phase' in df_clean.columns else 1.0

    # Ensure canonical column order matching downstream scripts
    cols_order = [
        'onset', 'event_type', 'response', 'rt', 'SingletonPresent', 'TargetDigit',
        'TargetLoc', 'SingletonDigit', 'SingletonLoc', 'Non-Singleton2Digit',
        'Non-Singleton2Loc', 'Non-Singleton1Digit', 'Non-Singleton1Loc', 'Priming',
        'HP_Distractor_Loc', 'HP_Distractor_Prob', 'DistractorProb', 'ITI-Jitter',
        'Distractor_pos_x', 'Distractor_pos_y', 'Target_pos_x', 'Target_pos_y',
        'Control2_pos_x', 'Control2_pos_y', 'onset_abs', 'duration', 'IsCorrect',
        'SelectDistractor', 'SelectControl', 'SelectOther', 'Gender', 'Handedness',
        'Trial Nr', 'Phase', 'Block', 'Subject ID', 'Age'
    ]
    present_cols = [c for c in cols_order if c in df_clean.columns]
    df_clean = df_clean[present_cols].copy()

    # 8. Archive old broken clean files to legacy subfolder
    legacy_dir = os.path.join(prep_dir, 'legacy_clean')
    os.makedirs(legacy_dir, exist_ok=True)
    for old_file in existing_clean_files:
        if 'legacy_clean' not in old_file:
            try:
                dest = os.path.join(legacy_dir, os.path.basename(old_file))
                shutil.move(old_file, dest)
            except Exception as e:
                print(f"  Warning: Could not archive {old_file}: {e}")

    # 9. Save new clean behavioral file
    timestamp_str = datetime.now().strftime('%B_%d_%Y_%H_%M_%S')
    out_filename = f"sci-{subject_id}_clean_{timestamp_str}.csv"
    out_filepath = os.path.join(prep_dir, out_filename)
    df_clean.to_csv(out_filepath, index=False)
    print(f"[INFO] sci-{subject_id}: Successfully generated stimulus-centric clean behavioral file with {len(df_clean)} trials ({df_clean['rt'].notna().sum()} clicks) -> {out_filepath}")

    return df_clean, out_filepath


def clean_all_behavioral_data(subjects_to_clean=None, overwrite=False):
    """
    Cleans behavioral data for all specified subjects (defaults to subject_ids from SPACECUE.subjects).
    """
    if subjects_to_clean is None:
        subjects_to_clean = subject_ids

    print(f"\n=========================================================")
    print(f"   CLEANING BEHAVIORAL DATA FOR {len(subjects_to_clean)} SUBJECTS")
    print(f"=========================================================\n")

    results = {}
    for sub in subjects_to_clean:
        try:
            df, path = clean_subject_behavioral_data(sub, overwrite=overwrite)
            results[sub] = path
        except Exception as e:
            print(f"[ERROR] sci-{sub}: Failed to clean behavioral data: {e}")

    print(f"\n[INFO] Finished cleaning behavioral data for {len(results)}/{len(subjects_to_clean)} subjects.\n")
    return results

if __name__ == "__main__":
    clean_all_behavioral_data(overwrite=True)
