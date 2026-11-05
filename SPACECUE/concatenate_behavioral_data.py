import os
import sys
import glob
import numpy as np
import pandas as pd

# Ensure project root is in python path
try:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
except NameError:
    project_root = os.path.abspath(os.getcwd())
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import SPACECUE
import utils

def main():
    print("=========================================================")
    print("   SPACECUE BEHAVIORAL & TOWARDNESS CONCATENATION   ")
    print("=========================================================\n")

    # 1. Resolve data path using the package utilities
    data_path = SPACECUE.get_data_path()
    print(f"Data Path: {data_path}")

    raw_dir = os.path.join(data_path, 'sourcedata', 'raw')
    prep_dir = os.path.join(data_path, 'derivatives', 'preprocessing')

    # 2. Dynamically discover subjects present in both raw and preprocessing folders
    print("Searching for subject data folders...")
    raw_subjects = set()
    for p in glob.glob(os.path.join(raw_dir, 'sci-*')):
        try:
            raw_subjects.add(int(os.path.basename(p).split('-')[1]))
        except (ValueError, IndexError):
            continue

    prep_subjects = set()
    for p in glob.glob(os.path.join(prep_dir, 'sci-*')):
        try:
            prep_subjects.add(int(os.path.basename(p).split('-')[1]))
        except (ValueError, IndexError):
            continue

    from SPACECUE.subjects import subject_ids
    subjects = sorted(list(raw_subjects.intersection(prep_subjects)))
    subjects = [s for s in subjects if s in subject_ids]
    if not subjects:
        print("Error: No subjects found matching subjects.py with both raw and preprocessed data.")
        sys.exit(1)

    print(f"Discovered {len(subjects)} subjects (filtered by subjects.py): {subjects}\n")

    # 3. Load behavioral data
    print("Loading behavioral files...")
    df_list = []
    for sub_id in subjects:
        sub_beh_files = glob.glob(os.path.join(prep_dir, f"sci-{sub_id}", "beh", f"sci-{sub_id}_clean*.csv"))
        valid_files = [f for f in sub_beh_files if "BROKEN" not in f and "legacy" not in f.lower()]
        if not valid_files:
            print(f"  Subject {sub_id:2d}: Warning - No valid clean file found!")
            continue
        latest_file = max(valid_files, key=os.path.getmtime)
        print(f"  Subject {sub_id:2d}: Loading {os.path.basename(latest_file)}")
        df_list.append(pd.read_csv(latest_file))

    if not df_list:
        print("Error: No behavioral data loaded.")
        sys.exit(1)

    df_beh = pd.concat(df_list, ignore_index=True)

    # Filter out empty/invalid trials and ensure valid subject IDs
    df_beh = df_beh.dropna(subset=['Subject ID', 'Block', 'Trial Nr'])
    df_beh = df_beh[df_beh['Subject ID'].isin(subjects)].copy()

    # Normalize column names & types for analysis
    if "IsCorrect" in df_beh.columns:
        df_beh["IsCorrect"] = df_beh["IsCorrect"].replace({'True': 1, 'False': 0, True: 1, False: 0})
        df_beh["IsCorrect"] = pd.to_numeric(df_beh["IsCorrect"], errors='coerce')

    df_beh['Subject ID'] = df_beh['Subject ID'].astype(int)
    df_beh['Block'] = df_beh['Block'].astype(int)
    df_beh['Trial Nr'] = df_beh['Trial Nr'].astype(int)
    
    if 'HP_Distractor_Loc' in df_beh.columns:
        df_beh['HP_Distractor_Loc'] = df_beh['HP_Distractor_Loc'].replace({1: 'Left', 2: 'Front', 3: 'Right'})
    if 'HP_Distractor_Prob' in df_beh.columns:
        df_beh['HP_Distractor_Prob'] = df_beh['HP_Distractor_Prob'].astype(float)

    df_beh['subject_id'] = df_beh['Subject ID']
    df_beh['block'] = df_beh['Block']
    df_beh['trial_nr'] = df_beh['Trial Nr']

    # 4. Load raw mouse trajectory data
    print("\nLoading raw mouse trajectories...")
    df_mouse = utils.load_raw_mouse_trajectories(subjects, data_path, calculate_phase=True)
    if df_mouse.empty:
        print("Error: No mouse trajectory data found.")
        sys.exit(1)
    print(f"  Loaded {len(df_mouse)} raw trajectory coordinates.")

    # 5. Align trajectories and convert to canonical 1.0 dva radius
    print("Aligning trajectories to canonical reference frame (radius = 1.0 dva)...")
    df_mouse_aligned = utils.align_raw_mouse_trajectories(df_mouse, df_beh, target_radius_dva=1.0)
    if df_mouse_aligned.empty:
        print("Error: Aligned trajectory dataframe is empty.")
        sys.exit(1)

    # Also convert raw stimulus position columns in df_beh from raw radius (2.0) to 1.0 dva
    raw_stim_radius = np.hypot(df_beh['Target_pos_x'], df_beh['Target_pos_y'])
    stim_scale = np.where(raw_stim_radius > 0, 1.0 / raw_stim_radius, 0.5)
    for pos_col in ['Target_pos_x', 'Target_pos_y', 'Distractor_pos_x', 'Distractor_pos_y',
                    'Control2_pos_x', 'Control2_pos_y']:
        if pos_col in df_beh.columns:
            df_beh[pos_col] = df_beh[pos_col] * stim_scale

    # 6. Compute average trajectory vector per trial (RESPONSE INTERVAL, UP TO
    #    AND INCLUDING THE RESPONSE CLICK)
    # The 'phase' column (from utils.load_raw_mouse_trajectories with
    # calculate_phase=True) encodes 0 = stimulus, 1 = response interval,
    # 2 = ITI. Within the response interval the logged clock runs from
    # -1.75 s (interval start) up to 0 (interval end). The click time on this
    # clock is t_cut = rt - 2.0 s: behavioral 'rt' is logged from STIMULUS
    # onset, the stimulus lasts 0.25 s, and the response interval is 1.75 s,
    # so the click occurs (rt - 0.25) s after response onset = rt - 2.0 on
    # the response clock. This anchor was validated empirically against the
    # trajectories: at t_cut the cursor sits at its final parked position
    # (median distance 0.00 dva), and the hand stops moving ~70 ms before the
    # logged press (arrive-then-press pattern).
    #
    # Only response-phase samples up to the click enter the average. This
    # excludes (a) stimulus-interval samples (cursor at the origin -> pure
    # dilution of the vector), (b) ITI samples (cursor parked at the clicked
    # digit, never reset), and (c) the post-click dwell *within* the response
    # interval (~0.4 s on average), which would otherwise pin the 'movement'
    # vector to the response endpoint. Trials without a logged rt (no click /
    # timeout) get NaN towardness. If the click lies after the last logged
    # response sample, the full response interval is kept.
    print("Computing average trajectory vector for each trial (response interval, up to the click)...")
    if 'phase' not in df_mouse_aligned.columns:
        raise ValueError("Phase column missing from trajectory data - "
                         "load_raw_mouse_trajectories must be called with calculate_phase=True.")
    if 'rt' not in df_beh.columns:
        raise ValueError("Behavioral data has no 'rt' column - cannot locate the response click.")
    df_response = df_mouse_aligned[df_mouse_aligned['phase'] == 1].merge(
        df_beh[['subject_id', 'block', 'trial_nr', 'rt']],
        on=['subject_id', 'block', 'trial_nr'], how='left')
    df_response['t_cut'] = df_response['rt'] - 2.0

    n_trials_aligned = df_mouse_aligned.groupby(['subject_id', 'block', 'trial_nr']).ngroups
    n_trials_response = df_response.groupby(['subject_id', 'block', 'trial_nr']).ngroups
    n_trials_click = int(df_beh['rt'].notna().sum())
    print(f"  Trials with aligned trajectory samples: {n_trials_aligned}")
    print(f"  Trials with response-interval (phase 1) samples: {n_trials_response}")
    print(f"  Trials with a logged click (rt available): {n_trials_click}")
    if n_trials_response < n_trials_aligned:
        print(f"  NOTE: {n_trials_aligned - n_trials_response} trial(s) have no "
              f"response-phase samples -> towardness will be NaN for those trials.")

    # Trials whose click lies after the last logged response sample keep the
    # full response interval (reported here for transparency). The comparison
    # is aligned onto the trajectory index because behavioral trials without
    # mouse data are absent from the groupby result.
    t_max_per_trial = df_response.groupby(['subject_id', 'block', 'trial_nr'])['time'].max()
    rt_per_trial = (df_beh.drop_duplicates(['subject_id', 'block', 'trial_nr'])
                          .set_index(['subject_id', 'block', 'trial_nr'])['rt'])
    t_cut_per_trial = (rt_per_trial - 2.0).reindex(t_max_per_trial.index)
    n_clipped = int((t_cut_per_trial > t_max_per_trial).fillna(False).sum())
    print(f"  Trials where the click lies after the last logged response sample "
          f"(full interval used): {n_clipped}")

    df_until_click = df_response[df_response['t_cut'].notna()
                                 & (df_response['time'] <= df_response['t_cut'])]
    agg_dict = {
        'avg_x_dva': ('x_aligned', 'mean'),
        'avg_y_dva': ('y_aligned', 'mean'),
        'target_x_aligned': ('target_x_aligned', 'first'),
        'target_y_aligned': ('target_y_aligned', 'first'),
        'distractor_x_aligned': ('distractor_x_aligned', 'first'),
        'distractor_y_aligned': ('distractor_y_aligned', 'first'),
        'control_x_aligned': ('control_x_aligned', 'first'),
        'control_y_aligned': ('control_y_aligned', 'first'),
    }
    avg_vectors = df_until_click.groupby(['subject_id', 'block', 'trial_nr']).agg(**agg_dict).reset_index()

    # Merge average coordinates back into main behavioral dataframe
    df_beh = pd.merge(df_beh, avg_vectors, on=['subject_id', 'block', 'trial_nr'], how='left')

    # 7. Compute Towardness Scores
    print("Calculating towardness scores for all trials...")
    towardness_scores_list = []
    for idx, row in df_beh.iterrows():
        if pd.isna(row['target_x_aligned']) or pd.isna(row['avg_x_dva']):
            towardness_scores_list.append({
                'target_towardness': np.nan,
                'distractor_towardness': np.nan,
                'control_towardness': np.nan
            })
            continue

        locations_map = {
            1: (row['target_x_aligned'], row['target_y_aligned']),
            2: (row['distractor_x_aligned'], row['distractor_y_aligned']),
            3: (row['control_x_aligned'], row['control_y_aligned'])
        }

        row_copy = row.copy()
        row_copy['TargetDigit'] = 1
        row_copy['SingletonDigit'] = 2
        row_copy['SingletonPresent'] = 1
        row_copy['Non-Singleton1Digit'] = np.nan
        row_copy['Non-Singleton2Digit'] = 3

        scores = utils.calculate_trajectory_projections(row_copy, locations_map=locations_map, percentage=True)
        towardness_scores_list.append({
            'target_towardness': scores['proj_target'],
            'distractor_towardness': scores['proj_distractor'],
            'control_towardness': scores['proj_control_avg']
        })

    towardness_df = pd.DataFrame(towardness_scores_list)
    df_final = pd.concat([df_beh.reset_index(drop=True), towardness_df.reset_index(drop=True)], axis=1)

    # 7b. Remove target towardness outliers (> 2 SD from the within-subject mean)
    # Outlier detection is performed separately for each subject, so trials are
    # compared against that subject's own mean and SD (ddof=1) of target
    # towardness, not the group-level statistics.
    print("\nRemoving target towardness outliers (> 2 SD from within-subject mean)...")
    n_before_outlier = len(df_final)
    sub_mean = df_final.groupby('subject_id')['target_towardness'].transform('mean')
    sub_sd = df_final.groupby('subject_id')['target_towardness'].transform('std')
    is_outlier = ((df_final['target_towardness'] - sub_mean).abs() > (2 * sub_sd)).fillna(False)

    removed_per_subject = df_final.loc[is_outlier, 'subject_id'].value_counts().sort_index()
    for sub_id, n_removed_sub in removed_per_subject.items():
        print(f"  Subject {sub_id:2d}: removed {n_removed_sub} outlier trial(s)")

    df_final = df_final[~is_outlier].reset_index(drop=True)
    n_removed_outlier = int(is_outlier.sum())
    print(f"  Total: removed {n_removed_outlier} of {n_before_outlier} trials "
          f"({100 * n_removed_outlier / max(n_before_outlier, 1):.2f}%)")

    # 8. Save concatenated behavioral CSV
    output_dir = os.path.join(data_path, "concatenated")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "concatenated_behavioral.csv")

    print(f"\nSaving final concatenated behavioral data to: {output_file}")
    df_final.to_csv(output_file, index=False)

    # 9. Resample response-interval trajectories to 101 normalized steps (r = 1.0 dva)
    #    and save aligned_mouse_resampled_101.npz
    print("\nResampling response-interval trajectories to 101 normalized steps (radius = 1.0 dva)...")
    norm_steps = 101
    steps_arr = np.arange(norm_steps, dtype=float)
    time_norm_arr = np.linspace(0.0, 1.0, norm_steps)

    meta_cols = ['subject_id', 'block', 'trial_nr', 'Age', 'SingletonLoc',
                 'HP_Distractor_Loc', 'IsCorrect', 'SingletonPresent']
    df_meta = df_beh[meta_cols].drop_duplicates(['subject_id', 'block', 'trial_nr']).copy()
    df_meta['is_hp'] = (df_meta['SingletonLoc'] == df_meta['HP_Distractor_Loc']).astype(float)
    df_meta['is_correct'] = df_meta['IsCorrect'].fillna(0.0).astype(float)
    df_meta['singleton_present'] = df_meta['SingletonPresent'].fillna(1.0).astype(float)
    df_meta['age'] = df_meta['Age'].astype(float)

    df_resp_meta = df_response.merge(
        df_meta[['subject_id', 'block', 'trial_nr', 'age', 'is_hp', 'is_correct', 'singleton_present']],
        on=['subject_id', 'block', 'trial_nr'], how='left'
    ).sort_values(['subject_id', 'block', 'trial_nr', 'time'])

    resampled_blocks = []
    for (sub_id, blk, trl), grp in df_resp_meta.groupby(['subject_id', 'block', 'trial_nr'], sort=True):
        t_vals = grp['time'].to_numpy(dtype=float)
        x_vals = grp['x_aligned'].to_numpy(dtype=float)
        y_vals = grp['y_aligned'].to_numpy(dtype=float)
        if t_vals.size < 2:
            x_interp = np.full(norm_steps, x_vals[0] if x_vals.size == 1 else np.nan)
            y_interp = np.full(norm_steps, y_vals[0] if y_vals.size == 1 else np.nan)
        else:
            t_grid = np.linspace(t_vals[0], t_vals[-1], norm_steps)
            x_interp = np.interp(t_grid, t_vals, x_vals)
            y_interp = np.interp(t_grid, t_vals, y_vals)

        row0 = grp.iloc[0]
        trial_block = np.column_stack([
            np.full(norm_steps, float(sub_id)),
            np.full(norm_steps, float(blk)),
            np.full(norm_steps, float(trl)),
            np.full(norm_steps, float(row0['age'])),
            np.full(norm_steps, float(row0['is_hp'])),
            np.full(norm_steps, float(row0['is_correct'])),
            np.full(norm_steps, float(row0['singleton_present'])),
            steps_arr,
            time_norm_arr,
            x_interp,
            y_interp,
        ])
        resampled_blocks.append(trial_block)

    resampled_data = np.vstack(resampled_blocks)
    resampled_columns = np.array([
        'subject_id', 'block', 'trial_nr', 'age', 'is_hp', 'is_correct',
        'singleton_present', 'step', 'time_norm', 'x_aligned', 'y_aligned'
    ])
    npz_file = os.path.join(output_dir, "aligned_mouse_resampled_101.npz")
    print(f"Saving 101-step resampled trajectory archive ({len(resampled_blocks)} trials) to: {npz_file}")
    np.savez(npz_file, data=resampled_data, columns=resampled_columns)
    print("Processing complete! Both files saved successfully.")

if __name__ == "__main__":
    main()
