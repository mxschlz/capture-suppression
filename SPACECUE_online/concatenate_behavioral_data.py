import os
import re
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

import SPACECUE_online

# Filename pattern: sce-<subject>_block_<block>_data_<Date>.csv
FILE_PATTERN = re.compile(r"sce-(\d+)_block_(\d+)_(data|trajectories)_")


def load_block_files(kind):
    """Load and concatenate all per-block csv files of one kind ('data' or 'trajectories')."""
    # Raw data location: DataPipe saving folder ('G:\Meine Ablage\DataPipe\SpaceCue Online'
    # on Windows, see SPACECUE_online.get_data_path()), which holds old + new datasets.
    beh_dir = SPACECUE_online.get_data_path()
    if not os.path.isdir(beh_dir):
        raise FileNotFoundError(f"Raw data folder not found: {beh_dir}")
    files = sorted(glob.glob(os.path.join(beh_dir, f"*_{kind}_*.csv")))
    # NOTE: glob with the explicit '_<kind>_' infix only matches top-level files,
    # the 'obsolete' subfolder (old pilot runs of subject 02) is intentionally excluded.

    df_list = []
    for f in files:
        m = FILE_PATTERN.match(os.path.basename(f))
        if not m:
            print(f"  WARNING: could not parse subject/block from {os.path.basename(f)} - skipped")
            continue
        sub_from_name, block_from_name = int(m.group(1)), int(m.group(2))

        df = pd.read_csv(f)
        df["source_file"] = os.path.basename(f)
        df["subject_id"] = sub_from_name  # normalized (leading zeros stripped)
        if "block" in df.columns:
            df["block"] = df["block"].astype(int)
            # Cross-check the block recorded in the file against the filename
            if not (df["block"] == block_from_name).all():
                print(f"  WARNING: block mismatch in {os.path.basename(f)} "
                      f"(filename: {block_from_name}, file content: {sorted(df['block'].unique())})")
        df_list.append(df)

    df_all = pd.concat(df_list, ignore_index=True)
    return df_all, len(files)


# --- 1. Concatenate trial-level behavioral data -----------------------------
print("=========================================================")
print("   SPACECUE ONLINE BEHAVIORAL CONCATENATION              ")
print("=========================================================\n")

df_beh, n_files = load_block_files("data")
print(f"Loaded {n_files} trial-level data files -> {len(df_beh)} trials "
      f"from {df_beh['subject_id'].nunique()} subjects\n")

# Type normalization
for col in ["trial_nr", "block", "subject_id", "TargetDigit", "SingletonDigit",
            "TargetLoc", "SingletonLoc", "SingletonPresent", "age"]:
    if col in df_beh.columns:
        df_beh[col] = pd.to_numeric(df_beh[col], errors="coerce").astype("Int64")
for col in ["rt", "cue_stim_delay_jitter", "ITI-Jitter"]:
    if col in df_beh.columns:
        df_beh[col] = pd.to_numeric(df_beh[col], errors="coerce")
df_beh["response"] = pd.to_numeric(df_beh["response"], errors="coerce").astype("Int64")

# --- 2. Overview / sanity checks of the concatenated table ------------------
print("Per-subject overview (blocks x trials, cue types per subject):")
overview = (df_beh.groupby("subject_id")
            .agg(n_trials=("trial_nr", "size"),
                 n_blocks=("block", "nunique"),
                 cue_types=("CueInstruction", lambda s: ", ".join(sorted(s.unique()))))
            .reset_index())
print(overview.to_string(index=False))

dups = df_beh.duplicated(subset=["subject_id", "block", "trial_nr"], keep=False)
if dups.any():
    print(f"\nWARNING: {int(dups.sum())} duplicated (subject, block, trial_nr) rows!")
    print(df_beh.loc[dups, ["subject_id", "block", "trial_nr", "source_file"]].head(20).to_string(index=False))

# --- 3. Save concatenated behavioral data ------------------------------------
out_dir = SPACECUE_online.get_concatenated_data_path()
os.makedirs(out_dir, exist_ok=True)
out_file = os.path.join(out_dir, "concatenated_behavioral.csv")
df_beh.to_csv(out_file, index=False)
print(f"\nSaved trial-level data to: {out_file}")

# --- 4. Concatenate mouse-trajectory files (same loop, larger tables) -------
df_traj, n_traj_files = load_block_files("trajectories")
print(f"\nLoaded {n_traj_files} trajectory files -> {len(df_traj)} rows")
out_traj = os.path.join(out_dir, "concatenated_trajectories.csv")
df_traj.to_csv(out_traj, index=False)
print(f"Saved trajectory data to: {out_traj}")

print("\nProcessing complete!")
