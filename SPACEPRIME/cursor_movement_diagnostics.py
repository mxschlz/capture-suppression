'''
This script investigates the sharp peak of first mouse movements
right before the response interval ends... Turns out: artifacts cause this pattern,
because it is exclusively 5 response trials.
'''
from utils import *


def run_diagnostics(first_movement_times_df, df_clean, df, analysis_df_classification, MOVEMENT_THRESHOLD=0.05,
                    RESAMP_FREQ=60):
    # --- Diagnostic Step: Investigate the peak near the response time ---
    print("\n--- Investigating the sharp peak in initial movement times ---")

    # Based on your observation, the peak occurs around -0.2s before the response.
    # Let's define a narrow window around this peak to isolate these specific trials.
    peak_window_start = -0.4
    print(
        f"Isolating trials where the first movement was detected between {peak_window_start}s (relative to response).")

    # Get the trials that contribute to this peak
    peak_trials_df = first_movement_times_df[
        (first_movement_times_df['first_movement_s'] > peak_window_start)
    ]

    print(f"\nFound {len(peak_trials_df)} trials within this peak window.")

    if not peak_trials_df.empty:
        # To understand these trials, let's look at their properties from the main behavioral dataframe.
        # We need to merge them back with `df_clean` which has the RT and response info.
        # Ensure dtypes match for merging.
        peak_trials_df['subject_id'] = peak_trials_df['subject_id'].astype(int)
        peak_trials_df['block'] = peak_trials_df['block'].astype(int)
        peak_trials_df['trial_nr'] = peak_trials_df['trial_nr'].astype(int)

        # Merge to get full trial info
        peak_details_df = pd.merge(
            df_clean,
            peak_trials_df,
            on=['subject_id', 'block', 'trial_nr'],
            how='inner'  # We only want the trials that are in both dataframes
        )

        print("\n--- Characteristics of trials in the peak ---")
        print("Distribution of their Reaction Times (rt):")
        print(peak_details_df['rt'].describe())

        print("\nDistribution of their Response Correctness (select_target):")
        print(peak_details_df['select_target'].value_counts(normalize=True, dropna=False))

        print("\n(For comparison, the RT distribution of ALL trials in df_clean):")
        print(df_clean['rt'].describe())

        # --- Visualize a few example trajectories from the peak ---
        print("\n--- Visualizing a few example trajectories from the peak ---")
        # We will use the `analysis_df` as it has all the necessary columns for plotting
        # and has already been merged with initial movement data.
        peak_viz_df = pd.merge(
            analysis_df_classification,
            peak_trials_df[['subject_id', 'block', 'trial_nr']],  # Just use the identifiers to filter
            on=['subject_id', 'block', 'trial_nr'],
            how='inner'
        )

        if not peak_viz_df.empty:
            # Take up to 3 random trials from the peak to visualize
            num_to_plot = min(len(peak_viz_df), 3)
            print(f"Plotting {num_to_plot} random example trajectories...")
            for i, trial_to_plot in peak_viz_df.sample(n=num_to_plot).iterrows():
                print(
                    f"Plotting sub-{int(trial_to_plot['subject_id'])}, block-{int(trial_to_plot['block'])}, trial-{int(trial_to_plot['trial_nr'])}")
                visualize_full_trajectory(trial_to_plot, df, MOVEMENT_THRESHOLD, target_hz=RESAMP_FREQ)
        else:
            print("Could not find matching trials in `analysis_df` to visualize.")

    else:
        print("No trials found in the specified peak window.")
