import mne
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import t
from mne.stats import permutation_cluster_1samp_test
from SPACECUE import load_concatenated_epochs
import SPACECUE

sns.set_theme(context="talk", style="ticks")
plt.ion()

# --- Script Parameters ---
subject_ids = SPACECUE.subjects.subject_ids

# --- P3 Analysis Parameters ---
# Vertex/Midline ROI where surprisal P3 components are typically maximal
VERTEX_ROI = ["Fz", "Cz", "Pz"]

# --- Metadata Configuration for Expected vs Unexpected ---
# IMPORTANT: Modify these based on how the distractor rules are coded in your epochs.metadata!
META_COL_HP_DIST_LOC = 'HP_Distractor_Loc' 
META_VAL_LEFT = 1  # e.g., change to '1', 'left', or 'Left' based on your metadata
META_VAL_RIGHT = 3 # e.g., change to '3', 'right', or 'Right' based on your metadata

# Epoching and Plotting
EPOCH_TMIN, EPOCH_TMAX = -0.1, 0.7  # Seconds
AMPLITUDE_SCALE_FACTOR = 1e6
PLOT_P3_TOPO = True
P3_TOPO_WINDOW = (0.25, 0.35) # Time window to plot the summary topography of the P3 difference

# Cluster Permutation Test Parameters
N_JOBS = 5
SEED = 42
N_PERMUTATIONS_CLUSTER = 10000
ALPHA_STAT_CLUSTER = 0.05
CLUSTER_TAIL = 0  # 0 for two-tailed

# --- Data Storage ---
subject_data = {
    'expected': [], 'unexpected': [], 'diff': [], 'diff_topo': []
}
subject_data_early = {
    'expected': [], 'unexpected': [], 'diff': []
}
processed_subjects = []
processed_subjects_early = []
times_vector = None
epochs_info = None

# --- Load and Preprocess Data ---
print("--- Loading and Preprocessing Data ---")
epochs = load_concatenated_epochs("spacecue").crop(EPOCH_TMIN, EPOCH_TMAX)
montage = mne.channels.make_standard_montage("easycap-M1")
epochs.set_montage(montage)

# --- Subject Loop ---
for subject_id_num in subject_ids:
    subject_str = f"sci-{subject_id_num}"
    print(f"\n--- Processing Subject: {subject_str} ---")
    
    try:
        epochs_sub = epochs[f"`Subject ID` == {subject_id_num}"]
    except Exception as e:
        print(f"  Error loading data for {subject_str}: {e}. Skipping.")
        continue

    if times_vector is None:
        times_vector = epochs_sub.times.copy()
        epochs_info = epochs_sub.info.copy()

    meta = epochs_sub.metadata
    if meta is None or META_COL_HP_DIST_LOC not in meta.columns:
        print(f"  Metadata column '{META_COL_HP_DIST_LOC}' not found for {subject_str}. Skipping.")
        continue

    # Create a copy to prevent pandas SettingWithCopy warnings
    meta = meta.copy()
    # Track when the HP_Distractor_Loc value changes to reset the cumcount block-by-block
    block_ids = (meta[META_COL_HP_DIST_LOC] != meta[META_COL_HP_DIST_LOC].shift(1)).cumsum()
    meta['trials_since_switch'] = meta.groupby(block_ids).cumcount()
    epochs_sub.metadata = meta

    all_conds_sub = list(epochs_sub.event_id.keys())
    
    # We look at trials where the target is central (2) and the distractor is lateral (1 or 3)
    left_dist_events = [x for x in all_conds_sub if "Target-2-Singleton-1" in x]
    right_dist_events = [x for x in all_conds_sub if "Target-2-Singleton-3" in x]
    
    if not left_dist_events or not right_dist_events:
        print(f"  Missing lateral distractor conditions for {subject_str}. Skipping.")
        continue

    try:
        # Find the epochs for each of the four sub-conditions by combining event names and metadata queries
        exp_left_epochs = epochs_sub[left_dist_events][f"`{META_COL_HP_DIST_LOC}` == {repr(META_VAL_LEFT)}"]
        exp_right_epochs = epochs_sub[right_dist_events][f"`{META_COL_HP_DIST_LOC}` == {repr(META_VAL_RIGHT)}"]
        
        unexp_left_epochs = epochs_sub[left_dist_events][f"`{META_COL_HP_DIST_LOC}` == {repr(META_VAL_RIGHT)}"]
        unexp_right_epochs = epochs_sub[right_dist_events][f"`{META_COL_HP_DIST_LOC}` == {repr(META_VAL_LEFT)}"]

        # Combine them into two main conditions using MNE's concatenation
        epochs_expected = mne.concatenate_epochs([exp_left_epochs, exp_right_epochs], on_mismatch='raise')
        epochs_unexpected = mne.concatenate_epochs([unexp_left_epochs, unexp_right_epochs], on_mismatch='raise')

        # Extract only the first 10 trials (indices 0 to 9) after a rule switch
        epochs_exp_early = epochs_expected["trials_since_switch < 10"]
        epochs_unexp_early = epochs_unexpected["trials_since_switch < 10"]

        print(f"  Trials found -> Expected: {len(epochs_expected)}, Unexpected: {len(epochs_unexpected)}")
        print(f"  Early Trials found -> Expected: {len(epochs_exp_early)}, Unexpected: {len(epochs_unexp_early)}")

        if len(epochs_expected) == 0 or len(epochs_unexpected) == 0:
            print(f"  Zero trials in one of the expected/unexpected conditions. Skipping.")
            continue

        # 1. Create evoked objects for the vertex ROI and average over channels
        evk_expected = epochs_expected.average(picks=VERTEX_ROI)
        evk_unexpected = epochs_unexpected.average(picks=VERTEX_ROI)

        exp_sub_erp = evk_expected.data.mean(axis=0)  # Average over channels in ROI
        unexp_sub_erp = evk_unexpected.data.mean(axis=0) # Average over channels in ROI
        diff_sub_erp = unexp_sub_erp - exp_sub_erp  # Unexpected - Expected
        
        # 2. Create a difference wave for all channels for topography plotting
        if PLOT_P3_TOPO:
            # Create evoked objects for all channels first
            evk_all_ch_expected = epochs_expected.average()
            evk_all_ch_unexpected = epochs_unexpected.average()

            # Calculate the difference wave using mne.combine_evoked
            evk_diff_topo = mne.combine_evoked(
                [evk_all_ch_unexpected, evk_all_ch_expected],
                weights=[1, -1]
            )
            subject_data['diff_topo'].append(evk_diff_topo.data)

        # Store
        subject_data['expected'].append(exp_sub_erp)
        subject_data['unexpected'].append(unexp_sub_erp)
        subject_data['diff'].append(diff_sub_erp)
        processed_subjects.append(subject_id_num)

        # Process Early Trials Evoked
        if len(epochs_exp_early) > 0 and len(epochs_unexp_early) > 0:
            evk_exp_early = epochs_exp_early.average(picks=VERTEX_ROI)
            evk_unexp_early = epochs_unexp_early.average(picks=VERTEX_ROI)
            
            exp_early_erp = evk_exp_early.data.mean(axis=0)
            unexp_early_erp = evk_unexp_early.data.mean(axis=0)
            diff_early_erp = unexp_early_erp - exp_early_erp
            
            subject_data_early['expected'].append(exp_early_erp)
            subject_data_early['unexpected'].append(unexp_early_erp)
            subject_data_early['diff'].append(diff_early_erp)
            processed_subjects_early.append(subject_id_num)

    except Exception as e:
        print(f"  Error processing expected/unexpected logic for {subject_str}: {e}")
        continue

# --- Group Level Analysis & Stats ---
n_subs = len(processed_subjects)
if n_subs < 2:
    raise RuntimeError("Need at least 2 subjects to run statistics and plot.")

ga_expected = np.mean(subject_data['expected'], axis=0)
ga_unexpected = np.mean(subject_data['unexpected'], axis=0)
ga_diff = np.mean(subject_data['diff'], axis=0)

deg_f = n_subs - 1
t_thresh_cluster = t.ppf(1 - ALPHA_STAT_CLUSTER / 2, deg_f)

print(f"\n--- Running Cluster Permutation Test (N={n_subs}) ---")
t_obs, clusters, cluster_pv, h0 = permutation_cluster_1samp_test(
    np.array(subject_data['diff']), threshold=t_thresh_cluster, 
    n_permutations=N_PERMUTATIONS_CLUSTER, tail=CLUSTER_TAIL, 
    n_jobs=N_JOBS, out_type="mask", seed=SEED, verbose=False
)

# --- Plotting ERPs ---
fig_erp, ax = plt.subplots(1, 1, figsize=(10, 6))

ax.plot(times_vector, ga_expected * AMPLITUDE_SCALE_FACTOR, color='tab:blue', linestyle='--', label="Expected Distractor", linewidth=2)
ax.plot(times_vector, ga_unexpected * AMPLITUDE_SCALE_FACTOR, color='tab:orange', linestyle='-', label="Unexpected Distractor", linewidth=2)
ax.plot(times_vector, ga_diff * AMPLITUDE_SCALE_FACTOR, color='black', label="Unexpected - Expected (Diff)", linewidth=2.5)

ax.axhline(y=0, color='k', linestyle='-', linewidth=0.8, alpha=0.5)
ax.axvline(x=0, color='k', linestyle='-', linewidth=0.8, alpha=0.5)

# Plot Significant Clusters
significant_cluster_plotted = False
for i_c, cl_mask in enumerate(clusters):
    if cluster_pv[i_c] < ALPHA_STAT_CLUSTER:
        cluster_times_sig = times_vector[cl_mask]
        if len(cluster_times_sig) > 0:
            ax.hlines(y=-1.0,  # Offset for the significance line
                      xmin=cluster_times_sig[0], xmax=cluster_times_sig[-1],
                      color='black', linewidth=5, alpha=0.7,
                      label='Sig. Cluster (p < 0.05)' if not significant_cluster_plotted else '_nolegend_')
            significant_cluster_plotted = True
            
            print(f"Significant Cluster: {cluster_times_sig[0]*1000:.0f} - {cluster_times_sig[-1]*1000:.0f} ms (p={cluster_pv[i_c]:.4f})")

ax.set_title(f"P3 Response to Distractor Violation at Vertex ({', '.join(VERTEX_ROI)})")
ax.set_ylabel("Amplitude [µV]")
ax.set_xlabel("Time [s]")
ax.legend(loc='upper right', frameon=True)
sns.despine(ax=ax)
fig_erp.tight_layout()

# --- Plotting Early Trials ERPs (First 10 Post-Switch) ---
n_subs_early = len(processed_subjects_early)
if n_subs_early >= 2:
    ga_expected_early = np.mean(subject_data_early['expected'], axis=0)
    ga_unexpected_early = np.mean(subject_data_early['unexpected'], axis=0)
    ga_diff_early = np.mean(subject_data_early['diff'], axis=0)

    t_thresh_cluster_early = t.ppf(1 - ALPHA_STAT_CLUSTER / 2, n_subs_early - 1)

    print(f"\n--- Running Cluster Permutation Test for Early Trials (N={n_subs_early}) ---")
    t_obs_early, clusters_early, cluster_pv_early, h0_early = permutation_cluster_1samp_test(
        np.array(subject_data_early['diff']), threshold=t_thresh_cluster_early, 
        n_permutations=N_PERMUTATIONS_CLUSTER, tail=CLUSTER_TAIL, 
        n_jobs=N_JOBS, out_type="mask", seed=SEED, verbose=False
    )

    fig_erp_early, ax_early = plt.subplots(1, 1, figsize=(10, 6))
    ax_early.plot(times_vector, ga_expected_early * AMPLITUDE_SCALE_FACTOR, color='tab:blue', linestyle='--', label="Expected Distractor", linewidth=2)
    ax_early.plot(times_vector, ga_unexpected_early * AMPLITUDE_SCALE_FACTOR, color='tab:orange', linestyle='-', label="Unexpected Distractor", linewidth=2)
    ax_early.plot(times_vector, ga_diff_early * AMPLITUDE_SCALE_FACTOR, color='black', label="Unexpected - Expected (Diff)", linewidth=2.5)

    ax_early.axhline(y=0, color='k', linestyle='-', linewidth=0.8, alpha=0.5)
    ax_early.axvline(x=0, color='k', linestyle='-', linewidth=0.8, alpha=0.5)

    significant_cluster_plotted_early = False
    for i_c, cl_mask in enumerate(clusters_early):
        if cluster_pv_early[i_c] < ALPHA_STAT_CLUSTER:
            cluster_times_sig = times_vector[cl_mask]
            if len(cluster_times_sig) > 0:
                ax_early.hlines(y=-1.0, xmin=cluster_times_sig[0], xmax=cluster_times_sig[-1],
                          color='black', linewidth=5, alpha=0.7,
                          label='Sig. Cluster (p < 0.05)' if not significant_cluster_plotted_early else '_nolegend_')
                significant_cluster_plotted_early = True
                print(f"Significant Cluster (Early): {cluster_times_sig[0]*1000:.0f} - {cluster_times_sig[-1]*1000:.0f} ms (p={cluster_pv_early[i_c]:.4f})")

    ax_early.set_title(f"Early Trials (First 10 post-switch): P3 Response ({', '.join(VERTEX_ROI)})")
    ax_early.set_ylabel("Amplitude [µV]")
    ax_early.set_xlabel("Time [s]")
    ax_early.legend(loc='upper right', frameon=True)
    sns.despine(ax=ax_early)
    fig_erp_early.tight_layout()

# --- Plotting Difference Topography ---
if PLOT_P3_TOPO:
    ga_diff_topo = np.mean(subject_data['diff_topo'], axis=0)
    
    # Find time indices for the P3 window
    t_start_idx = np.argmin(np.abs(times_vector - P3_TOPO_WINDOW[0]))
    t_end_idx = np.argmin(np.abs(times_vector - P3_TOPO_WINDOW[1]))
    
    # Average over time
    p3_topo_data = np.mean(ga_diff_topo[:, t_start_idx:t_end_idx+1], axis=1) * AMPLITUDE_SCALE_FACTOR
    vmax = np.nanmax(np.abs(p3_topo_data))
    
    fig_topo, ax_topo = plt.subplots(1, 1, figsize=(6, 5))
    im, _ = mne.viz.plot_topomap(
        p3_topo_data, 
        epochs_info, 
        axes=ax_topo, 
        cmap='RdBu_r', 
        vlim=(-vmax, vmax), 
        show=False,
        sensors=True,
        outlines='head'
    )
    
    cbar = plt.colorbar(im, ax=ax_topo, shrink=0.7)
    cbar.set_label('Amplitude Diff [µV]')
    ax_topo.set_title(f"Unexpected - Expected Difference\n({P3_TOPO_WINDOW[0]*1000:.0f} - {P3_TOPO_WINDOW[1]*1000:.0f} ms)")
    fig_topo.tight_layout()

plt.show(block=True)