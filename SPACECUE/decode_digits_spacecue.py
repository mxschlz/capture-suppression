import sys
import os
import glob
import pickle
import numpy as np
import pandas as pd
import mne
import seaborn as sns
import matplotlib
# Use 'Agg' backend unless we are in an interactive console (like sys.ps1 or IPython)
if not (hasattr(sys, 'ps1') or 'ipykernel' in sys.modules):
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, ttest_rel
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from mne.decoding import SlidingEstimator

# Ensure project root is in python path
try:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
except NameError:
    project_root = os.path.abspath(os.getcwd())
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import SPACECUE

def decode_digits_balanced(X_hp, y_hp, X_lp, y_lp, clf, cv, n_iter=3, seed=42):
    """
    Decodes multinomial digits (1-9) while strictly equalizing trial counts and
    digit distributions between HP and LP distractor conditions via repeated
    stratified subsampling.
    """
    rng = np.random.default_rng(seed)
    digits = np.unique(np.concatenate([y_hp, y_lp]))

    # Minimum count per digit between HP and LP to guarantee identical distributions
    min_per_digit = {d: min(np.sum(y_hp == d), np.sum(y_lp == d)) for d in digits}
    total_bal = sum(min_per_digit.values())
    if total_bal < len(digits) * cv.n_splits:
        return None, None, None

    dec_hp = SlidingEstimator(clf, n_jobs=-1, scoring='accuracy', verbose=False)
    dec_lp = SlidingEstimator(clf, n_jobs=-1, scoring='accuracy', verbose=False)
    dec_all = SlidingEstimator(clf, n_jobs=-1, scoring='accuracy', verbose=False)

    scores_hp_iters, scores_lp_iters, scores_all_iters = [], [], []

    for it in range(n_iter):
        hp_idx_bal, lp_idx_bal = [], []
        for d in digits:
            n_d = min_per_digit[d]
            if n_d > 0:
                hp_idx_bal.extend(rng.choice(np.where(y_hp == d)[0], n_d, replace=False))
                lp_idx_bal.extend(rng.choice(np.where(y_lp == d)[0], n_d, replace=False))

        hp_idx_bal = np.array(hp_idx_bal)
        lp_idx_bal = np.array(lp_idx_bal)

        X_hp_bal, y_hp_bal = X_hp[hp_idx_bal], y_hp[hp_idx_bal]
        X_lp_bal, y_lp_bal = X_lp[lp_idx_bal], y_lp[lp_idx_bal]

        # 1. HP distractor trials (strictly balanced size and digit frequencies)
        splits_hp = list(cv.split(X_hp_bal, y_hp_bal))
        folds_hp = []
        for tr, te in splits_hp:
            dec_hp.fit(X_hp_bal[tr], y_hp_bal[tr])
            folds_hp.append(dec_hp.score(X_hp_bal[te], y_hp_bal[te]))
        scores_hp_iters.append(np.mean(folds_hp, axis=0))

        # 2. LP distractor trials (strictly balanced size and digit frequencies)
        splits_lp = list(cv.split(X_lp_bal, y_lp_bal))
        folds_lp = []
        for tr, te in splits_lp:
            dec_lp.fit(X_lp_bal[tr], y_lp_bal[tr])
            folds_lp.append(dec_lp.score(X_lp_bal[te], y_lp_bal[te]))
        scores_lp_iters.append(np.mean(folds_lp, axis=0))

        # 3. Combined balanced dataset (50% HP, 50% LP)
        X_all_bal = np.concatenate([X_hp_bal, X_lp_bal], axis=0)
        y_all_bal = np.concatenate([y_hp_bal, y_lp_bal], axis=0)
        splits_all = list(cv.split(X_all_bal, y_all_bal))
        folds_all = []
        for tr, te in splits_all:
            dec_all.fit(X_all_bal[tr], y_all_bal[tr])
            folds_all.append(dec_all.score(X_all_bal[te], y_all_bal[te]))
        scores_all_iters.append(np.mean(folds_all, axis=0))

    return np.mean(scores_all_iters, axis=0), np.mean(scores_hp_iters, axis=0), np.mean(scores_lp_iters, axis=0)


def run_subject_decoding(sub_id, concatenated_epochs, times, clf, cv):
    if concatenated_epochs.metadata is None or 'Subject ID' not in concatenated_epochs.metadata.columns:
        print("  Error: 'Subject ID' not found in concatenated epochs metadata.")
        return None

    sub_mask = (concatenated_epochs.metadata['Subject ID'] == float(sub_id)) | (concatenated_epochs.metadata['Subject ID'] == sub_id)
    if not np.any(sub_mask):
        print(f"  Warning: Epochs not found for subject {sub_id} in concatenated file. Skipping.")
        return None

    try:
        epochs = concatenated_epochs[sub_mask].load_data()
        epochs.crop(times[0], times[-1], verbose=False)

        # Resample to 100 Hz to speed up decoding computations substantially
        epochs.resample(100, verbose=False)

        montage = mne.channels.make_standard_montage("easycap-M1")
        epochs.set_montage(montage, on_missing='ignore', verbose=False)

        # Extract digits
        metadata = epochs.metadata
        y_target = metadata['TargetDigit'].values
        y_distractor = metadata['SingletonDigit'].values
        y_control = metadata['Non-Singleton2Digit'].values

        # Identify HP vs LP distractor trials
        is_hp_dist = (metadata['DistractorProb'].values == 'high-probability')
        is_lp_dist = (metadata['DistractorProb'].values == 'low-probability')

        # Filter for valid digit trials (1-9) and LATERAL targets only (exclude frontal targets)
        is_lateral_target = (metadata['TargetLoc'].values != 'Front') & (metadata['TargetLoc'].isin(['Left', 'Right']).values)
        valid = (y_target >= 1) & (y_target <= 9) & (y_distractor >= 1) & (y_distractor <= 9) & (y_control >= 1) & (y_control <= 9) & is_lateral_target

        X = epochs.get_data()[valid]
        y_t = y_target[valid]
        y_d = y_distractor[valid]
        y_c = y_control[valid]

        hp_mask = is_hp_dist[valid]
        lp_mask = is_lp_dist[valid]

        # Subject results dict
        sub_results = {
            'times': epochs.times,
            'target_all': None, 'distractor_all': None, 'control_all': None,
            'target_hp': None, 'distractor_hp': None, 'control_hp': None,
            'target_lp': None, 'distractor_lp': None, 'control_lp': None,
        }

        # Run balanced decoders for Targets, Distractors, and Controls
        sub_seed = int(sub_id) if isinstance(sub_id, (int, float)) and not np.isnan(sub_id) else 42

        # 1. Targets (balanced HP vs LP)
        t_all, t_hp, t_lp = decode_digits_balanced(
            X[hp_mask], y_t[hp_mask], X[lp_mask], y_t[lp_mask], clf, cv, n_iter=3, seed=sub_seed * 10 + 1
        )
        sub_results['target_all'] = t_all
        sub_results['target_hp'] = t_hp
        sub_results['target_lp'] = t_lp

        # 2. Distractors (balanced HP vs LP)
        d_all, d_hp, d_lp = decode_digits_balanced(
            X[hp_mask], y_d[hp_mask], X[lp_mask], y_d[lp_mask], clf, cv, n_iter=3, seed=sub_seed * 10 + 2
        )
        sub_results['distractor_all'] = d_all
        sub_results['distractor_hp'] = d_hp
        sub_results['distractor_lp'] = d_lp

        # 3. Controls (balanced HP vs LP)
        c_all, c_hp, c_lp = decode_digits_balanced(
            X[hp_mask], y_c[hp_mask], X[lp_mask], y_c[lp_mask], clf, cv, n_iter=3, seed=sub_seed * 10 + 3
        )
        sub_results['control_all'] = c_all
        sub_results['control_hp'] = c_hp
        sub_results['control_lp'] = c_lp

        sub_results['age'] = float(metadata['Age'].dropna().iloc[0]) if 'Age' in metadata.columns else None

        print(f"    Successfully decoded sci-{sub_id} ({np.sum(hp_mask)} HP, {np.sum(lp_mask)} LP trials -> balanced)")
        return sub_results
    except Exception as e:
        print(f"    Error decoding subject {sub_id}: {e}")
        return None

def main():
    print("=========================================================")
    print("      SPACECUE EEG MULTINOMIAL DIGIT DECODING            ")
    print("=========================================================\n")

    plots_dir = os.path.join(project_root, "spacecue", "plots")
    os.makedirs(plots_dir, exist_ok=True)

    _, epochs_exist, _, epochs_fpath, _ = SPACECUE.check_if_concatenated_eeg_exists('spacecue')
    if not epochs_exist:
        print(f"Error: Concatenated epochs file not found at {epochs_fpath}")
        sys.exit(1)

    print(f"Loading concatenated epochs header from: {epochs_fpath}...")
    concatenated_epochs = mne.read_epochs(epochs_fpath, preload=False, verbose=False)

    from SPACECUE.subjects import subject_ids
    all_subjects = subject_ids
    times_range = [-0.2, 1.0]

    # Model definition: 9 classes, chance level is 11.1%
    clf = make_pipeline(StandardScaler(), LogisticRegression(solver='lbfgs', max_iter=200))
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    results = {}
    decoding_times = None

    cache_path = os.path.join(plots_dir, "digit_decoding_cache.pkl")
    cached_results = {}
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "rb") as f:
                cached_results = pickle.load(f)
            print(f"Loaded {len(cached_results)} previously decoded subjects from cache ({cache_path}).")
        except Exception as e:
            print(f"Warning: Could not read cache file ({e}). Recomputing.")

    for sub_id in all_subjects:
        if sub_id in cached_results:
            res = cached_results[sub_id]
            print(f"    Loaded sci-{sub_id} from cache.")
        else:
            res = run_subject_decoding(sub_id, concatenated_epochs, times_range, clf, cv)
            if res is not None:
                cached_results[sub_id] = res
                try:
                    with open(cache_path, "wb") as f:
                        pickle.dump(cached_results, f)
                except Exception:
                    pass
        if res is not None:
            results[sub_id] = res
            if decoding_times is None:
                decoding_times = res['times']

    if not results:
        print("Error: No subjects were decoded successfully.")
        sys.exit(1)

    cohorts = {'Young': [], 'Old': []}
    for sub_id, res in results.items():
        if res.get('age') is not None:
            if res['age'] < 36:
                cohorts['Young'].append(sub_id)
            else:
                cohorts['Old'].append(sub_id)
        else:
            print(f"Warning: Age missing for subject {sub_id}, skipping in cohort comparison.")

    # 1. Age Cohort Comparison (Young vs. Old)
    print("\nComputing statistics for Young vs. Old cohorts...")
    y_subs = cohorts['Young']
    o_subs = cohorts['Old']

    metrics = ['target_all', 'distractor_all', 'control_all']
    labels = {'target_all': 'Target', 'distractor_all': 'Distractor', 'control_all': 'Control'}
    colors = {'target_all': '#2CA02C', 'distractor_all': '#D62728', 'control_all': '#7F7F7F'}
    chance_level = 1.0 / 9.0

    # Age Split Figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharey=True)
    sns.set_theme(style="ticks", context="talk")

    report_lines = [
        "=========================================================",
        "      SPACECUE DIGIT DECODING COMPARISON REPORT          ",
        "=========================================================\n",
        f"Chance Level: {chance_level*100:.2f}% (1/9 classes)",
        f"Epoch window: {times_range[0]}s to {times_range[1]}s\n",
        "---------------------------------------------------------",
        "  1. COHORT COMPARISON (YOUNG VS. OLD ADULTS)",
        "---------------------------------------------------------"
    ]

    for idx, m in enumerate(metrics):
        ax = axes[idx]
        y_data = np.array([results[s][m] for s in y_subs])
        o_data = np.array([results[s][m] for s in o_subs])

        mean_y = y_data.mean(axis=0)
        sem_y = y_data.std(axis=0) / np.sqrt(len(y_data))
        mean_o = o_data.mean(axis=0)
        sem_o = o_data.std(axis=0) / np.sqrt(len(o_data))

        # Plot Young
        ax.plot(decoding_times * 1000, mean_y, color='#1F77B4', lw=2.5, label=f'Young (N={len(y_data)})')
        ax.fill_between(decoding_times * 1000, mean_y - sem_y, mean_y + sem_y, color='#1F77B4', alpha=0.15)

        # Plot Old
        ax.plot(decoding_times * 1000, mean_o, color='#FF7F0E', lw=2.5, label=f'Old (N={len(o_data)})')
        ax.fill_between(decoding_times * 1000, mean_o - sem_o, mean_o + sem_o, color='#FF7F0E', alpha=0.15)

        ax.axhline(chance_level, color='black', linestyle='--', alpha=0.5, label='Chance')
        ax.axvline(0, color='gray', linestyle=':', alpha=0.5)

        # Statistical comparison Young vs. Old (Cluster-Based Permutation Test)
        from mne.stats import permutation_cluster_test
        F_obs, clusters, cluster_p_values, H0 = permutation_cluster_test(
            [y_data, o_data], n_permutations=1000, threshold=None, out_type='indices', verbose=False
        )

        is_sig = np.zeros(len(decoding_times), dtype=bool)
        if clusters:
            for cluster, p_val in zip(clusters, cluster_p_values):
                if p_val < 0.05:
                    is_sig[cluster[0]] = True

        # Shade significant difference windows (p < 0.05)
        for i in range(len(decoding_times)):
            if is_sig[i]:
                ax.axvspan(decoding_times[i] * 1000, decoding_times[i] * 1000 + (decoding_times[1]-decoding_times[0])*1000,
                           color='purple', alpha=0.08, lw=0)

        # Report stats
        peak_y_idx = np.argmax(mean_y)
        peak_o_idx = np.argmax(mean_o)
        report_lines.append(f"\nMetric: {labels[m]}")
        report_lines.append(f"  Young Peak Accuracy: {mean_y[peak_y_idx]*100:.2f}% at {decoding_times[peak_y_idx]*1000:.0f} ms")
        report_lines.append(f"  Old Peak Accuracy:   {mean_o[peak_o_idx]*100:.2f}% at {decoding_times[peak_o_idx]*1000:.0f} ms")

        # List significant time intervals and calculate effect size per AGENTS.md rule
        sig_windows = []
        if clusters:
            for c_idx, (cluster, p_val) in enumerate(zip(clusters, cluster_p_values)):
                if p_val < 0.05:
                    indices = cluster[0]
                    # Temporal average across cluster bounds for each subject
                    y_clust = y_data[:, indices].mean(axis=1)
                    o_clust = o_data[:, indices].mean(axis=1)
                    # Two-sample Welch t-test on collapsed data
                    from scipy.stats import ttest_ind
                    t_res = ttest_ind(y_clust, o_clust, equal_var=False)
                    # Cohen's d (pooled)
                    n_y, n_o = len(y_clust), len(o_clust)
                    s_pooled = np.sqrt(((n_y - 1) * np.var(y_clust, ddof=1) + (n_o - 1) * np.var(o_clust, ddof=1)) / (n_y + n_o - 2))
                    d_val = (np.mean(y_clust) - np.mean(o_clust)) / s_pooled if s_pooled > 0 else 0.0
                    df_val = n_y + n_o - 2
                    sig_windows.append(f"{decoding_times[indices[0]]*1000:.0f}-{decoding_times[indices[-1]]*1000:.0f} ms (p={p_val}, t({df_val})={t_res.statistic:.3f}, d={d_val:.3f})")
        if sig_windows:
            report_lines.append(f"  Significant cohort difference windows: {', '.join(sig_windows)}")
        else:
            report_lines.append("  No significant cohort differences found.")

        ax.set_title(f"{labels[m]} Decoding", fontsize=14, fontweight='bold')
        ax.set_xlabel("Time (ms)")
        if idx == 0:
            ax.set_ylabel("Decoding Accuracy")
        ax.set_xlim(times_range[0] * 1000, times_range[1] * 1000)
        ax.legend(loc='upper right', frameon=False, fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.3)
        sns.despine(ax=ax)

    plt.suptitle("Multinomial Digit Decoding by Age Cohort", fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plot_age_path = os.path.join(plots_dir, "decoding_age_split.png")
    plt.savefig(plot_age_path, dpi=300, bbox_inches='tight')
    if hasattr(sys, 'ps1') or 'ipykernel' in sys.modules:
        plt.show()
    else:
        plt.close()

    # 2. Distractor Location Comparison (HP vs. LP Distractor Location)
    print("\nComputing statistics for HP vs. LP distractor locations...")
    report_lines.append("\n---------------------------------------------------------")
    report_lines.append("  2. LOCATION COMPARISON (HP VS. LP DISTRACTOR LOCATION)")
    report_lines.append("---------------------------------------------------------")

    valid_subjects_loc = [s for s in results if results[s]['target_hp'] is not None and results[s]['target_lp'] is not None]

    fig_loc, axes_loc = plt.subplots(1, 3, figsize=(18, 5.5), sharey=True)

    for idx, m in enumerate(metrics):
        ax = axes_loc[idx]
        m_hp = m.replace('_all', '_hp')
        m_lp = m.replace('_all', '_lp')

        hp_data = np.array([results[s][m_hp] for s in valid_subjects_loc])
        lp_data = np.array([results[s][m_lp] for s in valid_subjects_loc])

        mean_hp = hp_data.mean(axis=0)
        sem_hp = hp_data.std(axis=0) / np.sqrt(len(hp_data))
        mean_lp = lp_data.mean(axis=0)
        sem_lp = lp_data.std(axis=0) / np.sqrt(len(lp_data))

        # Plot HP Distractor Location
        ax.plot(decoding_times * 1000, mean_hp, color='#2CA02C', lw=2.5, label='HP Distractor Loc')
        ax.fill_between(decoding_times * 1000, mean_hp - sem_hp, mean_hp + sem_hp, color='#2CA02C', alpha=0.15)

        # Plot LP Distractor Location
        ax.plot(decoding_times * 1000, mean_lp, color='#D62728', lw=2.5, label='LP Distractor Loc')
        ax.fill_between(decoding_times * 1000, mean_lp - sem_lp, mean_lp + sem_lp, color='#D62728', alpha=0.15)

        ax.axhline(chance_level, color='black', linestyle='--', alpha=0.5, label='Chance')
        ax.axvline(0, color='gray', linestyle=':', alpha=0.5)

        # Statistical comparison HP vs. LP (Cluster-Based Permutation Paired Test)
        from mne.stats import permutation_cluster_1samp_test
        diff_data = hp_data - lp_data
        T_obs, clusters, cluster_p_values, H0 = permutation_cluster_1samp_test(
            diff_data, n_permutations=1000, tail=0, threshold=None, out_type='indices', verbose=False
        )

        is_sig_loc = np.zeros(len(decoding_times), dtype=bool)
        if clusters:
            for cluster, p_val in zip(clusters, cluster_p_values):
                if p_val < 0.05:
                    is_sig_loc[cluster[0]] = True

        # Shade significant difference windows (p < 0.05)
        for i in range(len(decoding_times)):
            if is_sig_loc[i]:
                ax.axvspan(decoding_times[i] * 1000, decoding_times[i] * 1000 + (decoding_times[1]-decoding_times[0])*1000,
                           color='purple', alpha=0.08, lw=0)

        # Report stats
        peak_hp_idx = np.argmax(mean_hp)
        peak_lp_idx = np.argmax(mean_lp)
        report_lines.append(f"\nMetric: {labels[m]}")
        report_lines.append(f"  HP Loc Peak Accuracy: {mean_hp[peak_hp_idx]*100:.2f}% at {decoding_times[peak_hp_idx]*1000:.0f} ms")
        report_lines.append(f"  LP Loc Peak Accuracy: {mean_lp[peak_lp_idx]*100:.2f}% at {decoding_times[peak_lp_idx]*1000:.0f} ms")

        # List significant time intervals and calculate effect size per AGENTS.md rule
        sig_windows_loc = []
        if clusters:
            for c_idx, (cluster, p_val) in enumerate(zip(clusters, cluster_p_values)):
                if p_val < 0.05:
                    indices = cluster[0]
                    # Temporal average of difference wave for each subject across cluster bounds
                    diff_sub = diff_data[:, indices].mean(axis=1)
                    # Cohen's dz = Mean / SD on subject-level collapsed 1D array
                    mean_diff = np.mean(diff_sub)
                    sd_diff = np.std(diff_sub, ddof=1)
                    dz = mean_diff / sd_diff if sd_diff > 0 else 0.0
                    n_sub = len(diff_sub)
                    t_val = mean_diff / (sd_diff / np.sqrt(n_sub)) if sd_diff > 0 else 0.0
                    df_val = n_sub - 1
                    sig_windows_loc.append(f"{decoding_times[indices[0]]*1000:.0f}-{decoding_times[indices[-1]]*1000:.0f} ms (p={p_val}, t({df_val})={t_val:.3f}, dz={dz:.3f})")
        if sig_windows_loc:
            report_lines.append(f"  Significant location difference windows: {', '.join(sig_windows_loc)}")
        else:
            report_lines.append("  No significant location differences found.")

        ax.set_title(f"{labels[m]} Decoding", fontsize=14, fontweight='bold')
        ax.set_xlabel("Time (ms)")
        if idx == 0:
            ax.set_ylabel("Decoding Accuracy")
        ax.set_xlim(times_range[0] * 1000, times_range[1] * 1000)
        ax.legend(loc='upper right', frameon=False, fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.3)
        sns.despine(ax=ax)

    plt.suptitle("Multinomial Digit Decoding by Distractor Location (HP vs. LP)", fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plot_loc_path = os.path.join(plots_dir, "decoding_hp_vs_lp.png")
    plt.savefig(plot_loc_path, dpi=300, bbox_inches='tight')
    if hasattr(sys, 'ps1') or 'ipykernel' in sys.modules:
        plt.show()
    else:
        plt.close()

    # Save and output report
    report_text = "\n".join(report_lines)
    report_path = os.path.join(plots_dir, "decoding_report.txt")
    with open(report_path, "w") as f:
        f.write(report_text)

    print(report_text)
    print(f"\nSaved age-split plot to: {plot_age_path}")
    print(f"Saved location plot to:  {plot_loc_path}")
    print(f"Saved decoding report to: {report_path}")

if __name__ == "__main__":
    main()
