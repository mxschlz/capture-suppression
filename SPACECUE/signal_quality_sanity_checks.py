"""
SPACECUE EEG Signal-Quality Sanity Checks (Younger vs. Older Adults)
====================================================================

This script performs data-quality sanity checks on the concatenated SPACECUE
epochs and compares signal-quality metrics between the two age cohorts
(Young: Age <= 35, Old: Age > 35).

Analyses
--------
1. Total number of AutoReject-rejected epochs per subject (read from the
   per-subject AutoReject log files saved during preprocessing:
   derivatives/epoching/sci-{id}/eeg/sci-{id}_task-spacecue-epo_log.npz,
   key 'bad_epochs').
1b. Number of ICA components rejected per subject during preprocessing
    (extended-Infomax ICA + ICLabel; class not 'brain'/'other' with p > 0.9),
    compared as absolute count and as % of fitted components (the number of
    components follows the subject's data rank). Breakdown by ICLabel class
    is exploratory (Mann-Whitney U, Holm-corrected).
2. Average Global Field Power (GFP) between groups. GFP is computed as the
   spatial standard deviation across all 64 average-referenced EEG channels
   of the subject-level evoked response. Because the data are
   average-referenced, the spatial mean is ~0 and GFP corresponds to the
   root-mean-square across the sensor field. Group GFP trajectories are
   compared with a non-parametric cluster-based permutation test (controls
   the family-wise error rate across time points).
3. Event-related signal-to-noise ratio (SNR) in the 0-500 ms post-stimulus
   window, estimated with the alternating-inversion (Schimmel, 1967)
   procedure:
       - Signal: RMS across channels and time of the plain trial average
         (the event-related potential).
       - Noise: RMS across channels and time of the average obtained after
         multiplying every other trial by -1. This cancels the
         stimulus-locked signal while residual (non-phase-locked) activity
         is attenuated by 1/sqrt(N) exactly as in the signal average, so
         the ratio of the two RMS values is a trial-count-independent
         estimate of the single-trial evoked SNR.
       - SNR = Signal RMS / Noise RMS (also reported in dB).

Statistical reporting follows the workspace conventions: test statistics,
degrees of freedom, exact (unrounded) p-values and effect sizes are printed
and stored; Shapiro-Wilk (normality) and Levene (homoscedasticity) checks
precede each parametric test; a Mann-Whitney U test is reported as a
non-parametric robustness check. Cluster effect sizes are computed on
subject-level GFP averaged across each significant cluster's temporal
bounds (never on point-wise T-values, which would be circular).

The script is written console-first: it can be executed sequentially
line-by-line in an interactive IPython/Jupyter console.
"""

import sys
import os
import numpy as np
import pandas as pd
import mne
import matplotlib
# Use 'Agg' backend unless we are in an interactive console (like sys.ps1 or IPython)
if not (hasattr(sys, "ps1") or "ipykernel" in sys.modules):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pingouin as pg
from mne.stats import permutation_cluster_test

# Ensure project root is in python path
try:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
except NameError:
    # Fallback when running directly in interactive python console
    project_root = os.path.abspath(os.getcwd())
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import SPACECUE
from SPACECUE.subjects import subject_ids

sns.set_theme(context="talk", style="ticks")
if hasattr(sys, "ps1") or "ipykernel" in sys.modules:
    plt.ion()  # interactive plotting feedback when running in a console

mne.set_log_level("WARNING")

# =============================================================================
# 1. General Script Parameters
# =============================================================================
PLOTS_DIR = os.path.join(project_root, "SPACECUE", "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

DATA_PATH = SPACECUE.get_data_path()

# Cohort assignment (identical to SPACECUE.behavioral_age_comparison):
# Young: Age <= 35, Old: Age > 35
COHORT_AGE_THRESHOLD = 35
COHORTS = ["Young", "Old"]

# Time window kept after loading the concatenated epochs (seconds).
# The epochs were saved without baseline correction (1 Hz high-pass), which
# is appropriate for GFP/SNR analyses (baseline correction would only add a
# constant offset per channel and is unnecessary for spatial-deviation
# metrics).
EPOCH_TMIN, EPOCH_TMAX = -0.2, 0.8

# Window for signal & noise estimation (post-stimulus onset, seconds)
SNR_TMIN, SNR_TMAX = 0.0, 0.5

# Window for the subject-level mean-GFP group comparison (seconds)
GFP_MEAN_TMIN, GFP_MEAN_TMAX = 0.0, 0.5

# Cluster-based permutation test parameters (for the GFP time courses)
N_PERMUTATIONS_CLUSTER = 10000
ALPHA_STAT_CLUSTER = 0.05
SEED = 42
N_JOBS = 5

AMPLITUDE_SCALE_FACTOR = 1e6  # Volts -> Microvolts

# Colorblind-safe cohort colors (no raw primary colors)
_palette = sns.color_palette("colorblind")
COHORT_COLORS = {"Young": _palette[0], "Old": _palette[1]}

# --- Report handling: print and collect everything in one text report ---
report_lines = []


def report(msg=""):
    """Print a message and append it to the saved text report."""
    print(msg)
    report_lines.append(str(msg))


# =============================================================================
# 2. Load the concatenated epochs
# =============================================================================
report("=========================================================")
report("   SPACECUE SIGNAL-QUALITY SANITY CHECKS (YOUNG VS. OLD)   ")
report("=========================================================")
report()

# Load all subjects' autoreject-cleaned, ICA-corrected, average-referenced
# epochs (1-40 Hz band-pass, 250 Hz, t = -1..1 s) and crop to the analysis
# window to save memory.
# NOTE: We deliberately load WITHOUT preloading: reading the entire ~8 GB
# concatenated file in a single operation fails on Windows with
# "[Errno 22] Invalid argument" (single reads >2 GB are not supported).
# Instead, the data are read lazily per subject inside the loop below,
# which also keeps the memory footprint low.
from SPACECUE import check_if_concatenated_eeg_exists
_, epochs_available, _, epochs_file_path, _ = check_if_concatenated_eeg_exists("spacecue")
if not epochs_available:
    raise FileNotFoundError(
        f"Concatenated epochs file not found at {epochs_file_path}. "
        f"Run SPACECUE.concatenate_eeg_and_save('spacecue') first.")
epochs = mne.read_epochs(epochs_file_path, preload=False)
# NOTE: pick_types()/crop() require preloaded data, so channels are instead
# selected by index when reading each subject below.
eeg_picks = np.array([i for i, ch_type in enumerate(epochs.get_channel_types())
                      if ch_type == "eeg"])
report(f"Loaded {len(epochs)} epochs from {len(epochs.ch_names)} channels "
       f"({len(eeg_picks)} EEG), sfreq = {epochs.info['sfreq']} Hz, "
       f"t = [{epochs.tmin}, {epochs.tmax}] s")

# Time axes. NOTE: crop() requires preloaded data, so we keep the full
# -1..1 s epochs on disk and select all analysis/signal windows via
# boolean time masks (applied to the lazily read per-subject data).
times_full = epochs.times                # full epoch time axis (s), stimulus-locked
analysis_time_mask = (times_full >= EPOCH_TMIN) & (times_full <= EPOCH_TMAX)
times = times_full[analysis_time_mask]   # analysis-window time axis
times_ms = times * 1000.0
metadata = epochs.metadata.reset_index(drop=True)

# =============================================================================
# 3. Per-subject extraction of signal-quality metrics
# =============================================================================
# Time masks reused below (defined on the FULL epoch time axis, because the
# per-subject data are read lazily with all 501 samples)
snr_time_mask = (times_full >= SNR_TMIN) & (times_full <= SNR_TMAX)

subject_metrics = []     # list of dicts -> DataFrame at the end
gfp_timeseries = {}      # subject_id -> GFP time course (uV)

report("--- Per-subject metrics ---")
report(f"{'Sub':>5} {'Age':>4} {'Cohort':>7} {'n_tot':>6} {'n_rej':>6} {'%rej':>6} "
       f"{'GFPmean':>8} {'Signal':>8} {'Noise':>8} {'SNR':>6} {'SNR(dB)':>8}")
report("-" * 88)

for sub_id in subject_ids:
    sub_str = f"sci-{sub_id}"
    trial_mask = (metadata["Subject ID"] == float(sub_id)).values
    n_kept = int(trial_mask.sum())
    if n_kept == 0:
        report(f"  Warning: no epochs found for subject {sub_id} in the "
               f"concatenated file. Skipping.")
        continue

    # --- Demographics / cohort assignment from the epoch metadata ---
    age = metadata.loc[trial_mask, "Age"].dropna().iloc[0]
    cohort = "Young" if age <= COHORT_AGE_THRESHOLD else "Old"

    # --- (1) Rejected epochs from the AutoReject log ---
    # The log's 'bad_epochs' array refers to ALL epochs before AutoReject
    # rejection (one entry per acquired trial).
    log_path = os.path.join(DATA_PATH, "derivatives", "epoching", sub_str, "eeg",
                            f"{sub_str}_task-spacecue-epo_log.npz")
    if os.path.exists(log_path):
        bad_epochs = np.load(log_path)["bad_epochs"]
        n_total = int(len(bad_epochs))
        n_rejected = int(bad_epochs.sum())
    else:
        report(f"  Warning: AutoReject log not found for subject {sub_id} "
               f"({log_path}). Rejection metrics set to NaN.")
        n_total, n_rejected = np.nan, np.nan

    # Consistency check: epochs kept in the concatenated file must equal
    # n_total - n_rejected
    if n_total == n_total and (n_total - n_rejected) != n_kept:
        report(f"  Warning sci-{sub_id}: log implies {n_total - n_rejected} kept "
               f"epochs but {n_kept} found in concatenated file!")

    pct_rejected = 100.0 * n_rejected / n_total if n_total == n_total else np.nan

    # --- (2) Global Field Power of the subject-level evoked response ---
    # GFP = spatial SD across channels (Lehmann; with average reference the
    # spatial mean is ~0, so this equals the RMS across the sensor field).
    # Lazy per-subject read from the (non-preloaded) concatenated epochs,
    # restricted to EEG channels:
    sub_data = epochs.get_data(item=np.where(trial_mask)[0])[:, eeg_picks, :]  # (n_trials, n_ch, n_time), Volts
    evoked_signal = sub_data.mean(axis=0)                 # (n_ch, n_time), Volts
    gfp = evoked_signal.std(axis=0, ddof=1) * AMPLITUDE_SCALE_FACTOR  # uV
    gfp_timeseries[sub_id] = gfp[analysis_time_mask]  # restricted to the analysis window

    # Subject-level mean GFP in the post-stimulus window
    gfp_mean = gfp[snr_time_mask].mean()

    # --- (3) SNR via alternating trial inversion (Schimmel, 1967) ---
    # Signal: plain average (event-related potential).
    # Noise: average with every other trial multiplied by -1, which cancels
    # the stimulus-locked response and retains non-phase-locked residual
    # activity. Both averages attenuate noise by 1/sqrt(N), so the RMS
    # ratio is a single-trial SNR estimate independent of trial count.
    sub_data_snr = sub_data[:, :, snr_time_mask]           # (n_trials, n_ch, n_time)
    signs = np.where(np.arange(len(sub_data_snr)) % 2 == 0, 1.0, -1.0)
    evoked_noise = (sub_data_snr * signs[:, None, None]).mean(axis=0)

    signal_rms_uv = np.sqrt(np.mean(evoked_signal[:, snr_time_mask] ** 2)) * AMPLITUDE_SCALE_FACTOR
    noise_rms_uv = np.sqrt(np.mean(evoked_noise ** 2)) * AMPLITUDE_SCALE_FACTOR
    snr = signal_rms_uv / noise_rms_uv
    snr_db = 20.0 * np.log10(snr)

    report(f"{sub_id:>5} {age:>4.0f} {cohort:>7} {n_total:>6.0f} {n_rejected:>6.0f} "
           f"{pct_rejected:>6.2f} {gfp_mean:>8.3f} {signal_rms_uv:>8.3f} "
           f"{noise_rms_uv:>8.3f} {snr:>6.3f} {snr_db:>8.2f}")

    subject_metrics.append(dict(
        subject=sub_id, age=age, cohort=cohort,
        n_total=n_total, n_kept=n_kept, n_rejected=n_rejected,
        pct_rejected=pct_rejected,
        gfp_mean_0_500ms=gfp_mean,
        signal_rms_uv=signal_rms_uv, noise_rms_uv=noise_rms_uv,
        snr=snr, snr_db=snr_db,
    ))

del sub_data, sub_data_snr, evoked_signal, evoked_noise  # memory guardrail

df = pd.DataFrame(subject_metrics)
young_ids = df.loc[df["cohort"] == "Young", "subject"].tolist()
old_ids = df.loc[df["cohort"] == "Old", "subject"].tolist()

# --- Demographics ---
report()
report("--- Demographics ---")
for coh in COHORTS:
    ages = df.loc[df["cohort"] == coh, "age"]
    report(f"  {coh}: N = {len(ages)}, Age M = {ages.mean():.1f} "
           f"(SD = {ages.std(ddof=1):.1f}, range {ages.min():.0f}-{ages.max():.0f})")
report()

# =============================================================================
# 4. Helper: group comparison with assumption checks & effect sizes
# =============================================================================
def group_comparison(young_vals, old_vals, label):
    """Independent-samples comparison Young vs. Old with full reporting.

    Runs Shapiro-Wilk normality checks per group, Levene's test for
    homoscedasticity, a Welch t-test (parametric, does not assume equal
    variances), Cohen's d, and a Mann-Whitney U test as a non-parametric
    robustness check. All statistics are printed and returned as a dict.
    """
    young_vals = np.asarray(young_vals, dtype=float)
    old_vals = np.asarray(old_vals, dtype=float)

    report(f"\n--- Group comparison: {label} (Young vs. Old) ---")
    report(f"  Young: N = {len(young_vals)}, M = {np.mean(young_vals):.4f}, "
           f"SD = {np.std(young_vals, ddof=1):.4f}")
    report(f"  Old:   N = {len(old_vals)}, M = {np.mean(old_vals):.4f}, "
           f"SD = {np.std(old_vals, ddof=1):.4f}")

    # Assumption checks
    sw_y = pg.normality(young_vals).iloc[0]
    sw_o = pg.normality(old_vals).iloc[0]
    lev = pg.homoscedasticity([young_vals, old_vals], method="levene").iloc[0]
    report(f"  Shapiro-Wilk normality: Young W = {sw_y['W']:.4f}, p = {sw_y['pval']:.6g} "
           f"({'ok' if sw_y['pval'] > 0.05 else 'VIOLATED'}); "
           f"Old W = {sw_o['W']:.4f}, p = {sw_o['pval']:.6g} "
           f"({'ok' if sw_o['pval'] > 0.05 else 'VIOLATED'})")
    report(f"  Levene homoscedasticity: W = {lev['W']:.4f}, p = {lev['pval']:.6g} "
           f"({'ok' if lev['pval'] > 0.05 else 'VIOLATED'})")

    # Welch t-test (pingouin's correction=True applies Welch's correction,
    # i.e. no homoscedasticity assumption) + Cohen's d
    tt = pg.ttest(young_vals, old_vals, correction=True).iloc[0]
    report(f"  Welch t-test: t({tt['dof']:.1f}) = {tt['T']:.4f}, "
           f"p = {tt['p-val']:.10g}, Cohen's d = {tt['cohen-d']:.4f}")

    # Non-parametric robustness check
    mwu = pg.mwu(young_vals, old_vals).iloc[0]
    report(f"  Mann-Whitney U (robustness): U = {mwu['U-val']:.1f}, "
           f"p = {mwu['p-val']:.10g}, RBC = {mwu['RBC']:.4f}")

    return dict(t=float(tt["T"]), dof=float(tt["dof"]), p=float(tt["p-val"]),
                cohen_d=float(tt["cohen-d"]), mwu_p=float(mwu["p-val"]))


# =============================================================================
# 5. ANALYSIS 1: Rejected epochs (AutoReject) between groups
# =============================================================================
report("\n" + "=" * 60)
report("   ANALYSIS 1: AUTOREJECT-REJECTED EPOCHS (YOUNG VS. OLD)")
report("=" * 60)

rej_stats_count = group_comparison(
    df.loc[df["cohort"] == "Young", "n_rejected"],
    df.loc[df["cohort"] == "Old", "n_rejected"],
    "Number of rejected epochs")

rej_stats_pct = group_comparison(
    df.loc[df["cohort"] == "Young", "pct_rejected"],
    df.loc[df["cohort"] == "Old", "pct_rejected"],
    "Percentage of rejected epochs (%)")

# --- Plot: rejected epochs per group (mean +/- SEM + subject-level dots) ---
fig_rej, axes_rej = plt.subplots(1, 2, figsize=(12, 5))
for ax, col, ylabel in [
    (axes_rej[0], "n_rejected", "Rejected Epochs (count)"),
    (axes_rej[1], "pct_rejected", "Rejected Epochs (%)"),
]:
    for ci, coh in enumerate(COHORTS):
        vals = df.loc[df["cohort"] == coh, col].astype(float)
        ax.bar(ci, vals.mean(), yerr=vals.sem(), color=COHORT_COLORS[coh],
               width=0.6, alpha=0.85, capsize=6,
               error_kw=dict(lw=2, capthick=2))
        # Jittered subject-level dots for transparency about group spread
        rng = np.random.default_rng(SEED)
        jitter = rng.uniform(-0.12, 0.12, size=len(vals))
        ax.scatter(np.full(len(vals), ci) + jitter, vals, color="black",
                   s=28, alpha=0.55, zorder=3)
    ax.set_xticks(range(len(COHORTS)), COHORTS)
    ax.set_ylabel(ylabel)
    ax.spines[["top", "right"]].set_visible(False)
fig_rej.suptitle("AutoReject Epoch Rejection Rate by Age Cohort", y=1.02)
fig_rej.tight_layout()
fig_rej_path = os.path.join(PLOTS_DIR, "signal_quality_rejected_epochs.svg")
fig_rej.savefig(fig_rej_path, dpi=300, bbox_inches="tight")
report(f"\nSaved rejected-epochs plot to: {fig_rej_path}")
if not (hasattr(sys, "ps1") or "ipykernel" in sys.modules):
    plt.close(fig_rej)

# =============================================================================
# 5b. ANALYSIS 1b: ICA-rejected components between groups
# =============================================================================
# Extended-Infomax ICA was fitted per subject on the 1-100 Hz filtered,
# average-referenced continuous data (SPACECUE/preproc_main.py). A component
# was removed if its most probable ICLabel class was not 'brain'/'other' and
# that class probability exceeded 0.9.
#   - Excluded indices: *_ica_labels.txt (authoritative; the saved ICA's
#     `exclude` attribute stays empty because the indices were passed
#     directly to ica.apply()).
#   - Number of fitted components and ICLabel class of every component:
#     saved ICA object (`n_components_`, `labels_` written by mne_icalabel).
# The ICA was fitted with n_components=None, so the number of components is
# set by the data rank (reduced by the average reference and by interpolated
# channels) and differs between subjects -> both the absolute count and the
# percentage of fitted components are compared.
report("\n" + "=" * 60)
report("   ANALYSIS 1b: ICA-REJECTED COMPONENTS (YOUNG VS. OLD)")
report("=" * 60)

# ICLabel classes that can lead to rejection (keys as stored in ica.labels_)
ICA_REJECT_CLASSES = {"eog": "Ocular", "muscle": "Muscle", "ecg": "Cardiac",
                      "line_noise": "Line Noise", "ch_noise": "Channel Noise"}
ica_class_cols = [f"ica_rej_{cls}" for cls in ICA_REJECT_CLASSES]

report(f"{'Sub':>5} {'Cohort':>7} {'n_IC':>5} {'IC_rej':>7} {'%rej':>6}  "
       + " ".join(f"{cls:>10}" for cls in ICA_REJECT_CLASSES))
report("-" * 90)

ica_metrics = []
for sub_id, cohort in zip(df["subject"], df["cohort"]):
    sub_str = f"sci-{sub_id}"
    ica_dir = os.path.join(DATA_PATH, "derivatives", "preprocessing", sub_str, "eeg")
    excl_path = os.path.join(ica_dir, f"{sub_str}_task-spacecue_ica_labels.txt")
    ica_path = os.path.join(ica_dir, f"{sub_str}_task-spacecue_ica.fif")
    row = dict(subject=sub_id, n_ica_fitted=np.nan, n_ica_rejected=np.nan,
               pct_ica_rejected=np.nan, **{c: np.nan for c in ica_class_cols})
    if not (os.path.exists(excl_path) and os.path.exists(ica_path)):
        report(f"  Warning: ICA files not found for subject {sub_id} ({ica_dir}). "
               f"ICA metrics set to NaN.")
        ica_metrics.append(row)
        continue

    with open(excl_path) as f:
        excluded = {int(line) for line in f if line.strip()}  # empty file = 0 rejected
    ica = mne.preprocessing.read_ica(ica_path, verbose=False)
    row["n_ica_fitted"] = int(ica.n_components_)
    row["n_ica_rejected"] = len(excluded)
    row["pct_ica_rejected"] = 100.0 * len(excluded) / ica.n_components_

    # ICLabel (argmax) class of each excluded component
    if ica.labels_:
        for cls in ICA_REJECT_CLASSES:
            comp_idx = {int(i) for i in ica.labels_.get(cls, [])}
            row[f"ica_rej_{cls}"] = len(excluded & comp_idx)
        # Consistency check: every excluded component must carry a rejectable class
        n_classified = sum(row[c] for c in ica_class_cols)
        if n_classified != len(excluded):
            report(f"  Warning sci-{sub_id}: {len(excluded)} excluded components but "
                   f"only {n_classified} carry a rejectable ICLabel class!")
    else:
        report(f"  Warning sci-{sub_id}: saved ICA has no ICLabel labels; "
               f"class breakdown set to NaN.")
    del ica

    report(f"{sub_id:>5} {cohort:>7} {row['n_ica_fitted']:>5.0f} "
           f"{row['n_ica_rejected']:>7.0f} {row['pct_ica_rejected']:>6.2f}  "
           + " ".join(f"{row[c]:>10.0f}" for c in ica_class_cols))
    ica_metrics.append(row)

df = df.merge(pd.DataFrame(ica_metrics), on="subject", how="left")
df_ica = df.dropna(subset=["n_ica_rejected"])

# Number of fitted components per cohort (context for the absolute counts)
report("\n--- Number of fitted ICA components (data rank) ---")
for coh in COHORTS:
    n_fit = df_ica.loc[df_ica["cohort"] == coh, "n_ica_fitted"]
    report(f"  {coh}: M = {n_fit.mean():.2f} (SD = {n_fit.std(ddof=1):.2f}, "
           f"range {n_fit.min():.0f}-{n_fit.max():.0f})")

ica_stats_count = group_comparison(
    df_ica.loc[df_ica["cohort"] == "Young", "n_ica_rejected"],
    df_ica.loc[df_ica["cohort"] == "Old", "n_ica_rejected"],
    "Number of rejected ICA components")

ica_stats_pct = group_comparison(
    df_ica.loc[df_ica["cohort"] == "Young", "pct_ica_rejected"],
    df_ica.loc[df_ica["cohort"] == "Old", "pct_ica_rejected"],
    "Percentage of fitted ICA components rejected (%)")

# --- Breakdown by ICLabel class (exploratory) ---
# Per-class counts are sparse, zero-inflated integers -> Mann-Whitney U tests,
# Holm-corrected across the classes that were tested. Classes without any
# rejected component in either cohort are listed but not tested.
df_ica_cls = df_ica.dropna(subset=ica_class_cols)
report("\n--- Rejected ICA components by ICLabel class "
       "(exploratory; Mann-Whitney U, Holm-corrected) ---")
class_rows = []
for cls, cls_label in ICA_REJECT_CLASSES.items():
    vals_y = df_ica_cls.loc[df_ica_cls["cohort"] == "Young", f"ica_rej_{cls}"].astype(float)
    vals_o = df_ica_cls.loc[df_ica_cls["cohort"] == "Old", f"ica_rej_{cls}"].astype(float)
    cls_row = dict(ic_class=cls_label,
                   young_mean=vals_y.mean(), young_sd=vals_y.std(ddof=1),
                   young_n_any=int((vals_y > 0).sum()), young_n=len(vals_y),
                   old_mean=vals_o.mean(), old_sd=vals_o.std(ddof=1),
                   old_n_any=int((vals_o > 0).sum()), old_n=len(vals_o),
                   U=np.nan, p_unc=np.nan, p_holm=np.nan, RBC=np.nan)
    if vals_y.sum() + vals_o.sum() > 0:
        mwu = pg.mwu(vals_y, vals_o).iloc[0]
        cls_row.update(U=float(mwu["U-val"]), p_unc=float(mwu["p-val"]),
                       RBC=float(mwu["RBC"]))
    class_rows.append(cls_row)
ica_class_stats = pd.DataFrame(class_rows)
tested = ica_class_stats["p_unc"].notna()
if tested.any():
    ica_class_stats.loc[tested, "p_holm"] = pg.multicomp(
        ica_class_stats.loc[tested, "p_unc"].values, method="holm")[1]
for _, r in ica_class_stats.iterrows():
    msg = (f"  {r['ic_class']:<14} Young M = {r['young_mean']:.3f} "
           f"(SD = {r['young_sd']:.3f}, {r['young_n_any']}/{r['young_n']} with >=1) | "
           f"Old M = {r['old_mean']:.3f} "
           f"(SD = {r['old_sd']:.3f}, {r['old_n_any']}/{r['old_n']} with >=1)")
    if np.isnan(r["p_unc"]):
        msg += " | none rejected in either cohort, not tested"
    else:
        msg += (f" | U = {r['U']:.1f}, p = {r['p_unc']:.10g}, "
                f"p_holm = {r['p_holm']:.10g}, RBC = {r['RBC']:.4f}")
    report(msg)

# --- Plot: rejected ICA components (count, % of fitted, by ICLabel class) ---
fig_ica, axes_ica = plt.subplots(1, 3, figsize=(20, 5.5),
                                 gridspec_kw=dict(width_ratios=[1, 1, 2]))
for ax, col, ylabel in [
    (axes_ica[0], "n_ica_rejected", "Rejected Components (count)"),
    (axes_ica[1], "pct_ica_rejected", "Rejected Components (% of fitted)"),
]:
    for ci, coh in enumerate(COHORTS):
        vals = df_ica.loc[df_ica["cohort"] == coh, col].astype(float)
        ax.bar(ci, vals.mean(), yerr=vals.sem(), color=COHORT_COLORS[coh],
               width=0.6, alpha=0.85, capsize=6,
               error_kw=dict(lw=2, capthick=2))
        # Jittered subject-level dots for transparency about group spread
        rng = np.random.default_rng(SEED)
        jitter = rng.uniform(-0.12, 0.12, size=len(vals))
        ax.scatter(np.full(len(vals), ci) + jitter, vals, color="black",
                   s=28, alpha=0.55, zorder=3)
    ax.set_xticks(range(len(COHORTS)), COHORTS)
    ax.set_ylabel(ylabel)
    ax.spines[["top", "right"]].set_visible(False)

# Class breakdown: grouped bars, mean +/- SEM per cohort
ax = axes_ica[2]
x_cls = np.arange(len(ICA_REJECT_CLASSES))
for ci, coh in enumerate(COHORTS):
    coh_df = df_ica_cls.loc[df_ica_cls["cohort"] == coh, ica_class_cols].astype(float)
    ax.bar(x_cls + (ci - 0.5) * 0.4, coh_df.mean().values, yerr=coh_df.sem().values,
           width=0.36, color=COHORT_COLORS[coh], alpha=0.85, capsize=4,
           error_kw=dict(lw=1.5, capthick=1.5), label=coh)
ax.set_xticks(x_cls, list(ICA_REJECT_CLASSES.values()), rotation=30, ha="right")
ax.set_xlabel("ICLabel Class")
ax.set_ylabel("Rejected Components (count)")
ax.legend(frameon=False)
ax.spines[["top", "right"]].set_visible(False)
fig_ica.suptitle("ICA Component Rejection by Age Cohort", y=1.02)
fig_ica.tight_layout()
fig_ica_path = os.path.join(PLOTS_DIR, "signal_quality_ica_components.svg")
fig_ica.savefig(fig_ica_path, dpi=300, bbox_inches="tight")
report(f"\nSaved ICA-components plot to: {fig_ica_path}")
if not (hasattr(sys, "ps1") or "ipykernel" in sys.modules):
    plt.close(fig_ica)

# =============================================================================
# 6. ANALYSIS 2: Global Field Power (GFP) between groups
# =============================================================================
report("\n" + "=" * 60)
report("   ANALYSIS 2: GLOBAL FIELD POWER (YOUNG VS. OLD)")
report("=" * 60)

# Subject-level GFP time courses (uV), ordered by cohort
gfp_young = np.array([gfp_timeseries[s] for s in young_ids])   # (n_young, n_times)
gfp_old = np.array([gfp_timeseries[s] for s in old_ids])       # (n_old, n_times)

# --- Cluster-based permutation test across time points ---
# Two-sample test (F-statistic, equivalent to a two-tailed t-test squared
# for two groups) on the subject-level GFP trajectories; controls the
# family-wise error rate across the whole time axis.
report(f"\nRunning cluster-based permutation test on GFP time courses "
       f"({N_PERMUTATIONS_CLUSTER} permutations, seed = {SEED}) ...")
F_obs, clusters, cluster_pvals, H0 = permutation_cluster_test(
    [gfp_young, gfp_old],
    n_permutations=N_PERMUTATIONS_CLUSTER,
    tail=1,   # F-statistics are inherently one-sided (avoids MNE warning)
    seed=SEED,
    n_jobs=N_JOBS,
)

report("\n--- Cluster-based permutation test (GFP, Young vs. Old) ---")
report(f"  Number of clusters formed: {len(clusters)}")
sig_clusters = []
for c_idx, (cluster, p_val) in enumerate(zip(clusters, cluster_pvals)):
    idx = cluster[0]
    t_start_ms, t_end_ms = times_ms[idx[0]], times_ms[idx[-1]]
    sig = p_val < ALPHA_STAT_CLUSTER
    report(f"  Cluster {c_idx + 1}: {t_start_ms:6.1f} - {t_end_ms:6.1f} ms, "
           f"cluster p = {p_val:.10g}{'  * SIGNIFICANT' if sig else ''}")
    if sig:
        sig_clusters.append((idx, p_val))

# Effect sizes for significant clusters: collapse the raw GFP across the
# cluster's temporal bounds per subject (treating the cluster as a single,
# indivisible neurophysiological response), then compute one between-group
# Cohen's d on these subject-level averages. Point-wise T-value averaging
# would be circular because the cluster was defined by a threshold on
# those very T-values.
for c_idx, (idx, p_val) in enumerate(sig_clusters):
    t_start_ms, t_end_ms = times_ms[idx[0]], times_ms[idx[-1]]
    gfp_cluster_young = gfp_young[:, idx].mean(axis=1)
    gfp_cluster_old = gfp_old[:, idx].mean(axis=1)
    n_y, n_o = len(gfp_cluster_young), len(gfp_cluster_old)
    pooled_sd = np.sqrt(((n_y - 1) * np.var(gfp_cluster_young, ddof=1) +
                         (n_o - 1) * np.var(gfp_cluster_old, ddof=1)) / (n_y + n_o - 2))
    d_cluster = (np.mean(gfp_cluster_young) - np.mean(gfp_cluster_old)) / pooled_sd
    tt_cluster = pg.ttest(gfp_cluster_young, gfp_cluster_old, correction=True).iloc[0]
    report(f"  -> Cluster {c_idx + 1} effect size (subject-level mean GFP, "
           f"{t_start_ms:.0f}-{t_end_ms:.0f} ms): "
           f"Young M = {np.mean(gfp_cluster_young):.3f} uV, "
           f"Old M = {np.mean(gfp_cluster_old):.3f} uV, "
           f"Cohen's d = {d_cluster:.4f}, "
           f"post-hoc Welch t({tt_cluster['dof']:.1f}) = {tt_cluster['T']:.4f}, "
           f"p = {tt_cluster['p-val']:.10g}")

# --- Subject-level mean GFP in the 0-500 ms window ---
gfp_mean_stats = group_comparison(
    df.loc[df["cohort"] == "Young", "gfp_mean_0_500ms"],
    df.loc[df["cohort"] == "Old", "gfp_mean_0_500ms"],
    f"Mean GFP ({GFP_MEAN_TMIN * 1000:.0f}-{GFP_MEAN_TMAX * 1000:.0f} ms "
    f"post-stimulus, uV)")

# --- Plot: GFP time courses with SEM bands + significant clusters ---
fig_gfp, ax_gfp = plt.subplots(figsize=(12, 5))
for coh, gfp_mat in [("Young", gfp_young), ("Old", gfp_old)]:
    gfp_mean = gfp_mat.mean(axis=0)
    gfp_sem = gfp_mat.std(axis=0, ddof=1) / np.sqrt(len(gfp_mat))
    ax_gfp.plot(times_ms, gfp_mean, color=COHORT_COLORS[coh], lw=2.5, label=coh)
    ax_gfp.fill_between(times_ms, gfp_mean - gfp_sem, gfp_mean + gfp_sem,
                        color=COHORT_COLORS[coh], alpha=0.3)
# Mark significant cluster time spans (grey bars near the bottom of the axes)
for c_idx, (idx, p_val) in enumerate(sig_clusters):
    ax_gfp.axvspan(times_ms[idx[0]], times_ms[idx[-1]], color="grey",
                   alpha=0.25, ymin=0.02, ymax=0.12)
    ax_gfp.annotate(f"cluster p = {p_val:.3f}",
                    xy=(np.mean([times_ms[idx[0]], times_ms[idx[-1]]]), 0),
                    xycoords=("data", "axes fraction"), xytext=(0, 2),
                    textcoords="offset points", ha="center", va="bottom",
                    fontsize=10)
ax_gfp.axvline(0, color="black", ls="--", lw=1, alpha=0.6)  # stimulus onset
ax_gfp.set_xlabel("Time (ms)")
ax_gfp.set_ylabel("Global Field Power (µV)")
ax_gfp.set_title("Global Field Power by Age Cohort (shaded band: SEM; "
                 "grey bar: significant cluster)")
ax_gfp.legend(frameon=False)
ax_gfp.spines[["top", "right"]].set_visible(False)
fig_gfp.tight_layout()
fig_gfp_path = os.path.join(PLOTS_DIR, "signal_quality_gfp.svg")
fig_gfp.savefig(fig_gfp_path, dpi=300, bbox_inches="tight")
report(f"\nSaved GFP plot to: {fig_gfp_path}")
if not (hasattr(sys, "ps1") or "ipykernel" in sys.modules):
    plt.close(fig_gfp)

# =============================================================================
# 7. ANALYSIS 3: Event-related SNR (alternating-inversion) between groups
# =============================================================================
report("\n" + "=" * 60)
report("   ANALYSIS 3: EVENT-RELATED SNR, 0-500 MS (YOUNG VS. OLD)")
report("=" * 60)
report(f"Signal: RMS of the trial-average (ERP) across channels & time, "
       f"{SNR_TMIN * 1000:.0f}-{SNR_TMAX * 1000:.0f} ms post-stimulus.")
report("Noise: same RMS computed on the average of every-other-trial-"
       "inverted epochs (Schimmel, 1967), which cancels the ERP and retains "
       "the non-phase-locked residual.")
report("SNR = Signal RMS / Noise RMS (both are attenuated by 1/sqrt(N) "
       "alike, so the ratio estimates the single-trial evoked SNR).")

snr_stats = group_comparison(
    df.loc[df["cohort"] == "Young", "snr"],
    df.loc[df["cohort"] == "Old", "snr"],
    "SNR (Signal RMS / Noise RMS)")

snr_db_stats = group_comparison(
    df.loc[df["cohort"] == "Young", "snr_db"],
    df.loc[df["cohort"] == "Old", "snr_db"],
    "SNR (dB)")

# --- Plot: signal / noise amplitudes and SNR by group ---
fig_snr, axes_snr = plt.subplots(1, 3, figsize=(16, 5))
for ax, col, ylabel in [
    (axes_snr[0], "signal_rms_uv", "Signal Amplitude (µV RMS)"),
    (axes_snr[1], "noise_rms_uv", "Noise Amplitude (µV RMS)"),
    (axes_snr[2], "snr", "SNR (Signal / Noise)"),
]:
    for ci, coh in enumerate(COHORTS):
        vals = df.loc[df["cohort"] == coh, col].astype(float)
        ax.boxplot(vals, positions=[ci], widths=0.5, patch_artist=True,
                   showfliers=False,
                   boxprops=dict(facecolor=COHORT_COLORS[coh], alpha=0.85),
                   medianprops=dict(color="black", lw=2),
                   whiskerprops=dict(lw=1.5), capprops=dict(lw=1.5))
        rng = np.random.default_rng(SEED + ci)
        jitter = rng.uniform(-0.08, 0.08, size=len(vals))
        ax.scatter(np.full(len(vals), ci) + jitter, vals, color="black",
                   s=28, alpha=0.55, zorder=3)
    ax.set_xticks(range(len(COHORTS)), COHORTS)
    ax.set_xlabel("")
    ax.set_ylabel(ylabel)
    ax.spines[["top", "right"]].set_visible(False)
fig_snr.suptitle("Event-Related Signal Quality (0-500 ms) by Age Cohort", y=1.02)
fig_snr.tight_layout()
fig_snr_path = os.path.join(PLOTS_DIR, "signal_quality_snr.svg")
fig_snr.savefig(fig_snr_path, dpi=300, bbox_inches="tight")
report(f"\nSaved SNR plot to: {fig_snr_path}")
if not (hasattr(sys, "ps1") or "ipykernel" in sys.modules):
    plt.close(fig_snr)

# =============================================================================
# 8. Save per-subject metrics & full text report
# =============================================================================
metrics_csv_path = os.path.join(PLOTS_DIR, "signal_quality_subject_metrics.csv")
df.to_csv(metrics_csv_path, index=False)
report(f"\nSaved per-subject metrics to: {metrics_csv_path}")

report_path = os.path.join(PLOTS_DIR, "signal_quality_report.txt")
with open(report_path, "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines))
report(f"Saved full statistical report to: {report_path}")

report("\n=========================================================")
report("   SIGNAL-QUALITY SANITY CHECKS FINISHED   ")
report("=========================================================")