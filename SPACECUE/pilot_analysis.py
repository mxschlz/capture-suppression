import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statannotations.Annotator import Annotator
from scipy.stats import ttest_ind, pearsonr
import numpy as np
import pingouin as pg
plt.ion()


df = pd.read_csv("G:\\Meine Ablage\\PhD\\data\\SPACECUE_behavioral_pilot\\concatenated\\all_subjects_merged_data.csv")
# --- Corrected & Robust Cohort Assignment ---

# 1. Calculate a single, consistent age for each subject (using the mean).
#    This prevents a subject from being in multiple cohorts if their age varies in the raw data.
subject_age_map = df.groupby('subject_id')['age'].mean()

# 2. Map this consistent age back to every row in the original DataFrame.
df['consistent_age'] = df['subject_id'].map(subject_age_map)

# 3. Create the 'cohort' column based on this new, consistent age.
df['cohort'] = np.where(df['consistent_age'] < 35, 'young', 'old')

# --- Verification ---
# Now, the count will be correct because each subject can only belong to one cohort.
print("Corrected participant counts per cohort:")
cohort_counts = df.groupby('cohort')['subject_id'].nunique()
print(cohort_counts)

# You can also verify the total number of unique participants, which should now match your expectation.
print(f"\nTotal unique participants: {df['subject_id'].nunique()}")
print(f"Sum of cohorts: {cohort_counts.sum()}")

df_mean = df.groupby(["subject_id", "CueInstruction", "CueDesignStrategy", "cohort"])[["select_target", "rt", "age"]].mean().reset_index()
df_mean["select_target"] = df_mean["select_target"].astype(float)
comparison_correction = "holm"

# --- Create the figure with two subplots (RT and Accuracy by CueInstruction) ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

plot_params_cue_instruction = [
    {"y_var": "rt", "title": "Reaction Time by Cue Instruction", "ylim": (2, 3), "ax_idx": 0},
    {"y_var": "select_target", "title": "Accuracy by Cue Instruction", "ylim": (0.4, 0.7), "ax_idx": 1}
]

comparison_pairs_cue = [
    ('cue_distractor_location', 'cue_neutral'),
    ('cue_neutral', 'cue_target_location'),
    ('cue_distractor_location', 'cue_target_location')
]
x_order_cue = ['cue_distractor_location', 'cue_neutral', 'cue_target_location']
x_labels_cue = ["distractor", "neutral", "target"]

for params in plot_params_cue_instruction:
    ax = axes[params["ax_idx"]]
    sns.barplot(data=df_mean, x="CueInstruction", y=params["y_var"], ax=ax,
                errorbar=("se", 1), order=x_order_cue, hue="cohort")
    ax.set_xticklabels(x_labels_cue)
    ax.set_ylim(params["ylim"])
    ax.set_title(params["title"])
    """
    annotator = Annotator(ax=ax, pairs=comparison_pairs_cue, data=df_mean,
                          x="CueInstruction", y=params["y_var"], order=x_order_cue)
    annotator.configure(test='t-test_paired', text_format='star', loc='inside', verbose=0,
                        comparisons_correction=comparison_correction)
    annotator.apply_and_annotate()
    """
plt.tight_layout()

# --- 2x3 Mixed ANOVA (Cohort x CueInstruction) ---
print("\n--- 2x3 Mixed ANOVA Accuracy ---")
# The dv is the dependent variable ('rt' or 'select_target')
# within is the within-subject factor
# between is the between-subject factor
# subject is the participant identifier
aov_acc = pg.mixed_anova(
    data=df_mean,
    dv='select_target',
    within='CueInstruction',
    between='cohort',
    subject='subject_id'
)
# Display the ANOVA results in a clean, rounded format
print(pg.print_table(aov_acc.round(3)))


# --- 2x3 Mixed ANOVA (Cohort x CueInstruction) ---
print("\n--- 2x3 Mixed ANOVA rt ---")
# The dv is the dependent variable ('rt' or 'select_target')
# within is the within-subject factor
# between is the between-subject factor
# subject is the participant identifier
aov_rt = pg.mixed_anova(
    data=df_mean,
    dv='rt',
    within='CueInstruction',
    between='cohort',
    subject='subject_id'
)
# Display the ANOVA results in a clean, rounded format
print(pg.print_table(aov_rt.round(3)))


# If the interaction is significant, these tests tell you where the differences are.
print("\n--- Post-Hoc Pairwise T-Tests for accuracy ---")
posthoc_acc = pg.pairwise_tests(
    data=df_mean,
    dv='select_target',
    within='CueInstruction',
    between='cohort',
    subject='subject_id',
    padjust='holm'  # Using the same Holm correction as in your plots
)
print(pg.print_table(posthoc_acc.round(3)))

# --- Follow-up Post-Hoc Tests for Reaction Time ---
# If the interaction is significant, these tests tell you where the differences are.
print("\n--- Post-Hoc Pairwise T-Tests for Reaction Time ---")
posthoc_rt = pg.pairwise_tests(
    data=df_mean,
    dv='rt',
    within='CueInstruction',
    between='cohort',
    subject='subject_id',
    padjust='holm'  # Using the same Holm correction as in your plots
)
print(pg.print_table(posthoc_rt.round(3)))

# --- Plot: Accuracy by Cue Instruction and Design Strategy ---
plt.figure(figsize=(8, 6))
ax_acc_strategy = plt.gca()
sns.barplot(data=df_mean, x="CueInstruction", y="select_target", hue="CueDesignStrategy",
            ax=ax_acc_strategy, errorbar=("se", 1), order=x_order_cue)
ax_acc_strategy.set_xticklabels(x_labels_cue, rotation=45, ha='right')
plt.title("Accuracy by Cue Instruction and Design Strategy")

plt.tight_layout()

# --- 2x3 Mixed ANOVA (Cohort x CueInstruction) on Reaction Time ---
print("\n--- 2x3 Mixed ANOVA: Reaction Time (rt) ---")
# The dv is the dependent variable ('rt' or 'select_target')
# within is the within-subject factor
# between is the between-subject factor
# subject is the participant identifier
aov = pg.mixed_anova(
    data=df_mean,
    dv='select_target',
    within='CueInstruction',
    between='CueDesignStrategy',
    subject='subject_id'
)
# Display the ANOVA results in a clean, rounded format
print(pg.print_table(aov.round(3)))


# --- Follow-up Post-Hoc Tests for Reaction Time ---
# If the interaction is significant, these tests tell you where the differences are.
print("\n--- Post-Hoc Pairwise T-Tests for Reaction Time ---")
posthoc = pg.pairwise_tests(
    data=df_mean,
    dv='select_target',
    within='CueInstruction',
    between='CueDesignStrategy',
    subject='subject_id',
    padjust='holm'  # Using the same Holm correction as in your plots
)
print(pg.print_table(posthoc.round(3)))

# --- Follow-up Post-Hoc Tests for Reaction Time ---
# If the interaction is significant, these tests tell you where the differences are.
print("\n--- Post-Hoc Pairwise T-Tests for Reaction Time ---")
posthoc = pg.pairwise_tests(
    data=df_mean,
    dv='rt',
    within='CueInstruction',
    between='CueDesignStrategy',
    subject='subject_id',
    padjust='holm'  # Using the same Holm correction as in your plots
)
print(pg.print_table(posthoc.round(3)))


# --- Plots for "Priming" ---
# Prepare data grouped by Priming
if "Priming" in df.columns:
    df_priming_mean = df.groupby(["subject_id", "Priming"])[["rt", "select_target"]].mean().reset_index()
    priming_categories = sorted(df_priming_mean["Priming"].unique()) # Sort for consistent order

    if len(priming_categories) >= 2:
        # Define pairs for priming comparison. Adapt as needed.
        # Example: compare all pairs if 3 categories, or specific pairs.
        pairs_priming = []
        if len(priming_categories) == 2:
            pairs_priming.append((priming_categories[0], priming_categories[1]))
        elif len(priming_categories) > 2:
            # Example: compare first vs second, second vs third, first vs third
            pairs_priming.append((priming_categories[0], priming_categories[1]))
            pairs_priming.append((priming_categories[1], priming_categories[2])) # Assuming at least 3
            if len(priming_categories) >=3:
                 pairs_priming.append((priming_categories[0], priming_categories[2]))


        # Plot: RT by Priming
        plt.figure(figsize=(6, 5))
        ax_rt_priming = plt.gca()
        sns.barplot(data=df_priming_mean, x="Priming", y="rt", ax=ax_rt_priming,
                    errorbar=("se", 1), order=priming_categories)
        plt.title("RT by Priming")

        if pairs_priming:
            annotator_rt_priming = Annotator(ax=ax_rt_priming, pairs=pairs_priming, data=df_priming_mean,
                                             x="Priming", y="rt", order=priming_categories)
            annotator_rt_priming.configure(test='t-test_paired', text_format='star', loc='inside', verbose=0,
                                           comparisons_correction=comparison_correction)
            annotator_rt_priming.apply_and_annotate()
        plt.tight_layout()


        # Plot: Target Selection by Priming
        plt.figure(figsize=(6, 5))
        ax_sel_priming = plt.gca()
        sns.barplot(data=df_priming_mean, x="Priming", y="select_target", ax=ax_sel_priming,
                    errorbar=("se", 1), order=priming_categories)
        plt.title("Target Selection by Priming")

        if pairs_priming:
            annotator_sel_priming = Annotator(ax=ax_sel_priming, pairs=pairs_priming, data=df_priming_mean,
                                              x="Priming", y="select_target", order=priming_categories)
            annotator_sel_priming.configure(test='t-test_paired', text_format='star', loc='inside', verbose=0,
                                            comparisons_correction=comparison_correction)
            annotator_sel_priming.apply_and_annotate()
        plt.tight_layout()
    else:
        print("Not enough 'Priming' categories to compare.")
else:
    print("'Priming' column not found in DataFrame. Skipping priming plots.")

sns.barplot(data=df_mean, x="CueDesignStrategy", y="select_target")
ttest_ind(df_mean.query("CueDesignStrategy=='Trial'")["select_target"], df_mean.query("CueDesignStrategy=='Block'")["select_target"])

sns.lmplot(data=df_mean, x="age", y="rt")

pearsonr(df_mean["age"], df_mean["rt"])


# --- Analysis of Cue-Stimulus Delay (Corrected for Jitter) ---
# 1. CORRECT THE REACTION TIME
# This is the crucial step. We subtract the delay from the measured RT
# to get the true response time from stimulus onset.
df['rt_corrected'] = df['rt'] - df['cue_stim_delay_jitter']

# Sanity Check: Look for negative corrected RTs.
# These would indicate responses made *before* the stimulus appeared.
# A small number might be due to fast guesses, but a large number could
# indicate a problem with the timing data.
negative_rts = df[df['rt_corrected'] < 0]
if not negative_rts.empty:
    print(f"\nWARNING: Found {len(negative_rts)} trials ({len(negative_rts)/len(df)*100:.2f}%) with negative corrected RTs.")
    print("These are responses that occurred before the stimulus appeared.")
    # Optional: You might want to filter these out for the analysis
    # df = df[df['rt_corrected'] >= 0].copy()
print("-" * 40)

# 2. Bin the delay durations into three equal-sized groups (tertiles).
df['delay_bin'] = pd.qcut(df['cue_stim_delay_jitter'],
                          q=3,
                          labels=['short', 'medium', 'long'])

print("\nValue counts for each delay bin:")
print(df['delay_bin'].value_counts())
print("-" * 40)

# 3. Aggregate the data using the NEW corrected RT.
# We now calculate the mean of 'rt_corrected'.
df_delay_mean = df.groupby(
    ["subject_id", "CueInstruction", "delay_bin", "cohort"]
)[["select_target", "rt_corrected"]].mean().reset_index().dropna() # Changed 'rt' to 'rt_corrected'
df_delay_mean["select_target"] = df_delay_mean["select_target"].astype(float)

# 4. Plot the results.
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Performance by Cue Instruction and Stimulus Delay (Corrected RT)', fontsize=16)

x_order_cue = ['cue_distractor_location', 'cue_neutral', 'cue_target_location']
x_labels_cue = ["Distractor", "Neutral", "Target"]
hue_order_delay = ['short', 'medium', 'long']

# --- Plot 1: Corrected Reaction Time ---
sns.barplot(data=df_delay_mean, x="CueInstruction", y="rt_corrected", hue="delay_bin", # Changed y to 'rt_corrected'
            ax=axes[0], errorbar=("se", 1), order=x_order_cue, hue_order=hue_order_delay)
axes[0].set_title("Corrected Reaction Time")
axes[0].set_xticklabels(x_labels_cue)
axes[0].set_xlabel("Cue Instruction")
axes[0].set_ylabel("Corrected Reaction Time (s)") # Updated label
axes[0].legend(title='Delay Bin')

# --- Plot 2: Accuracy (unchanged) ---
sns.barplot(data=df_delay_mean, x="CueInstruction", y="select_target", hue="delay_bin",
            ax=axes[1], errorbar=("se", 1), order=x_order_cue, hue_order=hue_order_delay)
axes[1].set_title("Accuracy")
axes[1].set_xticklabels(x_labels_cue)
axes[1].set_xlabel("Cue Instruction")
axes[1].set_ylabel("Proportion Target Selected")
axes[1].legend(title='Delay Bin')

# --- Add statistical annotations to the final plot ---

# 1. Define the pairs for comparison.
#    We will compare across CueInstruction types for each fixed delay bin.
comparison_pairs = []
for delay_level in hue_order_delay:
    # For each delay level, create all pairwise comparisons for the three cue types
    comparison_pairs.extend([
        (('cue_distractor_location', delay_level), ('cue_neutral', delay_level)),
        (('cue_neutral', delay_level), ('cue_target_location', delay_level)),
        (('cue_distractor_location', delay_level), ('cue_target_location', delay_level)),
    ])

# 2. Annotate the Corrected Reaction Time plot (axes[0])
annotator_rt = Annotator(
    ax=axes[0],
    pairs=comparison_pairs,
    data=df_delay_mean,
    x="CueInstruction",
    y="rt_corrected",
    hue="delay_bin",
    order=x_order_cue,
    hue_order=hue_order_delay
)
annotator_rt.configure(
    test='t-test_paired',
    text_format='star',
    loc='inside', # Changed to 'outside' to prevent overlap
    verbose=0,
    comparisons_correction="holm"
)
annotator_rt.apply_and_annotate()

# 3. Annotate the Accuracy plot (axes[1])
annotator_acc = Annotator(
    ax=axes[1],
    pairs=comparison_pairs,
    data=df_delay_mean,
    x="CueInstruction",
    y="select_target",
    hue="delay_bin",
    order=x_order_cue,
    hue_order=hue_order_delay
)
annotator_acc.configure(
    test='t-test_paired',
    text_format='star',
    loc='inside', # Changed to 'outside' to prevent overlap
    verbose=0,
    comparisons_correction="holm"
)
annotator_acc.apply_and_annotate()

# Adjust layout again to make sure annotations fit properly
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# 5. Statistical Analysis using Linear Mixed-Effects Models (LMM)
#    We use LMM because it can handle complex designs with multiple
#    within- and between-subject factors simultaneously, which AnovaRM cannot.
import statsmodels.formula.api as smf
print("\n--- 3x3x2 Mixed Linear Model (LMM): Corrected Reaction Time ---")
# The formula specifies the model:
# - 'rt_corrected' is the dependent variable.
# - 'C(variable)' treats the variable as categorical.
# - '*' includes the main effects and all interactions between the factors.
# - 'groups' specifies the random effect (i.e., the subject identifier).
lmm_rt = smf.mixedlm(
    "select_target ~ C(CueInstruction) * C(delay_bin)",
    data=df_delay_mean,
    groups="subject_id"
).fit()

print(lmm_rt.summary())


print("\n--- 3x3x2 Mixed Linear Model (LMM): Accuracy ---")
lmm_acc = smf.mixedlm(
    "select_target ~ C(CueInstruction) * C(delay_bin) * C(cohort)",
    data=df_delay_mean,
    groups="subject_id"
).fit()

print(lmm_acc.summary())
