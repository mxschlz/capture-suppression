import SPACEPRIME
import seaborn as sns
import matplotlib.pyplot as plt
import pingouin as pg
plt.ion()


df = SPACEPRIME.load_concatenated_csv("target_towardness_all_variables.csv")

# Make sure the dataframe is sorted by subject, block and trial to correctly identify the 'previous' trial
# I'm assuming the columns are named 'subject', 'block', and 'trial'.
# If they have different names, please adjust them in the sort_values() call.
df = df.sort_values(['subject_id', 'block', 'trial_nr']).reset_index(drop=True)

# Create shifted columns for the non-singleton distractors from the previous trial (n-1)
df['prev_ns1_loc'] = df.groupby(['subject_id', 'block'])['Non-Singleton1Loc'].shift(1)
df['prev_ns1_digit'] = df.groupby(['subject_id', 'block'])['Non-Singleton1Digit'].shift(1)
df['prev_ns2_loc'] = df.groupby(['subject_id', 'block'])['Non-Singleton2Loc'].shift(1)
df['prev_ns2_digit'] = df.groupby(['subject_id', 'block'])['Non-Singleton2Digit'].shift(1)

# Define the conditions for the new priming category
# Condition 1: Current target matches the first non-singleton from the previous trial
cond1 = (df['TargetLoc'] == df['prev_ns1_loc']) & (df['TargetDigit'] == df['prev_ns1_digit'])

# Condition 2: Current target matches the second non-singleton from the previous trial
cond2 = (df['TargetLoc'] == df['prev_ns2_loc']) & (df['TargetDigit'] == df['prev_ns2_digit'])

# Apply the conditions to update the 'Priming' column
df.loc[cond1 | cond2, 'Priming'] = 2

# You can now drop the temporary 'prev_' columns if you no longer need them
df = df.drop(columns=['prev_ns1_loc', 'prev_ns1_digit', 'prev_ns2_loc', 'prev_ns2_digit'])

# Count the number of trials for the new priming category
priming_2_count = len(df[df['Priming'] == 2])
print(f"\nNumber of trials where Priming is 2: {priming_2_count}")

# You can also see this in the context of all priming types
print("\nValue counts for 'Priming' column before remapping:")
print(df['Priming'].value_counts())

# Remap the numerical Priming categories to descriptive strings
priming_map = {
    2: "Control",
    -1: "Negative",
    0: "No",
    1: "Positive"
}
df['Priming'] = df['Priming'].map(priming_map)

print("\nValue counts for 'Priming' column after remapping:")
print(df['Priming'].value_counts())

# Define the dependent variables you want to plot
dependent_vars = ['target_towardness']

# Define the pairs of conditions you want to compare statistically
# For example, comparing the neutral condition (0) to all others,
# plus comparing positive (1) and negative (-1) priming.
comparison_pairs = [
    ("Control", "Negative"),
    ("Control", "No"),
    ("Control", "Positive"),
    ("Negative", "No"),
    ("Negative", "Positive"),
    ("No", "Positive")
]

# Define the desired order for the plot
plot_order = ["Control", "Negative", "No", "Positive"]

# Loop through each dependent variable and create a plot
for var in dependent_vars:
    # Create a figure and axes for a single plot
    fig, ax = plt.subplots(figsize=(8, 6))

    # Create the bar plot with error bars (95% CI by default across subjects)
    sns.barplot(
        data=df,
        x='Priming',
        y=var,
        ax=ax,
        order=plot_order  # Set the order of the bars
    )

    # --- Perform Pairwise T-tests with Multiple Comparison Correction ---
    # First, calculate the mean for each subject and condition
    subject_means = df.groupby(['subject_id', 'Priming'])[var].mean().reset_index()

    # Now, run all pairwise tests with Bonferroni correction
    pairwise_results = pg.pairwise_tests(
        data=subject_means,
        dv=var,
        within='Priming',
        subject='subject_id',
        padjust='bonf',
        effsize="cohen"
    ).round(3) # Round results for cleaner display

    # --- Annotation using the corrected results ---
    # Get y-axis limits to position annotations
    y_min, y_top = ax.get_ylim()

    # Find the max height of the bars to place annotations above them
    bar_heights = [p.get_height() for p in ax.patches if p.get_height() > 0]
    plot_max = max(bar_heights) if bar_heights else y_top

    # Define how much space to put between annotation lines
    offset_increment = (y_top - y_min) * 0.08
    current_offset = plot_max + offset_increment * 0.5

    # Get the order of categories on the x-axis
    category_order = [tick.get_text() for tick in ax.get_xticklabels()]

    for group1_val, group2_val in comparison_pairs:
        # Find the result for the current pair in the pairwise_results dataframe
        res_row = pairwise_results[
            ((pairwise_results['A'] == group1_val) & (pairwise_results['B'] == group2_val)) |
            ((pairwise_results['A'] == group2_val) & (pairwise_results['B'] == group1_val))
        ]

        if res_row.empty:
            continue

        # Get the corrected p-value and effect size
        p_corr = res_row['p-corr'].iloc[0]
        cohen_d = res_row['cohen'].iloc[0]

        # Format the text to display
        p_str = f"p_corr={p_corr:.3f}" if p_corr >= 0.001 else "p_corr<.001"
        d_str = f"d={cohen_d:.2f}"
        annotation_text = f"{p_str}\n{d_str}"

        # Find x-coordinates for the bars by finding the index in the category order
        x1 = category_order.index(str(group1_val))
        x2 = category_order.index(str(group2_val))

        # Draw annotation lines and text
        ax.plot([x1, x1, x2, x2], [current_offset, current_offset + offset_increment*0.2, current_offset + offset_increment*0.2, current_offset], lw=1.5, c='k')
        ax.text((x1 + x2) * 0.5, current_offset + offset_increment*0.25, annotation_text, ha='center', va='bottom', color='k')

        # Increase offset for the next annotation line
        current_offset += offset_increment * 1.5

    # Adjust y-limit to make space for annotations
    ax.set_ylim(y_min, current_offset)

    # Set plot titles and labels for clarity
    ax.set_xlabel("Priming Condition")
    ax.set_ylabel(f"Mean {var}")
    plt.tight_layout()
    sns.despine()
