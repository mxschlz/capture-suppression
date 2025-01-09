import matplotlib.pyplot as plt


# Function to plot individual subject lines
def plot_individual_lines(ax, data, x_col="Priming", y_col="iscorrect"):
    # get positions of barplots in axis
    bar_positions = [patch.get_x() + patch.get_width() / 2 for patch in ax.patches]

    subjects = data["subject_id"].dropna().unique()
    for subject in subjects:
        subject_df = data[data["subject_id"] == subject]
        # Calculate mean for each subject (optional, adjust if needed)
        subject_mean = subject_df.groupby(x_col)[y_col].mean()
        # Aligning subject data with bar positions
        x_positions = [bar_positions[i] for i, _ in enumerate(subject_df[x_col].unique())]
        # plot the data
        ax.plot(x_positions, subject_mean.values, label=subject, linestyle='--', marker='', alpha=0.7)
    plt.legend()