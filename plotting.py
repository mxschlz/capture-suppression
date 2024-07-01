import seaborn as sns
import matplotlib.pyplot as plt


def add_subject_lines(df, plot, groupby, **kwargs):
    df_mean = df.groupby(["subject_id", groupby]).mean(numeric_only=True).reset_index()
    bar_positions = []
    for patch in plot.patches:
        bar_positions.append(patch.get_x() + patch.get_width() / 2)

    for subject in df.subject_id.unique():
        subject_data = df_mean[df_mean['subject_id'] == subject]
        # Aligning subject data with bar positions
        x_positions = [bar_positions[i] for i, _ in enumerate(subject_data[groupby])]
        plt.plot(x_positions, subject_data[y], marker='o', linestyle='-', color='grey', alpha=0.5)


def barplot_with_single_subjects(df, x, y, subject_encoding="subject_id", groupby="SingletonPresent", **kwargs):
    df_mean = df.groupby([subject_encoding, groupby]).mean(numeric_only=True).reset_index()
    plot = sns.barplot(data=df, x=x, y=y, **kwargs)
    # get patch positions
    bar_positions = []
    for patch in plot.patches:
        bar_positions.append(patch.get_x() + patch.get_width() / 2)

    for subject in df.subject_id.unique():
        subject_data = df_mean[df_mean['subject_id'] == subject]
        # Aligning subject data with bar positions
        x_positions = [bar_positions[i] for i, _ in enumerate(subject_data[groupby])]
        plt.plot(x_positions, subject_data[y], marker='o', linestyle='-', color='grey', alpha=0.5)
