import matplotlib.pyplot as plt
import numpy as np
import mne


# Function to plot individual subject lines
def plot_individual_lines(ax, data, x_col="Priming", y_col="select_target"):
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


def difference_topos(epochs, montage):
    ch_pos = montage.get_positions()["ch_pos"]

    # --- Difference Wave Calculation ---
    diff_waves_target = {}
    diff_waves_distractor = {}
    # get all channels from epochs
    all_chs = epochs.ch_names
    # get all conditions
    all_conds = list(epochs.event_id.keys())
    for ch_name in all_chs:
        if not "z" in ch_name:
            # Determine hemisphere of current channel
            try:
                channel_position = ch_pos[ch_name]
            except:
                continue
            is_left = channel_position[0] < 0  # x coordinate negative = left
            print(f"channel {ch_name} is left {is_left}")
            # --- TARGET ---
            if is_left:
                contra_target_epochs = epochs[[x for x in all_conds if "Target-3-Singleton-2" in x]]
                ipsi_target_epochs = epochs[[x for x in all_conds if "Target-1-Singleton-2" in x]]
            else:  # Right hemisphere
                contra_target_epochs = epochs[[x for x in all_conds if "Target-1-Singleton-2" in x]]
                ipsi_target_epochs = epochs[[x for x in all_conds if "Target-3-Singleton-2" in x]]

            mne.epochs.equalize_epoch_counts([contra_target_epochs, ipsi_target_epochs], method="random")

            contra_target_data = contra_target_epochs.get_data(picks=ch_name).mean(axis=0)  # Average trials
            ipsi_target_data = ipsi_target_epochs.get_data(picks=ch_name).mean(axis=0)
            diff_waves_target[ch_name] = (contra_target_data - ipsi_target_data)[0]

            # --- DISTRACTOR ---
            if is_left:
                contra_distractor_epochs = epochs[[x for x in all_conds if "Target-2-Singleton-3" in x]]
                ipsi_distractor_epochs = epochs[[x for x in all_conds if "Target-2-Singleton-1" in x]]
            else:
                contra_distractor_epochs = epochs[[x for x in all_conds if "Target-2-Singleton-1" in x]]
                ipsi_distractor_epochs = epochs[[x for x in all_conds if "Target-2-Singleton-3" in x]]

            mne.epochs.equalize_epoch_counts([contra_distractor_epochs, ipsi_distractor_epochs], method="random")

            contra_distractor_data = contra_distractor_epochs.get_data(picks=ch_name).mean(axis=0)
            ipsi_distractor_data = ipsi_distractor_epochs.get_data(picks=ch_name).mean(axis=0)
            diff_waves_distractor[ch_name] = (contra_distractor_data - ipsi_distractor_data)[0]
        else:
            diff_waves_distractor[ch_name] = np.zeros((len(epochs.times)))
            diff_waves_target[ch_name] = np.zeros((len(epochs.times)))

    return diff_waves_target, diff_waves_distractor
