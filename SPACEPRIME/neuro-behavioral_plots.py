import seaborn as sns
import SPACEPRIME
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
sns.set_theme(context="talk", style="ticks")
plt.ion()


def plot_latency_amplitude_by_towardness(df, component_name, ax, subject_col='subject_id',
                                         latency_col='st_latency_50', amplitude_col='st_mean_amp_50',
                                         add_legend=True, center_data=False):
    """
    Generates a within-subject plot of ERP amplitude vs. latency on a given axis.

    This plot shows:
    1. Individual subject means for high/low towardness (dots).
    2. Lines connecting the within-subject data points.
    3. Summary crosses representing the group mean +/- standard error.

    Args:
        df (pd.DataFrame): The input dataframe with single-trial data.
        component_name (str): The name of the ERP component for plot titles.
        ax (matplotlib.axes.Axes): The subplot axis to draw the plot on.
        subject_col (str): The name of the column identifying subjects.
        latency_col (str): The name of the latency data column.
        amplitude_col (str): The name of the amplitude data column.
        add_legend (bool): If True, a legend is added to the plot.
        center_data (bool): If True, performs within-subject centering to reduce spread.
    """
    if df is None or df.empty:
        ax.text(0.5, 0.5, f'No data for {component_name}', ha='center', va='center')
        ax.set_title(component_name)
        return

    # --- 1. Data Preparation: Within-subject percentile split (25th/75th) and aggregation ---
    subject_means = []
    for subject_id, subject_df in df.groupby(subject_col):
        if subject_df.empty:
            continue

        q25 = subject_df['target_towardness'].quantile(0.25)
        q75 = subject_df['target_towardness'].quantile(0.75)
        low_df = subject_df[subject_df['target_towardness'] < q25]
        high_df = subject_df[subject_df['target_towardness'] > q75]

        # Only include subjects that have data for both conditions to ensure valid within-subject contrasts
        if low_df.empty or high_df.empty:
            continue

        subject_means.append({
            subject_col: subject_id,
            'towardness_level': 'Low',
            'latency': low_df[latency_col].mean(),
            'amplitude': low_df[amplitude_col].mean()
        })
        subject_means.append({
            subject_col: subject_id,
            'towardness_level': 'High',
            'latency': high_df[latency_col].mean(),
            'amplitude': high_df[amplitude_col].mean()
        })

    agg_df = pd.DataFrame(subject_means)
    if agg_df.empty:
        ax.text(0.5, 0.5, f'No data for {component_name}', ha='center', va='center')
        ax.set_title(component_name)
        return

    # --- 2. Optional Within-Subject Centering ---
    plot_df = agg_df.copy()
    if center_data:
        # Use transform to get each subject's grand mean and subtract it
        subject_grand_means = plot_df.groupby(subject_col)[['latency', 'amplitude']].transform('mean')
        plot_df['latency'] -= subject_grand_means['latency']
        plot_df['amplitude'] -= subject_grand_means['amplitude']
        x_label, y_label = 'Latency (Centered, s)', 'Amplitude (Centered, µV)'
    else:
        x_label, y_label = 'Latency (s)', 'Amplitude (µV)'

    # High-contrast color scheme (Okabe-Ito: Blue vs. Vermilion)
    colors = {'Low': '#0072B2', 'High': '#D55E00'}

    sns.scatterplot(
        data=plot_df, x='latency', y='amplitude', hue='towardness_level',
        palette=colors, s=80, alpha=0.4,
        ax=ax, legend=False
    )

    # --- 4. Summary Crosses (Mean +/- SEM) ---
    summary_stats = plot_df.groupby('towardness_level').agg(
        mean_latency=('latency', 'mean'),
        sem_latency=('latency', 'sem'),
        mean_amplitude=('amplitude', 'mean'),
        sem_amplitude=('amplitude', 'sem')
    ).reset_index()

    for _, row in summary_stats.iterrows():
        color = colors[row['towardness_level']]
        ax.errorbar(
            x=row['mean_latency'], y=row['mean_amplitude'],
            xerr=row['sem_latency'], yerr=row['sem_amplitude'],
            fmt='X', markersize=0, markeredgecolor='white', markerfacecolor=color,
            color=color, elinewidth=3, capsize=0, zorder=20
        )

    # --- 5. Final Touches ---
    ax.set_title(f'{component_name} Component', fontsize=18)
    ax.set_xlabel(x_label, fontsize=14)
    ax.set_ylabel(y_label, fontsize=14)

    if add_legend:
        legend_handles = [
            Line2D([0], [0], marker='o', color='w', label='Low', markerfacecolor=colors['Low'], markersize=10),
            Line2D([0], [0], marker='o', color='w', label='High', markerfacecolor=colors['High'], markersize=10)
        ]
        ax.legend(handles=legend_handles, labels=['Low Towardness', 'High Towardness'],
                  title='Condition', title_fontsize='14', fontsize='12', loc='best')


# Load data
n2ac_df = SPACEPRIME.load_concatenated_csv("spaceprime_n2ac_erp_behavioral_lmm_long_data_between-within.csv")
pd_df = SPACEPRIME.load_concatenated_csv("spaceprime_pd_erp_behavioral_lmm_long_data_between-within.csv")

# Create a figure with two subplots
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True, sharex=True) # sharey=False because N2ac is inverted

# --- Generate the plot for the N2ac component on the first axis ---
plot_latency_amplitude_by_towardness(
    df=n2ac_df.dropna(),
    component_name="N2ac",
    ax=axes[0],
    subject_col='subject_id',
    add_legend=True,
    center_data=True  # <-- Center the data
)

# --- Generate the plot for the Pd component on the second axis ---
plot_latency_amplitude_by_towardness(
    df=pd_df.dropna(),
    component_name="Pd",
    ax=axes[1],
    subject_col='subject_id',
    add_legend=False,
    center_data=True  # <-- Center the data
)

# Clean up the layout and display the plot
sns.despine(fig=fig)
plt.tight_layout() # Adjust layout to make room for suptitle

plt.savefig("neuro-behavioral_plots.svg")


def plot_target_towardness_schematic():
    """
    Generates a schematic layout of the digital response box and cursor trajectories
    illustrating the concepts of high and low target towardness, using real cursor
    trajectory data when available, and falling back to simulated paths otherwise.
    """
    import numpy as np
    import glob
    import os

    # Define grid locations (same as in the experiment)
    numpad_locations_dva = {
        7: (-0.6, -0.6), 8: (0, -0.6), 9: (0.6, -0.6),
        4: (-0.6, 0), 5: (0, 0), 6: (0.6, 0),
        1: (-0.6, 0.6), 2: (0, 0.6), 3: (0.6, 0.6),
    }

    target_digit = 2
    distractor_digit = 4

    high_x, high_y = None, None
    low_x, low_y = None, None

    try:
        # Load the concatenated variables to select trials
        all_vars_df = SPACEPRIME.load_concatenated_csv("target_towardness_all_variables.csv")
        
        # Filter for correct trials where Target is 2 and Distractor is 4
        match_df = all_vars_df[(all_vars_df['TargetDigit'] == 2) & 
                              (all_vars_df['SingletonDigit'] == 4) & 
                              (all_vars_df['response'] == 2)]
        
        if not match_df.empty:
            # Sort by towardness to pick the best trials
            match_sorted = match_df.sort_values(by='target_towardness')
            
            # Low towardness: pick the candidate with the lowest towardness (captured by distractor)
            low_info = match_sorted.iloc[0]
            # High towardness: pick the candidate with the highest towardness (direct path)
            high_info = match_sorted.iloc[-1]
            
            # Helper function to load raw mouse data for a specific trial
            def load_real_trajectory(sub, blk, trn):
                data_path = SPACEPRIME.get_data_path()
                sub_folder = f"sub-{int(sub)}"
                pattern = os.path.join(data_path, "sourcedata", "raw", sub_folder, "beh", f"{sub_folder}*mouse_data.csv")
                files = glob.glob(pattern)
                if not files:
                    return None
                
                raw_df = pd.read_csv(files[0])
                raw_df['subject_id'] = int(sub)
                
                # Reconstruct block numbers based on trial_nr resets
                raw_df['block'] = ((raw_df['trial_nr'] == 0) & (raw_df['trial_nr'].shift(1) != 0)).cumsum() - 1
                
                # Query specific trial up to the click (time <= 0)
                trial_df = raw_df[(raw_df['block'] == int(blk)) & 
                                  (raw_df['trial_nr'] == int(trn)) & 
                                  (raw_df['time'] <= 0)].copy()
                if trial_df.empty:
                    return None
                
                # Resample trajectory to 50 Hz
                from utils import resample_trial_trajectory
                resampled = resample_trial_trajectory(trial_df, target_hz=50)
                return resampled['x'].values, resampled['y'].values

            # Try to load the trajectories
            real_high = load_real_trajectory(high_info['subject_id'], high_info['block'], high_info['trial_nr'])
            if real_high is not None:
                high_x, high_y = real_high
                print(f"Loaded real high towardness trajectory from sub-{int(high_info['subject_id'])}, block-{int(high_info['block'])}, trial-{int(high_info['trial_nr'])}")
            
            real_low = load_real_trajectory(low_info['subject_id'], low_info['block'], low_info['trial_nr'])
            if real_low is not None:
                low_x, low_y = real_low
                print(f"Loaded real low towardness trajectory from sub-{int(low_info['subject_id'])}, block-{int(low_info['block'])}, trial-{int(low_info['trial_nr'])}")
    except Exception as e:
        print(f"Warning: Failed to load real trajectories: {e}. Using simulated paths as fallback.")

    # Create figure
    fig_schematic, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6.5))

    def draw_schematic(ax, title, towardness_type, x_traj, y_traj):
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(-0.8, 0.8)
        ax.set_ylim(-0.8, 0.8)
        
        # Draw response box buttons with premium styling
        for digit, (x, y) in numpad_locations_dva.items():
            if digit == target_digit:
                circle = plt.Circle((x, y), radius=0.09, facecolor='#E8F5E9', edgecolor='#2E7D32', linewidth=2, zorder=2)
                ax.add_patch(circle)
                ax.text(x, y + 0.02, str(digit), color='#2E7D32', ha='center', va='center', fontweight='bold', fontsize=14, zorder=3)
                ax.text(x, y - 0.035, "Target", color='#2E7D32', ha='center', va='center', fontweight='bold', fontsize=8, zorder=3)
            elif digit == distractor_digit:
                circle = plt.Circle((x, y), radius=0.09, facecolor='#FFEBEE', edgecolor='#C62828', linewidth=2, zorder=2)
                ax.add_patch(circle)
                ax.text(x, y + 0.02, str(digit), color='#C62828', ha='center', va='center', fontweight='bold', fontsize=14, zorder=3)
                ax.text(x, y - 0.035, "Dist.", color='#C62828', ha='center', va='center', fontweight='bold', fontsize=8, zorder=3)
            elif digit == 5:
                circle = plt.Circle((x, y), radius=0.09, facecolor='#ECEFF1', edgecolor='#37474F', linewidth=2, zorder=2)
                ax.add_patch(circle)
                ax.text(x, y + 0.02, "5", color='#37474F', ha='center', va='center', fontweight='bold', fontsize=14, zorder=3)
                ax.text(x, y - 0.035, "Start", color='#37474F', ha='center', va='center', fontweight='bold', fontsize=8, zorder=3)
            else:
                circle = plt.Circle((x, y), radius=0.09, facecolor='#FAFAFA', edgecolor='#ECEFF1', linewidth=1, zorder=2)
                ax.add_patch(circle)
                ax.text(x, y, str(digit), color='#90A4AE', ha='center', va='center', fontweight='bold', fontsize=14, zorder=3)

        # Define condition-specific colors (High: Vermilion, Low: Blue)
        cond_color = '#D55E00' if towardness_type == 'high' else '#0072B2'

        # Plot cursor trajectory
        ax.plot(x_traj, y_traj, color=cond_color, linewidth=2.5, zorder=4, label='Cursor Path')
        ax.plot(x_traj[0], y_traj[0], 'o', color='#4CAF50', markersize=9, zorder=5, label='Start Point')
        ax.plot(x_traj[-1], y_traj[-1], 'X', color='#212121', markersize=9, zorder=5, label='Click Point')
        
        # Calculate vectors
        avg_x = np.mean(x_traj)
        avg_y = np.mean(y_traj)
        
        # Ideal target vector
        ax.arrow(0, 0, 0, 0.6, color='#2E7D32', width=0.012, head_width=0.04, length_includes_head=True, zorder=6, label='Ideal Target Vector')
        
        # Ideal distractor vector
        ax.arrow(0, 0, -0.6, 0, color='#D32F2F', width=0.012, head_width=0.04, length_includes_head=True, zorder=6, label='Ideal Distractor Vector')
        
        # Average trajectory vector
        ax.arrow(0, 0, avg_x, avg_y, color='#7B1FA2', width=0.012, head_width=0.04, length_includes_head=True, zorder=7, label='Avg. Trajectory Vector')
        
        # Target towardness (projection) vector
        proj_y = avg_y
        ax.arrow(0, 0, 0, proj_y, color=cond_color, width=0.02, head_width=0.06, length_includes_head=True, zorder=8, label='Target Towardness (Proj.)')
        
        # Dashed projection helper line
        ax.plot([avg_x, 0], [avg_y, proj_y], color='#78909C', linestyle='--', linewidth=1.5, zorder=9)
        
        # Remove ticks and add a subtle border
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color('#CFD8DC')
            spine.set_linewidth(1.5)
            
        ax.set_title(title, fontsize=16, fontweight='bold', pad=15)

    # Use real trajectory if loaded, otherwise fall back to simulated coordinates
    if high_x is not None and high_y is not None:
        draw_schematic(ax1, "High Target Towardness\n(Direct Real Path)", 'high', high_x, high_y)
    else:
        t = np.linspace(0, 1, 100)
        # Deflected path to the right (opposite the distractor) to visually separate the average and projection vectors
        x_sim = 0.8 * t * (1 - t)
        y_sim = 0.6 * t
        draw_schematic(ax1, "High Target Towardness\n(Direct Path)", 'high', x_sim, y_sim)

    if low_x is not None and low_y is not None:
        draw_schematic(ax2, "Low Target Towardness\n(Deflected Real Path)", 'low', low_x, low_y)
    else:
        t = np.linspace(0, 1, 100)
        # Extremely deflected path bending close to the distractor at (-0.6, 0)
        x_sim = -2.4 * t * (1 - t)
        y_sim = -0.4 * t * (1 - t) + 0.6 * t**2
        draw_schematic(ax2, "Low Target Towardness\n(Deflected Path)", 'low', x_sim, y_sim)

    # Add legend to the figure
    handles, labels = ax1.get_legend_handles_labels()
    # Filter unique labels for legend
    by_label = dict(zip(labels, handles))
    fig_schematic.legend(by_label.values(), by_label.keys(), loc='lower center', ncol=4, fontsize=11, frameon=True, bbox_to_anchor=(0.5, -0.08))

    plt.tight_layout()
    plt.savefig("target_towardness_schematic.svg", bbox_inches='tight')
    print("Successfully saved target_towardness_schematic.svg")


plot_target_towardness_schematic()

