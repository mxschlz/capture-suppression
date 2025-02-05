import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import ttest_ind
import mne


def cohen_d(x, y):
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    return (np.mean(x) - np.mean(y)) / np.sqrt(((nx - 1) * np.std(x, ddof=1) ** 2 + (ny - 1) * np.std(y, ddof=1) ** 2) / dof)


def cohen_d_av(mean1, mean2, sd1, sd2):
    """
    Calculates Cohen's d using the average standard deviation (d_av).

    Args:
        mean_diff: Mean difference between conditions.
        sd1: Standard deviation of condition 1.
        sd2: Standard deviation of condition 2.

    Returns:
        Cohen's d_av.
    """
    pooled_sd = np.sqrt((sd1**2 + sd2**2) / 2)
    mean_diff = mean1 - mean2
    d_av = mean_diff / pooled_sd
    return d_av


def cohen_d_rm(mean_diff, sd1, sd2, correlation):
    """
    Calculates Cohen's d for repeated measures (d_rm).

    Args:
        mean_diff: Mean difference between conditions.
        sd1: Standard deviation of condition 1.
        sd2: Standard deviation of condition 2.
        correlation: Correlation between measures in the two conditions.

    Returns:
        Cohen's d_rm.
    """
    pooled_sd = np.sqrt((sd1**2 + sd2**2 - 2 * correlation * sd1 * sd2) / 2*(1-correlation))
    d_rm = (mean_diff / pooled_sd) * np.sqrt(2 * (1 - correlation))
    return d_rm

def permutation_test_behavior(group1, group2, n_permutations=10000, plot=True, **kwargs):
    """
    Perform a permutation test to compare two groups.

    Parameters:
    - group1 (pandas Series or numpy array): Data for the first group.
    - group2 (pandas Series or numpy array): Data for the second group.
    - n_permutations (int, optional): Number of permutations to perform (default is 10000).
    - plot (bool, optional): Whether to plot the permutation distribution (default is True).

    Returns:
    - p_value (float): The p-value for the permutation test.

    The permutation test assesses whether the difference between the means of two groups is
    statistically significant by randomly permuting the data and computing the p-value.

    """

    # Initialize a list to save the results of each permutation
    results = list()

    # Concatenate the two groups to create a pooled dataset
    pooled = pd.concat([group1, group2])

    # Get the length of the first group (used for sampling)
    len_group = len(group1)

    # Calculate the observed difference in means between the two groups
    observed_diff = group1.mean() - group2.mean()

    # Perform n_permutations permutations
    for _ in range(n_permutations):
        # Randomly permute the pooled data
        permuted = np.random.permutation(pooled)

        # Calculate the mean for each permuted group
        assigned1 = permuted[:len_group].mean()
        assigned2 = permuted[len_group:].mean()

        # Calculate the difference in means for this permutation
        results.append(ttest_ind(a=assigned1, b=assigned2))

    # Convert results to a numpy array and take absolute values
    results = np.abs(np.array(results))

    # Count how many permutations have a difference as extreme as or more extreme than observed_diff
    values_as_or_more_extreme = sum(results >= observed_diff)

    # Calculate the p-value
    num_simulations = results.shape[0]
    p_value = values_as_or_more_extreme / num_simulations

    if plot:
        # Plot the permutation distribution and observed difference
        density_plot = sns.kdeplot(results, fill=True, **kwargs)
        density_plot.set(
            xlabel='Absolute Mean Difference Between Groups',
            ylabel='Proportion of Permutations'
        )

        # Add a line to show the actual difference observed in the data
        density_plot.axvline(
            x=observed_diff,
            color='red',
            linestyle='--'
        )

        # Add a legend to the plot
        plt.legend(
            labels=['Permutation Distribution', f'Observed Difference: {round(observed_diff, 2)}'],
            loc='upper right'
        )

        # Display the plot
        plt.show()

    return p_value


def remove_outliers(df, column_name, threshold=2):
    """
    Marks outliers in a DataFrame column as NaN based on standard deviation.

    Args:
        df: The input DataFrame.
        column_name: The name of the column to check for outliers.
        threshold: The number of standard deviations to use as a threshold.

    Returns:
        A new DataFrame with outliers marked as NaN, or the original DataFrame if no outliers are found
        or if the column is not numeric. Returns None if the column does not exist.
    """
    if column_name not in df.columns:
        print(f"Column '{column_name}' not found in DataFrame.")
        return None

    if not pd.api.types.is_numeric_dtype(df[column_name]):
        print(f"Column '{column_name}' is not numeric. Outlier marking not possible.")
        return df

    mean = df[column_name].mean()
    std = df[column_name].std()

    if std == 0:
        print(f"Standard deviation of column '{column_name}' is zero. No outliers marked.")
        return df

    upper_bound = mean + threshold * std
    lower_bound = mean - threshold * std

    df_copy = df.copy()  # Operate on a copy to avoid modifying the original DataFrame

    outlier_mask = (df_copy[column_name] < lower_bound) | (df_copy[column_name] > upper_bound)
    num_outliers = outlier_mask.sum()

    df_copy.loc[outlier_mask, column_name] = np.nan

    if num_outliers > 0:
        print(f"{num_outliers} outliers marked as NaN in column '{column_name}'.")
    else:
        print(f"No outliers found in column '{column_name}'.")

    return df_copy


def spatem_cluster_test_sensor_data(evokeds, conditions, pval=.05, n_permutations=1000, n_jobs=-1, plot=True):
    """
    The permutation test expects the data to be in the shape: observations × time × space. Observations here are epochs
    for single subjects, or the evoked response from one subject for multi-subject analysis.
    Approach inspired by
    doi:10.1016/j.neuroimage.2008.03.061
    doi:10.1016/j.jneumeth.2007.03.024
    doi:10.1111/psyp.13335
    """
    evokeds_data = dict()
    for condition in evokeds:
        evokeds_data[condition] = np.array(
            [evokeds[condition][e].get_data() for e in range(len(evokeds[list(evokeds.keys())[0]]))])
        evokeds_data[condition] = evokeds_data[condition].transpose(0, 2, 1)
    adjacency, _ = mne.channels.find_ch_adjacency(
        evokeds[list(evokeds.keys())[0]][0].info, None)
    X = [evokeds_data[conditions[0]], evokeds_data[conditions[1]]]
    t_obs, clusters, cluster_pv, h0 = mne.stats.spatio_temporal_cluster_test(
        X, threshold=dict(start=.2, step=.2), adjacency=adjacency, n_permutations=n_permutations, n_jobs=n_jobs)
    significant_points = cluster_pv.reshape(t_obs.shape).T < pval
    if plot:
        cond0 = mne.grand_average(evokeds[conditions[0]])
        cond1 = mne.grand_average(evokeds[conditions[1]])
        evoked_diff = mne.combine_evoked(
            [cond0, cond1], weights=[1, -1])
        evoked_diff.plot_joint()
        plt.close()
        selections = mne.channels.make_1020_channel_selections(evoked_diff.info, midline="z")
        fig, axes = plt.subplots(nrows=3, figsize=(8, 8))
        axes = {sel: ax for sel, ax in zip(selections, axes.ravel())}
        evoked_diff.plot_image(axes=axes, group_by=selections, colorbar=False, show=False,
                               mask=significant_points, show_names="all", titles=None)
        plt.colorbar(axes["Left"].images[-1], ax=list(axes.values()), shrink=.3,
                     label="µV")
        plt.show()
    return t_obs, clusters, cluster_pv, h0, significant_points


def permutation_test_sensor_data(data1, data2, n_permutations=1000, tail='both', plot=True):
    """
    TODO: NOT WORKING ATM
    Performs a nonparametric permutation test on EEG data.

    Args:
        data1 (array): EEG data for condition 1 (subjects x electrodes x time/frequency points).
        data2 (array): EEG data for condition 2 (subjects x electrodes x time/frequency points).
        n_permutations (int): Number of permutations to perform.
        tail (str): Type of test ('both', 'left', 'right').
        plot (bool): Whether to plot the results.

    Returns:
        array: p-values for each electrode/time-frequency point.
    """

    n_subjects1 = data1.shape[0]
    n_subjects2 = data2.shape[0]
    n_obs = data1.shape[1:]

    # Calculate observed t-statistic
    t_obs, _ = ttest_ind(data1[0, :, :], data2[0, :, :], axis=0)

    # Concatenate data
    data_concat = np.concatenate((data1, data2), axis=0)

    # Perform permutations
    t_perm = np.zeros((n_permutations,) + n_obs)
    for i in range(n_permutations):
        # Shuffle data
        perm_indices = np.random.permutation(n_subjects1 + n_subjects2)
        data_perm1 = data_concat[perm_indices[:n_subjects1]]
        data_perm2 = data_concat[perm_indices[n_subjects1:]]

        # Calculate t-statistic for permuted data
        t_perm[i], _ = ttest_ind(data_perm1[0, :, :], data_perm2[0, :, :], axis=0)

    # Calculate p-values
    if tail == 'both':
        p_values = np.mean(np.abs(t_perm) >= np.abs(t_obs), axis=0)
    elif tail == 'left':
        p_values = np.mean(t_perm <= t_obs, axis=0)
    elif tail == 'right':
        p_values = np.mean(t_perm >= t_obs, axis=0)
    else:
        raise ValueError("Invalid tail argument. Must be 'both', 'left', or 'right'.")

    if plot:
        # Plot the results
        if len(n_obs) == 2:  # Spatio-temporal (ERP) data
            plt.figure(figsize=(10, 6))
            plt.imshow(p_values, cmap='viridis', aspect='auto', origin='lower')
            plt.colorbar(label='p-value')
            plt.xlabel('Time')
            plt.ylabel('Electrode')
            plt.title('Permutation Test Results (Spatio-temporal)')
            plt.show()

        elif len(n_obs) == 3:  # Spatio-spectro-temporal (time-frequency) data
            for i in range(data1.shape):
                plt.figure(figsize=(10, 6))
                plt.imshow(p_values[:, i,:], cmap='viridis', aspect='auto', origin='lower')
                plt.colorbar(label='p-value')
                plt.xlabel('Time')
                plt.ylabel('Frequency')
                plt.title(f'Permutation Test Results (Spatio-spectro-temporal) - Electrode {i+1}')
                plt.show()

        else:
            raise ValueError("Data must have 2 or 3 dimensions (subjects x electrodes x time/frequency points).")

    return p_values


if __name__ == "__main__":
    import os
    import glob

    # define data root dir
    data_root = "/home/max/Insync/schulz.max5@gmail.com/GoogleDrive/PhD/data/SPACEPRIME/derivatives/preprocessing/"
    # get all the subject ids
    subjects = os.listdir(data_root)
    # load epochs
    epochs = mne.concatenate_epochs([mne.read_epochs(glob.glob(
        f"/home/max/Insync/schulz.max5@gmail.com/GoogleDrive/PhD/data/SPACEPRIME/derivatives/epoching/{subject}/eeg/{subject}_task-spaceprime-epo.fif")[
                                                         0]) for subject in subjects if
                                     int(subject.split("-")[1]) in [103, 104, 105, 106, 107]])
    all_conds = list(epochs.event_id.keys())
    # Separate epochs based on distractor location
    left_singleton_epochs = epochs[[x for x in all_conds if "Target-2-Singleton-1" in x]]
    right_singleton_epochs = epochs[[x for x in all_conds if "Target-2-Singleton-3" in x]]
    mne.epochs.equalize_epoch_counts([left_singleton_epochs, right_singleton_epochs], method="random")
    # get the contralateral evoked response and average
    contra_singleton_data = np.mean([left_singleton_epochs.copy().average(picks=["C4"]).get_data(),
                                     right_singleton_epochs.copy().average(picks=["C3"]).get_data()], axis=0)
    # get the ipsilateral evoked response and average
    ipsi_singleton_data = np.mean([left_singleton_epochs.copy().average(picks=["C3"]).get_data(),
                                   right_singleton_epochs.copy().average(picks=["C4"]).get_data()], axis=0)
    # now, do the same for the lateral targets
    # Separate epochs based on target location
    left_target_epochs = epochs[[x for x in all_conds if "Target-1-Singleton-2" in x]]
    right_target_epochs = epochs[[x for x in all_conds if "Target-3-Singleton-2" in x]]
    mne.epochs.equalize_epoch_counts([left_target_epochs, right_target_epochs], method="random")
    # get the contralateral evoked response and average
    contra_target_data = np.mean([left_target_epochs.copy().average(picks=["C4"]).get_data(),
                                  right_target_epochs.copy().average(picks=["C3"]).get_data()], axis=0)
    # get the ipsilateral evoked response and average
    ipsi_target_data = np.mean([left_target_epochs.copy().average(picks=["C3"]).get_data(),
                                right_target_epochs.copy().average(picks=["C4"]).get_data()], axis=0)

    # get the trial-wise data for targets
    contra_target_epochs_data = np.mean(np.concatenate([left_target_epochs.copy().get_data(picks="C4"),
                                                        right_target_epochs.copy().get_data(picks="C3")], axis=1),
                                        axis=1)
    ipsi_target_epochs_data = np.mean(np.concatenate([left_target_epochs.copy().get_data(picks="C3"),
                                                      right_target_epochs.copy().get_data(picks="C4")], axis=1), axis=1)
    # get the trial-wise data for singletons
    contra_singleton_epochs_data = np.mean(np.concatenate([left_singleton_epochs.copy().get_data(picks="C4"),
                                                           right_singleton_epochs.copy().get_data(picks="C3")], axis=1),
                                           axis=1)
    ipsi_singleton_epochs_data = np.mean(np.concatenate([left_singleton_epochs.copy().get_data(picks="C3"),
                                                         right_singleton_epochs.copy().get_data(picks="C4")], axis=1),
                                         axis=1)
    permutation_test_sensor_data(data1=contra_target_epochs_data.reshape((1, 717, 376)),
                                 data2=ipsi_target_epochs_data.reshape(1, 717, 376),
                                 n_permutations=100)
