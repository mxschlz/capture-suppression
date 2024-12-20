import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


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

def permutation_test(group1, group2, n_permutations=10000, plot=True, **kwargs):
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
        results.append(assigned1 - assigned2)

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