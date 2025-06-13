import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import ttest_ind, f


def cohen_d(x, y):
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    return (np.mean(x) - np.mean(y)) / np.sqrt(((nx - 1) * np.std(x, ddof=1) ** 2 + (ny - 1) * np.std(y, ddof=1) ** 2) / dof)


def convert_effsize(ef, input_type, output_type, nx=None, ny=None):
    """Conversion between effect sizes.

    Parameters
    ----------
    ef : float
        Original effect size.
    input_type : string
        Effect size type of ef. Must be ``'cohen'`` or ``'pointbiserialr'``.
    output_type : string
        Desired effect size type. Available methods are:

        * ``'cohen'``: Unbiased Cohen d
        * ``'hedges'``: Hedges g
        * ``'pointbiserialr'``: Point-biserial correlation
        * ``'eta-square'``: Eta-square
        * ``'odds-ratio'``: Odds ratio
        * ``'AUC'``: Area Under the Curve
        * ``'none'``: pass-through (return ``ef``)

    nx, ny : int, optional
        Length of vector x and y. Required to convert to Hedges g.

    Returns
    -------
    ef : float
        Desired converted effect size

    See Also
    --------
    compute_effsize : Calculate effect size between two set of observations.
    compute_effsize_from_t : Convert a T-statistic to an effect size.

    Notes
    -----
    The formula to convert from a`point-biserial correlation
    <https://en.wikipedia.org/wiki/Point-biserial_correlation_coefficient>`_ **r** to **d** is
    given in [1]_:

    .. math:: d = \\frac{2r_{pb}}{\\sqrt{1 - r_{pb}^2}}

    The formula to convert **d** to a point-biserial correlation **r** is given in [2]_:

    .. math::

        r_{pb} = \\frac{d}{\\sqrt{d^2 + \\frac{(n_x + n_y)^2 - 2(n_x + n_y)}
        {n_xn_y}}}

    The formula to convert **d** to :math:`\\eta^2` is given in [3]_:

    .. math:: \\eta^2 = \\frac{(0.5 d)^2}{1 + (0.5 d)^2}

    The formula to convert **d** to an odds-ratio is given in [4]_:

    .. math:: \\text{OR} = \\exp (\\frac{d \\pi}{\\sqrt{3}})

    The formula to convert **d** to area under the curve is given in [5]_:

    .. math:: \\text{AUC} = \\mathcal{N}_{cdf}(\\frac{d}{\\sqrt{2}})

    References
    ----------
    .. [1] Rosenthal, Robert. "Parametric measures of effect size."
       The handbook of research synthesis 621 (1994): 231-244.

    .. [2] McGrath, Robert E., and Gregory J. Meyer. "When effect sizes
       disagree: the case of r and d." Psychological methods 11.4 (2006): 386.

    .. [3] Cohen, Jacob. "Statistical power analysis for the behavioral
       sciences. 2nd." (1988).

    .. [4] Borenstein, Michael, et al. "Effect sizes for continuous data."
       The handbook of research synthesis and meta-analysis 2 (2009): 221-235.

    .. [5] Ruscio, John. "A probability-based measure of effect size:
       Robustness to base rates and other factors." Psychological methods 1
       3.1 (2008): 19.
    """
    it = input_type.lower()
    ot = output_type.lower()

    # Pass-through option
    if it == ot or ot == "none":
        return ef

    # Convert point-biserial r to Cohen d (Rosenthal 1994)
    d = (2 * ef) / np.sqrt(1 - ef**2) if it == "pointbiserialr" else ef

    # Then convert to the desired output type
    if ot == "cohen":
        return d
    elif ot == "pointbiserialr":
        # McGrath and Meyer 2006
        if all(v is not None for v in [nx, ny]):
            a = ((nx + ny) ** 2 - 2 * (nx + ny)) / (nx * ny)
        else:
            a = 4
        return d / np.sqrt(d**2 + a)
    elif ot == "eta-square":
        # Cohen 1988
        return (d / 2) ** 2 / (1 + (d / 2) ** 2)
    elif ot == "odds-ratio":
        # Borenstein et al. 2009
        return np.exp(d * np.pi / np.sqrt(3))
    elif ot == "r":
        # https://github.com/raphaelvallat/pingouin/issues/302
        raise ValueError(
            "Using effect size 'r' in `pingouin.convert_effsize` has been deprecated. "
            "Please use 'pointbiserialr' instead."
        )
    else:  # ['auc']
        # Ruscio 2008
        from scipy.stats import norm

        return norm.cdf(d / np.sqrt(2))

def compute_effsize_from_t(tval, nx=None, ny=None, N=None, eftype="cohen"):
    """Compute effect size from a T-value.

    Parameters
    ----------
    tval : float
        T-value
    nx, ny : int, optional
        Group sample sizes.
    N : int, optional
        Total sample size (will not be used if nx and ny are specified)
    eftype : string, optional
        Desired output effect size.

    Returns
    -------
    ef : float
        Effect size

    See Also
    --------
    compute_effsize : Calculate effect size between two set of observations.
    convert_effsize : Conversion between effect sizes.

    Notes
    -----
    If both nx and ny are specified, the formula to convert from *t* to *d* is:

    .. math:: d = t * \\sqrt{\\frac{1}{n_x} + \\frac{1}{n_y}}

    If only N (total sample size) is specified, the formula is:

    .. math:: d = \\frac{2t}{\\sqrt{N}}
    """

    if not isinstance(tval, float):
        err = "T-value must be float"
        raise ValueError(err)

    # Compute Cohen d (Lakens, 2013)
    if nx is not None and ny is not None:
        d = tval * np.sqrt(1 / nx + 1 / ny)
    elif N is not None:
        d = 2 * tval / np.sqrt(N)
    else:
        raise ValueError("You must specify either nx + ny, or just N")
    return convert_effsize(d, "cohen", eftype, nx=nx, ny=ny)


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
    observed_diff = ttest_ind(group1.mean() - group2.mean())

    # Perform n_permutations permutations
    for _ in range(n_permutations):
        # Randomly permute the pooled data
        permuted = np.random.permutation(pooled)

        # Calculate the mean for each permuted group
        assigned1 = permuted[:len_group].mean()
        assigned2 = permuted[len_group:].mean()

        # Calculate the difference in means for this permutation
        results.append(ttest_ind(a=assigned1, b=assigned2)[0])

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


def remove_outliers(df, column_name, subject_id_column='subject_id', threshold=2):
    """
    Marks outliers in a DataFrame column as NaN, per subject, based on standard deviation.

    Args:
        df: The input DataFrame.
        column_name: The name of the column to check for outliers (e.g., 'reaction_time').
        subject_id_column: The name of the column identifying subjects (e.g., 'subject_id').
        threshold: The number of standard deviations to use as a threshold.

    Returns:
        A new DataFrame with outliers marked as NaN, or the original DataFrame if no outliers are found
        or if the column is not numeric. Returns None if required columns are missing.
    """

    if column_name not in df.columns:
        print(f"Column '{column_name}' not found in DataFrame.")
        return None

    if subject_id_column not in df.columns:
        print(f"Subject ID column '{subject_id_column}' not found in DataFrame.")
        return None

    if not pd.api.types.is_numeric_dtype(df[column_name]):
        print(f"Column '{column_name}' is not numeric. Outlier marking not possible.")
        return df

    df_copy = df.copy()  # Work on a copy

    def _remove_outliers_single_subject(subject_df, col_name, thresh):
        """Helper function to remove outliers for a single subject."""
        mean = subject_df[col_name].mean()
        std = subject_df[col_name].std()

        if std == 0:  # Handle cases where all RTs are identical for a subject
            print(
                f"Standard deviation of column '{col_name}' is zero for subject {subject_df[subject_id_column].iloc[0]}. No outliers marked.")
            return subject_df

        upper_bound = mean + thresh * std
        lower_bound = mean - thresh * std

        outlier_mask = (subject_df[col_name] < lower_bound) | (subject_df[col_name] > upper_bound)
        subject_df.loc[outlier_mask, col_name] = np.nan
        num_outliers = outlier_mask.sum()

        if num_outliers > 0:
            print(f"Subject: {subject_df[subject_id_column].iloc[0]}, {num_outliers} outliers marked")

        return subject_df

    # Group by subject ID and apply the outlier removal function to each subject's data
    df_copy = df_copy.groupby(subject_id_column, group_keys=False).apply(
        _remove_outliers_single_subject, col_name=column_name, thresh=threshold
    )
    return df_copy


def cronbach_alpha(
    data=None, items=None, scores=None, subject=None, nan_policy="pairwise", ci=0.95
):
    """Cronbach's alpha reliability measure.

    Parameters
    ----------
    data : :py:class:`pandas.DataFrame`
        Wide or long-format dataframe.
    items : str
        Column in ``data`` with the items names (long-format only).
    scores : str
        Column in ``data`` with the scores (long-format only).
    subject : str
        Column in ``data`` with the subject identifier (long-format only).
    nan_policy : bool
        If `'listwise'`, remove the entire rows that contain missing values
        (= listwise deletion). If `'pairwise'` (default), only pairwise
        missing values are removed when computing the covariance matrix.
        For more details, please refer to the :py:meth:`pandas.DataFrame.cov`
        method.
    ci : float
        Confidence interval (.95 = 95%)

    Returns
    -------
    alpha : float
        Cronbach's alpha

    Notes
    -----
    This function works with both wide and long format dataframe. If you pass a
    long-format dataframe, you must also pass the ``items``, ``scores`` and
    ``subj`` columns (in which case the data will be converted into wide
    format using the :py:meth:`pandas.DataFrame.pivot` method).

    Internal consistency is usually measured with Cronbach's alpha [1]_,
    a statistic calculated from the pairwise correlations between items.
    Internal consistency ranges between negative infinity and one.
    Coefficient alpha will be negative whenever there is greater
    within-subject variability than between-subject variability.

    Cronbach's :math:`\\alpha` is defined as

    .. math::

        \\alpha ={k \\over k-1}\\left(1-{\\sum_{{i=1}}^{k}\\sigma_{{y_{i}}}^{2}
        \\over\\sigma_{x}^{2}}\\right)

    where :math:`k` refers to the number of items, :math:`\\sigma_{x}^{2}`
    is the variance of the observed total scores, and
    :math:`\\sigma_{{y_{i}}}^{2}` the variance of component :math:`i` for
    the current sample of subjects.

    Another formula for Cronbach's :math:`\\alpha` is

    .. math::

        \\alpha = \\frac{k \\times \\bar c}{\\bar v + (k - 1) \\times \\bar c}

    where :math:`\\bar c` refers to the average of all covariances between
    items and :math:`\\bar v` to the average variance of each item.

    95% confidence intervals are calculated using Feldt's method [2]_:

    .. math::

        c_L = 1 - (1 - \\alpha) \\cdot F_{(0.025, n-1, (n-1)(k-1))}

        c_U = 1 - (1 - \\alpha) \\cdot F_{(0.975, n-1, (n-1)(k-1))}

    where :math:`n` is the number of subjects and :math:`k` the number of
    items.

    Results have been tested against the `psych
    <https://cran.r-project.org/web/packages/psych/psych.pdf>`_ R package.

    References
    ----------
    .. [1] http://www.real-statistics.com/reliability/cronbachs-alpha/

    .. [2] Feldt, Leonard S., Woodruff, David J., & Salih, Fathi A. (1987).
           Statistical inference for coefficient alpha. Applied Psychological
           Measurement, 11(1):93-103.

    Examples
    --------
    Binary wide-format dataframe (with missing values)

    data = pg.read_dataset('cronbach_alpha')
    pg.cronbach_alpha(data=data, items='Items', scores='Scores', subject='Subj')
    (0.5917188485995826, array([0.195, 0.84 ]))
    """
    # Safety check
    assert isinstance(data, pd.DataFrame), "data must be a dataframe."
    assert nan_policy in ["pairwise", "listwise"]

    if all([v is not None for v in [items, scores, subject]]):
        # Data in long-format: we first convert to a wide format
        data = data.pivot(index=subject, values=scores, columns=items)

    # From now we assume that data is in wide format
    n, k = data.shape
    assert k >= 2, "At least two items are required."
    assert n >= 2, "At least two raters/subjects are required."
    err = "All columns must be numeric."
    assert all([data[c].dtype.kind in "bfiu" for c in data.columns]), err
    if data.isna().any().any() and nan_policy == "listwise":
        # In R = psych:alpha(data, use="complete.obs")
        data = data.dropna(axis=0, how="any")

    # Compute covariance matrix and Cronbach's alpha
    C = data.cov(numeric_only=True)
    cronbach = (k / (k - 1)) * (1 - np.trace(C) / C.sum().sum())
    # which is equivalent to
    # v = np.diag(C).mean()
    # c = C.to_numpy()[np.tril_indices_from(C, k=-1)].mean()
    # cronbach = (k * c) / (v + (k - 1) * c)

    # Confidence intervals
    alpha = 1 - ci
    df1 = n - 1
    df2 = df1 * (k - 1)
    lower = 1 - (1 - cronbach) * f.isf(alpha / 2, df1, df2)
    upper = 1 - (1 - cronbach) * f.isf(1 - alpha / 2, df1, df2)
    return cronbach, np.round([lower, upper], 3)


def split_dataframe_by_blocks_balanced_subjects(df, block_col, subject_col, seed=None):
    """
    Splits a DataFrame into two halves based on blocks, ensuring that
    subjects are distributed as evenly as possible across the two halves.

    Args:
        df: The Pandas DataFrame to split.
        block_col: Name of the column containing the block number.
        subject_col: Name of the column containing the subject ID.
        seed: Optional integer seed for reproducibility.

    Returns:
        A tuple containing two DataFrames: (df1, df2), the two split halves.
    """

    if seed is not None:
        np.random.seed(seed)

    unique_blocks = df[block_col].unique()
    unique_subjects = df[subject_col].unique()
    num_blocks = len(unique_blocks)
    half_blocks = num_blocks // 2

    # --- Step 1:  Assign blocks to splits, balancing subjects ---

    # Shuffle the blocks to ensure randomness in block assignment
    shuffled_blocks = np.random.permutation(unique_blocks)

    split1_blocks = []
    split2_blocks = []
    split1_subjects = set()
    split2_subjects = set()

    for block in shuffled_blocks:
        # Get subjects participating in the current block
        block_subjects = set(df[df[block_col] == block][subject_col])

        # Check if adding the block to split1 would cause a subject imbalance
        potential_split1_subjects = split1_subjects.union(block_subjects)
        potential_split2_subjects = split2_subjects.union(block_subjects)  # Check the contrary as well.

        # Preferentially add the block to the split with fewer subjects *currently*.  This is the key balancing step.
        if (len(split1_blocks) < half_blocks and len(split1_subjects) <= len(split2_subjects)) or len(
                split2_blocks) >= half_blocks:
            split1_blocks.append(block)
            split1_subjects.update(block_subjects)
        else:
            split2_blocks.append(block)
            split2_subjects.update(block_subjects)

    # --- Step 2: Create the DataFrames based on the block assignments ---
    df1 = df[df[block_col].isin(split1_blocks)]
    df2 = df[df[block_col].isin(split2_blocks)]

    return df1, df2


def r_squared_mixed_model(lmm_results):
    """
    Calculates marginal and conditional R-squared for a statsmodels LMM.

    Based on Nakagawa & Schielzeth (2013) and extensions.

    Args:
        lmm_results: A fitted statsmodels MixedLMResults object.

    Returns:
        dict: A dictionary with 'marginal_r2' and 'conditional_r2'.
              Returns None if calculation fails (e.g., model did not converge).
    """
    # ICC
    var_subject = lmm_results.cov_re.iloc[0, 0]  # Variance of random intercepts
    var_residual = lmm_results.scale  # Residual variance
    icc = var_subject / (var_subject + var_residual)
    # R² calculations (Nakagawa & Schielzeth)
    var_fixed = np.var(lmm_results.fittedvalues)
    r2_marginal = var_fixed / (var_fixed + var_subject + var_residual)
    r2_conditional = (var_fixed + var_subject) / (var_fixed + var_subject + var_residual)
    # Output
    print(f"\nIntraclass Correlation Coefficient (ICC): {icc:.3f}")
    print(f"Marginal R²: {r2_marginal:.3f}")
    print(f"Conditional R²: {r2_conditional:.3f}")


if __name__ == "__main__":
    pass
