import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from SPACEPRIME import load_concatenated_csv

# 1. Load data
df = load_concatenated_csv("target_towardness_all_variables.csv")

# 2. Calculate behavioral metrics on a subject level
metrics = ['rt', 'select_target', 'target_towardness']
subject_metrics = df.groupby(['subject_id', 'PrimingCondition'])[metrics].mean().reset_index()

# 3. Compute statistics manually
results = []
conditions = ['Negative', 'No', 'Positive']

for cond in conditions:
    subset = subject_metrics[subject_metrics['PrimingCondition'] == cond]
    
    for metric in metrics:
        data = subset[metric].dropna()
        n_valid = len(data)
        mean_val = data.mean()
        std_val = data.std()
        sem_val = std_val / np.sqrt(n_valid) if n_valid > 0 else np.nan
        q1 = data.quantile(0.25)
        q3 = data.quantile(0.75)
            
        results.append({
            'Condition': cond,
            'Metric': metric,
            'N': n_valid,
            'Mean': mean_val,
            'SEM': sem_val,
            'Median': data.median(),
            'Std (Spread)': std_val,
            'Min': data.min(),
            'Max': data.max(),
            'IQR': q3 - q1
        })

summary_df = pd.DataFrame(results)

print("--- Comprehensive Descriptive Statistics ---")
print(summary_df.to_string(index=False))

# 4. Plot comparison: Manual Matplotlib vs Seaborn
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
palette = {'No': 'grey', 'Positive': 'green', 'Negative': 'red'}
plot_order = ['No', 'Positive', 'Negative']

for i, metric in enumerate(metrics):
    # --- TOP ROW: Manual Matplotlib Bars ---
    ax_manual = axes[0, i]
    metric_data = summary_df[summary_df['Metric'] == metric]
    # Sort to match the desired plot order
    metric_data = metric_data.set_index('Condition').loc[plot_order].reset_index()
    
    means = metric_data['Mean'].values
    sems = metric_data['SEM'].values
    colors = [palette[cond] for cond in metric_data['Condition']]
    
    # Plot bars with explicit, symmetric standard error cap sizes
    ax_manual.bar(metric_data['Condition'], means, yerr=sems, capsize=5, 
                  color=colors, alpha=0.8, edgecolor='black', error_kw={'linewidth': 1.5})
    
    # Overlay single-subject data points
    sns.stripplot(data=subject_metrics, x='PrimingCondition', y=metric, 
                  order=plot_order, color='black', alpha=0.4, jitter=True, size=4, ax=ax_manual)
    
    ax_manual.set_title(f'MANUAL: {metric}\n(1 SEM)')
    ax_manual.set_ylabel('Mean value')
    
    ax_manual.spines['top'].set_visible(False)
    ax_manual.spines['right'].set_visible(False)

    # --- BOTTOM ROW: Seaborn Bars ---
    ax_sns = axes[1, i]
    sns.barplot(data=subject_metrics, x='PrimingCondition', y=metric, 
                order=plot_order, errorbar='se', palette=palette, 
                ax=ax_sns, capsize=0.1, alpha=0.8, edgecolor='black', errwidth=1.5)
                
    # Overlay single-subject data points
    sns.stripplot(data=subject_metrics, x='PrimingCondition', y=metric, 
                  order=plot_order, color='black', alpha=0.4, jitter=True, size=4, ax=ax_sns)
                
    ax_sns.set_title(f'SEABORN: {metric}\n(errorbar="se")')
    ax_sns.set_ylabel('Mean value')
    ax_sns.set_xlabel('Priming Condition')
    
    ax_sns.spines['top'].set_visible(False)
    ax_sns.spines['right'].set_visible(False)

plt.tight_layout()
plt.show()