import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme(context="talk", style="ticks")
from SPACEPRIME import load_concatenated_csv
import pingouin as pg


df = load_concatenated_csv("target_towardness_all_variables.csv")

# 1. Calculate behavioral metrics on a subject level
subject_metrics = df.groupby(['subject_id', 'PrimingCondition'])[['rt', 'select_target', "target_towardness"]].mean().reset_index()

def plot_priming_comparison(data, condition_pairs, title_suffix):
    """Helper function to plot RT and Accuracy for specific condition pairs."""
    subset = data[data['PrimingCondition'].isin(condition_pairs)]
    
    if subset.empty:
        print(f"No data found for conditions: {condition_pairs}")
        return
    
    palette = {'No': 'grey', 'Positive': 'green', 'Negative': 'red'}
    
    # Calculate stats
    wide_df = subset.pivot(index='subject_id', columns='PrimingCondition', values=['rt', 'select_target', 'target_towardness']).dropna()
    
    # RT Stats
    rt_stats = pg.ttest(wide_df[('rt', condition_pairs[0])], wide_df[('rt', condition_pairs[1])], paired=True)
    rt_p = rt_stats['p-val'][0]
    rt_d = rt_stats['cohen-d'][0]
    
    # Accuracy Stats
    acc_stats = pg.ttest(wide_df[('select_target', condition_pairs[0])], wide_df[('select_target', condition_pairs[1])], paired=True)
    acc_p = acc_stats['p-val'][0]
    acc_d = acc_stats['cohen-d'][0]

    # Target Towardness Stats
    tt_stats = pg.ttest(wide_df[('target_towardness', condition_pairs[0])], wide_df[('target_towardness', condition_pairs[1])], paired=True)
    tt_p = tt_stats['p-val'][0]
    tt_d = tt_stats['cohen-d'][0]

    def add_significance(ax, y_data, p_val):
        y_max = y_data.max()
        y_start = y_max + (y_max * 0.05)
        h = y_max * 0.02
        
        if p_val < 0.001: sig = '***'
        elif p_val < 0.01: sig = '**'
        elif p_val < 0.05: sig = '*'
        else: sig = 'n.s.'
        
        ax.plot([0, 0, 1, 1], [y_start, y_start + h, y_start + h, y_start], lw=1.5, c='k')
        ax.text(0.5, y_start + h, sig, ha='center', va='bottom', color='k')
        return y_start + h

    fig, axes = plt.subplots(1, 3, figsize=(9, 6))

    # Plot Reaction Time
    sns.barplot(data=subset, x='PrimingCondition', y='rt', order=condition_pairs, ax=axes[0], errorbar=('ci', 95), alpha=1.0, palette=palette)
    #sns.stripplot(data=subset, x='PrimingCondition', y='rt', order=condition_pairs, ax=axes[0], color='black', alpha=0.2, jitter=True)
    axes[0].set_title(f'Reaction Time\n$d={rt_d:.2f}$')
    axes[0].set_ylabel('Mean RT (s)')
    add_significance(axes[0], subset['rt'], rt_p)

    # Plot Accuracy (select_target)
    sns.barplot(data=subset, x='PrimingCondition', y='select_target', order=condition_pairs, ax=axes[1], errorbar=('ci', 95), alpha=1.0, palette=palette)
    #sns.stripplot(data=subset, x='PrimingCondition', y='select_target', order=condition_pairs, ax=axes[1], color='black', alpha=0.2, jitter=True)
    axes[1].set_title(f'Accuracy\n$d={acc_d:.2f}$')
    axes[1].set_ylabel('Proportion Correct')
    
    top_y = add_significance(axes[1], subset['select_target'], acc_p)
    axes[1].set_ylim(0, max(1.1, top_y * 1.1))

    # Plot Target Towardness
    sns.barplot(data=subset, x='PrimingCondition', y='target_towardness', order=condition_pairs, ax=axes[2], errorbar=('ci', 95), alpha=1.0, palette=palette)
    axes[2].set_title(f'Target Towardness\n$d={tt_d:.2f}$')
    axes[2].set_ylabel('Target Towardness')
    add_significance(axes[2], subset['target_towardness'], tt_p)

    sns.despine()
    plt.tight_layout()
    plt.show()

# 2. Plot No Priming vs Positive Priming
plot_priming_comparison(subject_metrics, ['No', 'Positive'], "No vs Positive")
plt.savefig("plots/no_vs_positive.svg")

# 3. Plot No Priming vs Negative Priming
plot_priming_comparison(subject_metrics, ['No', 'Negative'], "No vs Negative")
plt.savefig("plots/no_vs_negative.svg")
