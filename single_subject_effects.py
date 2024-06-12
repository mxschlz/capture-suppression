import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
plt.ion()


fp = f"/home/max/data/behavior/all_subjects_additional_metrics.csv"
df = pd.read_csv(fp)
# drop unrealistically high RT value
# df.RT[df.RT > 2 * np.std(df.RT)] = np.nan  # drop values over 2 standard deviations of the mean

sns.barplot(data=df, y=df.correct, hue="subject")
plt.savefig("/home/max/figures/proportion_correct_single_subs_n7.png", dpi=400)
plt.close()
sns.barplot(data=df, y=df.RT, hue="subject")
plt.savefig("/home/max/figures/rt_single_subs_n7.png", dpi=400)
plt.close()
