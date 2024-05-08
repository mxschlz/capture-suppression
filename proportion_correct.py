import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
plt.ion()


fp = f"/home/max/data/behavior/all_subjects_additional_metrics.csv"
df = pd.read_csv(fp)

# remove outliers
outlier_thresh = 1.3
cleaned_rt = df.RT[df.RT < outlier_thresh]


sns.barplot(data=df, y=df.RT, hue="Singletonpres")
plt.savefig("/home/max/figures/rt_singleton_abs_vs_pres.png", dpi=400)
plt.close()

sns.barplot(data=df, y=df.correct, hue="Singletonpres")
plt.savefig("/home/max/figures/correct_singleton_abs_vs_pres.png", dpi=400)
plt.close()
