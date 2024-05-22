import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
plt.ion()


fp = f"/home/max/data/behavior/all_subjects_additional_metrics.csv"
df = pd.read_csv(fp)

crosstab = pd.crosstab(index=df.Singletonpos, columns=df.Targetpos, normalize=False)
crosstab_flipped = np.flip(crosstab.values, axis=0)

crosstab.plot(kind="bar")
plt.savefig("/home/max/figures/singletonpos_bar.png", dpi=400)
plt.close()

crosstab = pd.crosstab(index=df.Singletontime, columns=df.Targettime, normalize=False)
crosstab_flipped = np.flip(crosstab.values, axis=0)

crosstab.plot(kind="bar")
plt.savefig("/home/max/figures/singletontime_bar.png", dpi=400)
plt.close()