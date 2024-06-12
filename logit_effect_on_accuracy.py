import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import logit
plt.ion()


fp = f"/home/max/data/behavior/all_subjects_additional_metrics.csv"
df = pd.read_csv(fp)


cr_ct = pd.crosstab(df.Singletonpos, df.Targetpos, values=df.correct, aggfunc="mean")
cr_ct_flipped = np.flip(cr_ct.values, axis=0)
# logit transform
figure = sns.heatmap(logit(cr_ct_flipped), annot=True, xticklabels=range(1, 6), yticklabels=[5, 4, 3, 2, 1, 0],
                     cmap="Greys")
figure.set(xlabel="Targetpos", ylabel="Singletonpos")
plt.savefig("/home/max/figures/cr_crosstab_targetpos_vs_singletonpos_logit_transformed.png", dpi=400)
plt.close()


cr_ct = pd.crosstab(df.Singletontime, df.Targettime, values=df.correct, aggfunc="mean")
cr_ct_flipped = np.flip(cr_ct.values, axis=0)
# logit transform
figure = sns.heatmap(logit(cr_ct_flipped), annot=True, xticklabels=range(1, 6), yticklabels=[5, 4, 3, 2, 1, 0],
                     cmap="Greys")
figure.set(xlabel="Targettime", ylabel="Singletontime")
plt.savefig("/home/max/figures/cr_crosstab_targettime_vs_singletontime_logit_transformed.png", dpi=400)
plt.close()
