import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
import numpy as np
plt.ion()


fp = f"/home/max/data/behavior/all_subjects_additional_metrics.csv"
df = pd.read_csv(fp)

crosstab = pd.crosstab(index=df.Singletonpos, columns=df.Targetpos)
minmaxsclaer = preprocessing.MinMaxScaler()
crosstab_normalized = minmaxsclaer.fit_transform(crosstab)

figure = sns.heatmap(crosstab_normalized[1:], annot=True, xticklabels=range(1, 6), yticklabels=range(1, 6))
figure.set(xlabel="Singletonpos", ylabel="Targetpos")

plt.savefig("/home/max/figures/singletonpos_targetpos_crosstab.png", dpi=400)
plt.close()

sns.displot(df.Singletonpos[df.Singletonpres == 1], kde=False, rug=True)
plt.savefig("/home/max/figures/singletonpos_distribution.png", dpi=400)
plt.close()

sns.displot(df.Targetpos, kde=False, rug=True)
plt.savefig("/home/max/figures/targetpos_distribution.png", dpi=400)
plt.close()
