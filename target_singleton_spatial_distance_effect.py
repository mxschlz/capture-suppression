import pandas as pd
import statsmodels.api as sm
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf


fp = f"/home/max/data/behavior/all_subjects_additional_metrics.csv"
df = pd.read_csv(fp)

# calculate distance between target and singleton in dataframe
df = df[df.Singletonpres==1]
df["spatial_distance"] = np.abs(df["Targetpos"] - df["Singletonpos"])

md = smf.mixedlm("correct ~ spatial_distance", df, groups=df["subject"])
mdf = md.fit()
print(mdf.summary())


sns.scatterplot(data=df, x='spatial_distance', y='correct')  # Scatter plot of data points
sns.regplot(x='spatial_distance', y='correct', data=df, ci=95)  # Regression plot with 95% confidence interval
plt.xlabel('Target-to-Singleton Distance')
plt.ylabel('Accuracy')
plt.title('Linear Regression of Accuracy vs. Distance')
plt.savefig("/home/max/figures/spatial_distance_mixed_effects.png", dpi=400)
plt.show()
plt.close()
