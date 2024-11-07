import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from color_palette import get_subpalette
plt.ion()


# insert color palette
sns.set_palette(list(get_subpalette([14, 84, 44]).values()))
# load up dataframe
df = pd.read_csv("/home/max/Insync/schulz.max5@gmail.com/GoogleDrive/PhD/data/SPACEPRIME/derivatives/preprocessing/sub-102/beh/sub-102_clean.csv", index_col=0)
df = df[(df['event_type'] == 'mouse_click') & (df["phase"] != 2)]
# plot the data
barplot = sns.barplot(data=df, x="target_modulation", y="iscorrect", errorbar=None, hue="SingletonPresent")
plt.xlabel("Target pitch")
plt.ylabel("Proportion correct")
barplot.set_xticklabels(["Low", "High"])
plt.savefig("/home/max/figures/SPACEPRIME/iscorrect_high_and_low_targets.svg")
