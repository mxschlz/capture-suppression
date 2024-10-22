import matplotlib
matplotlib.use("Qt5Agg")
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from color_palette import get_subpalette
plt.ion()


# insert color palette
sns.set_palette(list(get_subpalette([14, 84, 44]).values()))
# load up dataframe
df = pd.read_excel("/home/max/data/SPACEPRIME/sub-101/beh/sub-101_task-spaceprime.xlsx", index_col=0)
df = df[(df['event_type'] == 'mouse_click')]
# plot the data
barplot = sns.barplot(data=df, x="target_modulation", y="iscorrect", errorbar=None, hue="SingletonPresent")
plt.xlabel("Target pitch")
plt.ylabel("Proportion correct")
barplot.set_xticklabels(["Low", "High"])
plt.savefig("/home/max/figures/SPACEPRIME/iscorrect_high_and_low_targets.svg")
