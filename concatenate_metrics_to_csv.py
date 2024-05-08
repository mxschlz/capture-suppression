import pandas as pd
import numpy as np


# load dataframe
fp = (f"/home/max/data/behavior/all_subjects.csv")
df = pd.read_csv(fp)

# make column for each metric
# df["prop_correct_singletonabs"] = np.zeros(len(df))
# df["prop_correct_singletonpres"] = np.zeros(len(df))
# df["rt_singletonpres"] = np.zeros(len(df))
# df["rt_singletonabs"] = np.zeros(len(df))
df["congruent"] = np.zeros(len(df))
df["correct"] = np.zeros(len(df))

# iterate over subjects and fill in indices with metric values
for subject in np.unique(df.subject):
    subdata = df[df.subject == subject]
    # get singleton absent and present trials
    # singletonpresent = subdata[subdata.Singletonpres == 1]
    # singletonabsent = subdata[subdata.Singletonpres == 0]

    # prop_corr_singletonpres = len(singletonpresent[singletonpresent.Targetdir == singletonpresent.Response]) / len(singletonpresent)
    # prop_corr_singletonabs = len(singletonabsent[singletonabsent.Targetdir == singletonabsent.Response]) / len(singletonabsent)
    # rt_pres = np.mean(singletonpresent.RT)
    # rt_abs = np.mean(singletonabsent.RT)
    # df["prop_correct_singletonabs"][df.subject == subject] = prop_corr_singletonabs
    # df["prop_correct_singletonpres"][df.subject == subject] = prop_corr_singletonpres
    # df["rt_singletonpres"][df.subject == subject] = rt_pres
    # df["rt_singletonabs"][df.subject == subject] = rt_abs

    # check for within-subject congruent trials (insert: 1 is congruent, 0 is incongruent)

    for idx, row in subdata.iterrows():
        targetdir = row["Targetdir"]
        singletondir = row["Singletondir"]
        if targetdir == singletondir:
            df["congruent"][idx] = 1
        else:
            df["congruent"][idx] = 0
        response = row["Response"]
        targetdir = row["Targetdir"]
        if response == targetdir:
            df["correct"][idx] = 1
        else:
            df["correct"][idx] = 0

df.to_csv("/home/max/data/behavior/all_subjects_additional_metrics.csv", index=False)


