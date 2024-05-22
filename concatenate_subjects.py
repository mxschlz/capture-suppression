import pandas as pd


# List of subject numbers
subjects = list(range(0, 7))

dfs = []

for subject in subjects:
    fp = f"/home/max/data/behavior/sub{subject}.csv"
    df = pd.read_csv(fp)

    # Add subject number as a level in the hierarchical index
    df['subject'] = subject
    df.set_index(['subject'], append=True, inplace=True)

    dfs.append(df)

# Concatenate DataFrames into one DataFrame with hierarchical indices
df = pd.concat(dfs)

# Reset index to flatten hierarchical index
df.reset_index(level='subject', inplace=True)

df = df.fillna(int(0))
df.to_csv("/home/max/data/behavior/all_subjects.csv", index=False)
