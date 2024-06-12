import pandas as pd


fp = f"/home/max/data/behavior/all_subjects_additional_metrics.csv"
df = pd.read_csv(fp)

df["spatial_priming"] = "no_priming"  # no priming as default

# Create columns for the previous trial's target and distractor positions
df['Prev_Target_spatial'] = df['Targetpos'].shift(1)
df['Prev_Distractor_spatial'] = df['Singletonpos'].shift(1)

# Set the conditions for Negative Priming and Competitor Priming
df.loc[df['Targetpos'] == df['Prev_Distractor_spatial'], 'spatial_priming'] = 'negative_priming'
df.loc[df['Singletonpos'] == df['Prev_Target_spatial'], 'spatial_priming'] = 'positive_priming'

df["temporal_priming"] = "no_priming"  # no priming as default

# Create columns for the previous trial's target and distractor positions
df['Prev_Target_temporal'] = df['Targettime'].shift(1)
df['Prev_Distractor_temporal'] = df['Singletontime'].shift(1)

# Set the conditions for Negative Priming and Competitor Priming
df.loc[df['Targettime'] == df['Prev_Distractor_temporal'], 'temporal_priming'] = 'negative_priming'
df.loc[df['Singletontime'] == df['Prev_Target_temporal'], 'temporal_priming'] = 'positive_priming'

# congruency priming
df["congruency_priming"] = "no_priming"  # no priming as default

# Create columns for the previous trial's target and distractor positions
df['Prev_Target_congruency'] = df['Targetdir'].shift(1)
df['Prev_Distractor_congruency'] = df['Singletondir'].shift(1)

# Set the conditions for Negative Priming and Competitor Priming
df.loc[df['Targetdir'] == df['Prev_Distractor_congruency'], 'congruency_priming'] = 'negative_priming'
df.loc[df['Singletondir'] == df['Prev_Target_congruency'], 'congruency_priming'] = 'positive_priming'

# save to csv
df.to_csv("/home/max/data/behavior/all_subjects_additional_metrics_and_priming.csv", index=False)
