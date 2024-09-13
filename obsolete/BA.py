import pandas as pd
import os
import glob


# define data directories
root = "/home/max/obsolete/BA/"
conditions = ["f2f", "vcall", "prerec"]
for condition in conditions:
	file_pattern = f"STIP_*_{condition}*_*.csv"  # Adjust pattern if needed
	all_files = glob.glob(os.path.join(root, file_pattern))

	# Create a dictionary to store paired dataframes
	paired_dataframes = {}

	for file in all_files:
		# Extract group and base filename (without group)
		group = "TARG" if "TARG" in file else "OBS"
		base_filename = file.replace("_TARG_", "_").replace("_OBS_", "_")

		# Read the dataframe
		df = pd.read_csv(file, header=None, names=[f'Rating_{group}', f'Time_{group}'])
		df['Condition'] = condition
		df['Group'] = group

		# Store in the dictionary, create a new list if the base filename is not seen before
		if base_filename not in paired_dataframes:
			paired_dataframes[base_filename] = []
		paired_dataframes[base_filename].append(df)

	# Process paired dataframes and create the final dataframe
	final_dataframes = []
	for base_filename, dfs in paired_dataframes.items():
		if len(dfs) != 2:
			print(f"Warning: {base_filename} does not have a pair. Skipping.")
			continue

		# Merge the two dataframes in the pair
		merged_df = pd.merge(dfs[0], dfs[1], left_index=True, right_index=True, suffixes=('', '_other'))
		#Extract observer-target identity and story from the filename
		filename_parts = base_filename.split('_')

		# Find the indices of "obs" and "targ" parts in the filename
		obs_index = next((i for i, part in enumerate(filename_parts) if part.startswith("obs")), None)
		targ_index = next((i for i, part in enumerate(filename_parts) if part.startswith("targ")), None)

		# Construct the Obs_Targ_Identity by combining the two parts
		if obs_index is not None and targ_index is not None:
			obs_targ_identity = f"{filename_parts[obs_index]}-{filename_parts[targ_index]}"
		else:
			obs_targ_identity = "Unknown"  # Handle cases where obs or targ is not found

		story = next((part for part in filename_parts if part.startswith("story")), None)

		merged_df['Obs_Targ_Identity'] = obs_targ_identity
		merged_df['Story'] = story
		# Drop the obsolete columns
		merged_df.drop(columns=['Group', 'Group_other', 'Time_TARG', "Condition_other"], inplace=True)  # Adjust 'Time_other' if needed

		final_dataframes.append(merged_df)

	# Concatenate all merged dataframes
	final_df = pd.concat(final_dataframes, ignore_index=True)
	final_df = final_df.rename(columns={'Time_OBS': 'Time'})
	# some tidying
	final_df = final_df.rename(columns={'Rating_TARG': 'Emotional_Intensity_Target', 'Rating_OBS': 'Emotional_Intensity_Observer'})
	# Split `Obs_Targ_Identity` into two columns
	final_df[['Observer_ID', 'Target_ID']] = final_df['Obs_Targ_Identity'].str.split('-', expand=True)
	# final_df.drop(columns=["Obs_Targ_Identity"], inplace=True)

	# Convert `Story` to categorical
	final_df['Story'] = final_df['Story'].astype('category')
	# save data
	final_df.to_excel(f"/home/max/obsolete/BA/final_data/{condition}_final.xlsx", index=False)


# plot some data
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_excel("/home/max/obsolete/BA/final_data/f2f_final.xlsx")
# Melt the dataframe to have 'Emotional_Intensity' and 'Role' columns
# Melt the dataframe to have 'Emotional_Intensity' and 'Role' columns
df_melted = df.melt(id_vars=['Time', 'Obs_Targ_Identity', 'Story'],
                    value_vars=['Emotional_Intensity_Observer', 'Emotional_Intensity_Target'],
                    var_name='Role', value_name='Emotional_Intensity')
# Get unique stories
unique_stories = df_melted['Story'].unique()
story1 = df_melted[df_melted['Story'] == unique_stories[0]]
story2 = df_melted[df_melted['Story'] == unique_stories[1]]
# get subsample of data
for identity in list(story2["Obs_Targ_Identity"].unique()):
	data = story2[story2["Obs_Targ_Identity"]==identity]
	# Create the line plot with different styles for each Obs_Targ_Identity
	sns.lineplot(x='Time', y='Emotional_Intensity', hue='Role', data=data)
	plt.savefig(f"/home/max/obsolete/BA/own_plots/f2f_{identity}_story2.jpg")
	plt.close()

# Define the window size for the moving average (adjust as needed)
window_size = 15  # You can experiment with different window sizes
# Calculate the moving average for each 'Role' and 'Obs_Targ_Identity' combination
df_melted['Smoothed_Emotional_Intensity'] = df_melted.groupby(['Role', 'Obs_Targ_Identity'])['Emotional_Intensity'].transform(
    lambda x: x.rolling(window=window_size, min_periods=1, center=True).mean()
)
# Get unique stories
unique_stories = df_melted['Story'].unique()
story1 = df_melted[df_melted['Story'] == unique_stories[0]]
story2 = df_melted[df_melted['Story'] == unique_stories[1]]
# get subsample of data
for identity in list(story1["Obs_Targ_Identity"].unique()):
	data = story1[story1["Obs_Targ_Identity"]==identity]
	# Create the line plot with different styles for each Obs_Targ_Identity
	sns.lineplot(x='Time', y='Smoothed_Emotional_Intensity', hue='Role', data=data)
	plt.savefig(f"/home/max/obsolete/BA/own_plots/smoothed/f2f_{identity}_story1.jpg")
	plt.close()
