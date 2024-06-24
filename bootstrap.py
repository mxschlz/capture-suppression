import pandas as pd
import os


results_path = '/home/max/data/behavior/wp1/'
df = pd.read_excel(os.path.join(results_path, 'results_June_21_2024_15_43_27.xlsx'), index_col=0)

# Get unique subject IDs
subject_ids = df['subject_id'].unique()

# Get max subject ID
max_subject_id = max(subject_ids)


# Define a function to generate simulated subjects
def generate_simulated_subjects(df, num_new_subjects, max_subject_id):
    # Create new subject IDs
    new_subject_ids = [max_subject_id + i + 1 for i in range(num_new_subjects)]

    # Create a list to store the simulated dataframes
    simulated_dfs = []

    # Generate simulated data for each new subject
    for subject_id in new_subject_ids:
        # Sample rows with replacement
        simulated_data = df.sample(n=len(df[df.subject_id==max_subject_id]), replace=True)

        # Reset index
        simulated_data = simulated_data.reset_index(drop=True)

        # Assign new subject ID
        simulated_data['subject_id'] = subject_id

        # Append to the list
        simulated_dfs.append(simulated_data)

    # Concatenate the simulated dataframes
    simulated_df = pd.concat(simulated_dfs, ignore_index=True)

    return simulated_df


# Generate 2 new subjects
simulated_df = generate_simulated_subjects(df, 4, max_subject_id)

appended = pd.concat([df, simulated_df])
appended["bootstrapped"] = 0
appended.bootstrapped[appended.subject_id > 99] = 1
# Print unique subject IDs in the simulated data
print(simulated_df['subject_id'].unique())

appended.to_excel(os.path.join(results_path, 'simulated_subjects_appended.xlsx'), index=False)