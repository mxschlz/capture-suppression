import pandas as pd


# get headgaze data
# Load the CSV file into a DataFrame
df = pd.read_csv('/home/max/PycharmProjects/head_gaze_tracking/test/sub-99_eye_tracking_log_2024-11-19-17-10-59.csv')
# line plot the data
df.plot(kind="line", x="Frame Nr", y=["Pitch", "Yaw"], colormap="Set1")