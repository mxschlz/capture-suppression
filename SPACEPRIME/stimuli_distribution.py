import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from SPACEPRIME.plotting import plot_individual_lines
import glob
import os
plt.ion()


# define data root dir
data_root = "/home/max/Insync/schulz.max5@gmail.com/GoogleDrive/PhD/data/SPACEPRIME/derivatives/preprocessing/"
# get all the subject ids
subjects = os.listdir(data_root)
# load data from children
df = pd.concat([pd.read_csv(glob.glob(f"/home/max/Insync/schulz.max5@gmail.com/GoogleDrive/PhD/data/SPACEPRIME/derivatives/preprocessing/{subject}/beh/{subject}_clean*.csv")[0]) for subject in subjects if int(subject.split("-")[1]) in [105, 106, 107]])
# Assuming your dataframe is named 'df'
response_counts = df['response'].value_counts()
# plot bar plot
response_counts.plot(kind='bar')
plt.xlabel("Response")
plt.ylabel("Frequency")
plt.title("Distribution of Responses")

target_counts = df["TargetDigit"].value_counts()
# plot bar plot
target_counts.plot(kind='bar')
plt.xlabel("Target")
plt.ylabel("Frequency")
plt.title("Distribution of Target Digit")

distractor_counts = df["SingletonDigit"].value_counts()
# plot bar plot
distractor_counts.plot(kind='bar')
plt.xlabel("Distractor")
plt.ylabel("Frequency")
plt.title("Distribution of Distractor Digit")

distractor_counts = df["Non-Singleton1Digit"].value_counts()
# plot bar plot
distractor_counts.plot(kind='bar')
plt.xlabel("Distractor")
plt.ylabel("Frequency")
plt.title("Distribution of Distractor Digit")

distractor_counts = df["Non-Singleton2Digit"].value_counts()
# plot bar plot
distractor_counts.plot(kind='bar')
plt.xlabel("Distractor")
plt.ylabel("Frequency")
plt.title("Distribution of Distractor Digit")
