import pandas as pd
import os


data_root = "/home/max/Insync/schulz.max5@gmail.com/GoogleDrive/PhD/data/SPACEPRIME/derivatives/preprocessing/"
subjects = os.listdir(data_root)

data = [pd.read_csv(f"/home/max/Insync/schulz.max5@gmail.com/GoogleDrive/PhD/data/SPACEPRIME/derivatives/preprocessing/{subject}/beh/{subject}_clean.csv") for subject in subjects]
df = pd.concat(data)