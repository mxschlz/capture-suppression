import pandas as pd
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
plt.ion()


def degrees_va_to_pixels(degrees, screen_pixels, screen_size_cm, viewing_distance_cm):
    """
    Converts degrees visual angle to pixels.

    Args:
        degrees: The visual angle in degrees.
        screen_pixels: The number of pixels on the screen (horizontal or vertical).
        screen_size_cm: The physical size of the screen in centimeters (width or height).
        viewing_distance_cm: The viewing distance in centimeters.

    Returns:
        The number of pixels corresponding to the given visual angle.
    """

    pixels = degrees * (screen_pixels / screen_size_cm) * (viewing_distance_cm * np.tan(np.radians(1)))
    return pixels

# define data root dir
data_root = "/home/max/Insync/schulz.max5@gmail.com/GoogleDrive/PhD/data/SPACEPRIME/derivatives/preprocessing/"
# get all the subject ids
subjects = os.listdir(data_root)
# load data from children
df = pd.concat([pd.read_csv(glob.glob(f"/home/max/Insync/schulz.max5@gmail.com/GoogleDrive/PhD/data/SPACEPRIME/sourcedata/raw/{subject}/beh/{subject}*mouse_data.csv")[0]) for subject in subjects if int(subject.split("-")[1]) in [105, 106, 107]])
# define some setup params
width = 1920
height = 1080
dg_va = 2
viewing_distance_cm = 70
x = df["x"] * degrees_va_to_pixels(degrees=dg_va, screen_pixels=width, screen_size_cm=40,
                                   viewing_distance_cm=viewing_distance_cm)
y = df["y"] * degrees_va_to_pixels(degrees=dg_va, screen_pixels=height, screen_size_cm=30,
                                   viewing_distance_cm=viewing_distance_cm)
data = np.array([x, y]).transpose()
# define height and width of the screen
# get the canvas
canvas = np.vstack((data[:, 0], data[:, 1]))  # shape (2, n_samples)
_range = [[0, height], [0, width]]
bins_x, bins_y = width, height
extent = [0, width, height, 0]
hist, _, _ = np.histogram2d(canvas[1, :], canvas[0, :], bins=(bins_y, bins_x), range=_range)
sfreq = 1600
sigma = 25
alpha = 1.0
hist /= sfreq
# Calculate the center of the screen
center_x = width / 2
center_y = height / 2
# Calculate the center of your data (you might want to adjust this based on your specific needs)
data_center_x = 0  # Or use a fixed value if you know the center
data_center_y = 0
# Shift the data so that the center of the data is at the center of the screen
x_shifted = x - data_center_x + center_x
y_shifted = y - data_center_y + center_y
# Recalculate the histogram with the shifted data
hist, _, _ = np.histogram2d(y_shifted, x_shifted, bins=(height, width), range=[[0, height], [0, width]]) # Note the change here as well
hist = gaussian_filter(hist, sigma=sigma)
extent = [0, width, height, 0]
plt.imshow(hist, extent=extent, origin='upper', aspect='auto', alpha=alpha)
plt.gca().invert_yaxis()  # Invert the y-axis
plt.xlabel("X Position (pixels)")
plt.ylabel("Y Position (pixels)")
plt.title("Cursor dwell time [s]")
#plt.colorbar(label="Density")
