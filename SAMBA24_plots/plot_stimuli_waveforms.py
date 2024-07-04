import slab
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from color_palette import get_subpalette
from color_palette import palette
plt.ion()

# insert color palette
sns.set_palette(list(get_subpalette([104, 22, 54]).values()))

soundroot = "/home/max/Musik/"

# get sounds
stim1 = slab.Sound.read(soundroot + "1.wav")
stim2 = slab.Sound.read(soundroot + "9.wav")
stim3 = slab.Sound.read(soundroot + "5.wav")

# plot nonsingleton distractor
plt.plot(stim1, color=palette[104])
plt.savefig("/home/max/figures/SAMBA24/nonsingleton.svg")  # 1
plt.close()
# plot singleton distractor
plt.plot(stim2, color=palette[22])
plt.savefig("/home/max/figures/SAMBA24/singleton.svg")  # 9
plt.close()
# amplitude modulation for target sound
y = stim2.data[:, 0]  # Extract data from the left channel (assuming monaural)
sr = stim2.samplerate
# Calculate duration and time array
duration = stim2.duration
t = np.linspace(0, duration, len(y), endpoint=False)
# Generate carrier sine wave
carrier = np.sin(2 * np.pi * 30 * t)
# Perform amplitude modulation
y_modulated = y * carrier
soundmod = slab.Sound(data=y_modulated, samplerate=sr)
# plot target
plt.plot(soundmod, color=palette[54])
plt.savefig("/home/max/figures/SAMBA24/target.svg")  # 5
plt.close()
