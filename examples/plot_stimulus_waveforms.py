import slab
from matplotlib import pyplot as plt


target = slab.Sound.read("C:\\Users\Max\PycharmProjects\psychopy-experiments\SPACEPRIME\stimuli\\targets_low_30_Hz\\8_amplitude_modulated_30.wav")
target.resample(8000).waveform()
plt.savefig("G:\\Meine Ablage\\PhD\\Conferences\\ICON25\\Poster\\target_waveform.svg")
distractor = slab.Sound.read("C:\\Users\Max\PycharmProjects\psychopy-experiments\SPACEPRIME\stimuli\\distractors_high\\2_high_pitched_factor_10.wav")
distractor.resample(8000).waveform()
plt.savefig("G:\\Meine Ablage\\PhD\\Conferences\\ICON25\\Poster\\distractor_waveform.svg")
control = slab.Sound.read("C:\\Users\Max\PycharmProjects\psychopy-experiments\SPACEPRIME\stimuli\\digits_all_250ms\\1.wav")
control.resample(8000).waveform()
plt.savefig("G:\\Meine Ablage\\PhD\\Conferences\\ICON25\\Poster\\control_waveform.svg")