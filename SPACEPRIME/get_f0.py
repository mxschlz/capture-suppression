import librosa
import slab
import matplotlib.pyplot as plt
import numpy as np


sound = slab.Sound.read("/home/max/PycharmProjects/capture-suppression/SPACEPRIME/digits_all_250ms/1.wav")
y = sound.data.flatten()

f0, voiced_flag, voiced_probs = librosa.pyin(y,
                                             sr=sound.samplerate,
                                             fmin=librosa.note_to_hz('C2'),
                                             fmax=librosa.note_to_hz('C7'))
times = librosa.times_like(f0, sr=sound.samplerate)

D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
fig, ax = plt.subplots()
img = librosa.display.specshow(D, x_axis='time', y_axis='log', ax=ax)
ax.set(title='pYIN fundamental frequency estimation')
fig.colorbar(img, ax=ax, format="%+2.f dB")
ax.plot(times, f0, label='f0', color='cyan', linewidth=3)
ax.legend(loc='upper right')

mean_f0 = f0[2:15].mean()
print("Mean f0 = {:.2f}".format(mean_f0))