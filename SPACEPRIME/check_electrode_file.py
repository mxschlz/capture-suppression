import mne
import matplotlib.pyplot as plt
plt.ion()

# Read the .bvef file
actichamp = mne.channels.read_custom_montage('/home/max/Downloads/actiCap_snap_CACS_CAS_GACS/actiCap_slim_for actiChamp_Plus/CACS-64/CACS-64_NO_REF.bvef')
brainamp_dc = mne.channels.read_custom_montage('/home/max/Downloads/actiCap_snap_CACS_CAS_GACS/actiCap_slim_for BrainAmpDC/CACS-64/CACS-64_REF.bvef')
custom_montage = mne.channels.read_custom_montage("/home/max/Insync/schulz.max5@gmail.com/GoogleDrive/PhD/data/SPACEPRIME/settings/CACS-64_NO_REF.bvef")

fig, ax = plt.subplots(1, 3)
actichamp.plot(kind='topomap', axes=ax[0]) # For a 2D topomap visualization
brainamp_dc.plot(kind='topomap', axes=ax[1])
custom_montage.plot(kind='topomap', axes=ax[2])