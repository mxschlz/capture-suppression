import mne
import matplotlib.pyplot as plt
plt.ion()


# load epochs
epochs = mne.read_epochs("/home/max/data/SPACEPRIME/derivatives/epoching/sub-101/eeg/sub-101_task-spaceprime-epo.fif",
                         preload=True)
# epochs.apply_baseline()
all_conds = list(epochs.event_id.keys())
# separate epochs based on priming conditions
c_epochs = epochs[[x for x in all_conds if "C" in x and not "Singleton-0" in x]]
np_epochs = epochs[[x for x in all_conds if "NP" in x and not "Singleton-0" in x]]
pp_epochs = epochs[[x for x in all_conds if "PP" in x and not "Singleton-0" in x]]
# randomly pick epochs equivalent to the minimum epoch condition count
mne.epochs.equalize_epoch_counts([np_epochs, pp_epochs, c_epochs], method="random")
# plot the conditions
mne.viz.plot_compare_evokeds([c_epochs.average(), np_epochs.average(), pp_epochs.average()], picks="Cz")
