import mne
from SPACEPRIME import get_data_path


""" Check If We Have EEG Electrodes That Formed Bridges Due to Too Much Gel """
    
# If we find some, there's not a lot we can do, but maybe we can exlude them as bad channels 
# or interpolate them.
    
# first check electrical distance metrics to estimate electrode bridging
# "The electrical distance is just the variance of signals subtracted pairwise. 
# Channels with activity that mirror another channel nearly exactly will have 
# very low electrical distance." - https://mne.tools/stable/auto_examples/preprocessing/eeg_bridging.html
    
# It's sufficient to use about 3 min of data to detect bridging, but we need a segment from the end of the 
# recording where the gel is already set.
    
subject_id = 120
raw = mne.io.read_raw_fif(f"{get_data_path()}derivatives/preprocessing/sub-{subject_id}/eeg/sub-{subject_id}_task-spaceprime_raw.fif",
                          preload=True)
# get the duration of the recording in seconds
total_duration = raw.times[-1]

# compute the start time stamp for the last 3 min
start_time_last_3_min = total_duration - 180  # 180 s = 3 min

# get the last 3 minutes of the recording
raw_bridges = raw.copy().crop(tmin = start_time_last_3_min, tmax = total_duration)
    
# check which electrodes have a low electrical distance (basically super high correlation)
ed_data = mne.preprocessing.compute_bridged_electrodes(raw_bridges)

# plot potentially bridged electrodes in yellow on topoplot:
bridged_idx, ed_matrix = ed_data        
mne.viz.plot_bridged_electrodes(raw_bridges.info,
                                bridged_idx,
                                ed_matrix,
                                title = "Bridged Electrodes of Participant XYZ",
                                topomap_args = dict(vlim = (None, 5)),
                                )
    
""" Interpolate Bridged Electrodes or Exclude Data of Current Participant """
# If there are bridges, but not too many (let's say 3, 
# that would be 6 affected electrodes at most), interpolate them. 
# This cutoff is completely deliberate, you can set a cutoff that suits you. 
# Maybe also check if the electrode bridges are in your ROI(s) or if they affect your analysis before excluding a participant.

if len(bridged_idx) > 0 and len(bridged_idx) <= 3:
    print("interpolating " + str(len(bridged_idx)) + " bridged electrodes!" )
                
    # Interpolate bridged channels
    raw = mne.preprocessing.interpolate_bridged_electrodes(raw, bridged_idx = bridged_idx)
        
    # In my script, I also save information on which & how many electrodes were affected here.
    
# if there are more than 3 bridges, exclude participant from further analysis and go to next one:
elif len(bridged_idx) > 3:
    print("Detected " + str(len(bridged_idx)) +  " electrode bridges for current participant! Excluding dataset from the analysis!")
        
    # In my script, I also save information on which & how many electrodes were affected here.
    
    # now you could skip this participant like this if you're using this snippet in a loop over participants:
    # next
        
# if there are no bridges, just keep the df as is
elif len(bridged_idx) == 0:
    print("no electrode bridging detected here :-)" )        



