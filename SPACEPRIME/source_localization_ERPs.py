import SPACEPRIME
import mne
import numpy as np
# Try forcing MNE to use the pyvista backend explicitly
mne.viz.set_3d_backend('pyvista')
from SPACEPRIME.encoding import *


# define subject
subject_id = 174
# load up raw data
raw = mne.io.read_raw_fif(f"{SPACEPRIME.get_data_path()}derivatives/preprocessing/sub-{subject_id}/eeg/sub-{subject_id}_task-spaceprime_raw.fif")
montage = raw.get_montage()

# The inverse modeling pipeline requires an average reference. We add it as a
# projector, which is the standard and recommended MNE workflow.
print("Applying average EEG reference as a projector...")
raw.set_eeg_reference('average', projection=True)

# get standard fMRI head model
fsaverage_dir = mne.datasets.fetch_fsaverage(verbose = True)

# Set up the source space
# This defines the cortical grid.
src = mne.setup_source_space(subject = "",
                             spacing = "ico3", # → 1,284 total sources
                             subjects_dir = fsaverage_dir,
                             add_dist = False)

# Create boundary element model (BEM) with 3 layers: scalp, skull and brain tissue
model = mne.make_bem_model(subject="",
                           ico=4,  # icosahedral tessellation level (4 is medium)
                           conductivity=(0.3, 0.006, 0.3),  # scalp, skull, brain
                           subjects_dir=fsaverage_dir)

# Make the BEM solution based on the model
# This creates a solution to be used in forward calculations.
bem = mne.make_bem_solution(model)

"""# Plot the BEM brain model
mne.viz.plot_bem(subject="",
                 subjects_dir=fsaverage_dir,
                 brain_surfaces="white",
                 orientation="sagittal",  # other options: coronal, axial
                 slices=[50, 100, 150, 200])"""

# Coregistration of EEG channels with the MRI (head transformation)
# This computes the transformation matrix from the EEG sensor
# space to the head space (for proper alignment of the EEG data
# with the anatomical model).

# get head --> MRI transform:
# get fiducials again as a dict
dig = raw.info['dig']
fiducials_dict = {}

for d in dig:
    if str(d['ident']) == '2 (FIFFV_POINT_NASION)':
        fiducials_dict['nasion'] = d['r']
    elif str(d['ident']) == '1 (FIFFV_POINT_LPA)':
        fiducials_dict['lpa'] = d['r']
    elif str(d['ident']) == '3 (FIFFV_POINT_RPA)':
        fiducials_dict['rpa'] = d['r']

# coregister standard MRI and our fiducials
coreg = mne.coreg.Coregistration(raw.info,
                                 subject="",
                                 subjects_dir=fsaverage_dir,
                                 fiducials=fiducials_dict)

# fit using fiducials first
coreg.fit_fiducials(verbose=True)

# refine with ICP (Iterative Closest Point)
coreg.fit_icp(n_iterations=20, nasion_weight=10.0, verbose=True)

# get the final transform
trans = coreg.trans

"""# plot the alignment between MRI and our
# head shape points / fiducials / electrode positions
mne.viz.plot_alignment(
    raw.info,  # EEG sensor information from raw data
    subject="",  # Subject for the fsaverage brain
    subjects_dir=fsaverage_dir,  # Path to the fsaverage directory
    trans=trans,  # Apply the transformation to align sensors with MRI
    src=src,  # Source space for the brain
    bem=bem,  # BEM model for the brain
    show_axes=False,  # Show the axes in the plot
    dig=True,  # Show digitization points
    eeg=True)  # show EEG sensors"""

# Compute the forward solution (leadfield)
fwd = mne.make_forward_solution(raw.info,
                                trans = coreg.trans,
                                src = src,
                                bem = bem,
                                meg = False,
                                eeg = True,
                                mindist = 5.0,
                                n_jobs = 10)

# Compute noise and data covariances
# Important: You need the number of excluded ICA components (& maybe bad EEG channels) for this.
# The excluded components should be stored in a list called excl_components.
# get all annotations:
annotations_list = [{'onset': annot['onset'], 'description': annot['description']} for annot in raw.annotations]

# 1. Define the annotations of interest
# These are the markers that will be used to find our time window.
calib_annotations = ['Stimulus/S  2', 'Stimulus/S  3']

# 2. Find all occurrences of these annotations in the raw data
all_annotations = raw.annotations
filtered_annotations = [
    annot for annot in all_annotations
    if annot['description'] in calib_annotations
]

# 3. IMPORTANT: Sort the found annotations by time to ensure chronological order
filtered_annotations.sort(key=lambda x: x['onset'])

print(f"Found {len(filtered_annotations)} annotations matching {calib_annotations}.")

# 4. Ensure we found at least two markers to define an interval
if len(filtered_annotations) < 2:
    raise ValueError(
        f"Could not find at least two markers from '{calib_annotations}' "
        "to define a time interval. Only found "
        f"{len(filtered_annotations)}."
    )

# 5. Get the onset times for the first and second marker from the sorted list.
# Even with 20 markers, this correctly selects the first two in time.
t_start_noise = filtered_annotations[0]['onset']
t_end_noise = filtered_annotations[1]['onset']

print(f"Defining noise period between the first two markers: "
      f"from {t_start_noise:.2f}s to {t_end_noise:.2f}s.")

# 6. Create a new raw object containing only the data between these two markers
noise_segment = raw.copy().crop(tmin=t_start_noise, tmax=t_end_noise)

# 7. Compute the noise covariance from this specific segment
# --- Load excluded ICA components from file for accurate rank calculation ---
# Define the path to the file containing the indices of ICA components marked for exclusion.
ica_labels_path = (
    f"{SPACEPRIME.get_data_path()}derivatives/preprocessing/sub-{subject_id}/eeg/"
    f"sub-{subject_id}_task-spaceprime_ica_labels.txt"
)

excl_components = []
try:
    with open(ica_labels_path, 'r') as f:
        # Read each line, strip whitespace, and convert to an integer.
        # This assumes the file contains one component index per line.
        excl_components = [int(line.strip()) for line in f if line.strip()]
    print(f"Successfully loaded {len(excl_components)} excluded ICA components from file: {excl_components}")
except FileNotFoundError:
    print(f"Warning: ICA labels file not found at '{ica_labels_path}'.\n"
          f"Assuming no ICA components were excluded.")
except Exception as e:
    print(f"An error occurred while reading the ICA labels file: {e}")

# Calculate the rank for the EEG data. Rank is the number of independent signals.
# It's typically: (num EEG channels) - 1 (for the reference) - (num removed ICs)
# Note: I've corrected the error from your original script which used a
# non-existent 'raw_source_space' variable.
n_eeg_channels = len(mne.pick_types(raw.info, eeg=True, meg=False))
rank_eeg = n_eeg_channels - 1 - len(excl_components)

# Now compute the noise covariance from the selected data segment
noise_cov = mne.compute_raw_covariance(
    noise_segment,
    method='shrunk',  # 'shrunk' is a robust and recommended method
    rank={'eeg': rank_eeg},
    tmin=None,        # Use the entire cropped segment
    tmax=None,
    verbose=True
)

print("\nSuccessfully computed noise covariance matrix:")
print(noise_cov)

# crop data to a short segment
crop_min = 15
crop_max = 30
raw_source_space_cropped = raw.copy().crop(tmin = crop_min * 60,
                                           tmax = crop_max * 60)


# 1. Create the inverse operator with the correct rank
#    We use the 'rank_eeg' variable calculated earlier for consistency and accuracy.
print("Creating inverse operator...")
inverse_operator = mne.minimum_norm.make_inverse_operator(
    raw.info,                # Use the info from the full raw data
    fwd,
    noise_cov,
    rank={'eeg': rank_eeg},  # Use the correctly calculated rank
    loose=0.2,
    depth=0.8,
    verbose=True
)

# 2. Define events and epoching parameters
#    This is the standard MNE approach for ERP analysis.
tmin, tmax = -0.2, 0.7  # Pre- and post-stimulus time in seconds

# 3. Get events from annotations in the raw data
events, _ = mne.events_from_annotations(raw)
print(f"Found {len(events)} events of interest.")

# 4. Create sensor-space epochs
#    We use baseline correction on the sensor data before source localization.
baseline = None
epochs = mne.Epochs(
    raw,
    events,
    event_id=encoding_sub_106,
    tmin=tmin,
    tmax=tmax,
    proj=True,          # Apply the average reference projector
    baseline=baseline,
    preload=True,       # Preload data for processing
    verbose=True
)

# 5. Compute the inverse solution on the epochs
#    This is more memory-efficient than applying to raw data.
lambda2 = 1.0 / 3.0 ** 2
method = "dSPM"

stcs = mne.minimum_norm.apply_inverse_epochs(
    epochs,
    inverse_operator,
    lambda2=lambda2,
    method=method,
    pick_ori="normal",
    verbose=True
)

# 'stcs' is a list of SourceEstimate objects, one for each epoch.
print(f"Created {len(stcs)} source-space epochs (trials).")

# 6. Average the source-space epochs to get the ERP
#    This averages the list of STCs into a single STC object.
#
#    The function `mne.grand_average` is for averaging sensor-space Evoked objects
#    across subjects. To average a list of SourceEstimate objects (trials),
#    we can simply use standard arithmetic.
if not stcs:
    raise RuntimeError("No source-space epochs were created. Cannot compute ERP.")

print(f"Averaging {len(stcs)} source-space trials to create ERP...")
erp_stc = sum(stcs) / len(stcs)

# 7. Plot the final source-space ERP
print("Plotting the source-space ERP...")
erp_stc.plot(
    hemi='both',
    subjects_dir=fsaverage_dir,
    initial_time=0.1,  # Example: view activity at 100ms
    clim=dict(kind="value", lims=[-3, 0, 3]),
    colormap='seismic',
    transparent=False,
    time_viewer=True   # You can still use the time viewer on the final ERP
)