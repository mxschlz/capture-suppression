import SPACEPRIME
import mne
mne.viz.set_3d_backend('pyvista')
import numpy as np
from SPACEPRIME.subjects import subject_ids
import matplotlib.pyplot as plt
import mne
from mne.beamformer import apply_lcmv_epochs, make_lcmv
from mne.datasets import fetch_fsaverage, sample


# plotting switch
plot = False
save = False

# Define the baseline period for noise calculation (e.g., -200ms to 0ms)
# This should match the baseline used during epoching.
noise_tmax = 0.0
noise_tmin = -0.5
signal_tmin = 0.0
signal_tmax = 0.25

# define subject
subject_id = 174
# load up epoched data
epochs = mne.read_epochs(f"{SPACEPRIME.get_data_path()}derivatives/epoching/sub-{subject_id}/eeg/sub-{subject_id}_task-spaceprime-epo.fif", preload=True)
epochs.set_eeg_reference('average', projection=True)  # We kind of need this as a projection I guess

epochs = epochs[:500]

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

if plot:
	mne.viz.plot_bem(subject = "",
	                 subjects_dir = fsaverage_dir,
	                 brain_surfaces = "white",
	                 orientation = "sagittal", # other options: coronal, axial
	                 slices = [50, 100, 150, 200])

# Coregistration of EEG channels with the MRI (head transformation)
# This computes the transformation matrix from the EEG sensor
# space to the head space (for proper alignment of the EEG data
# with the anatomical model).

# get head --> MRI transform:
# get fiducials again as a dict
dig = epochs.info['dig']

# --- Verification Step: Check the scale of digitization points ---
# MNE expects meters. Let's confirm the units are not in millimeters.
# We find a fiducial point (e.g., nasion) and check its coordinates.
# A value like ~80.0 would indicate mm, while ~0.08 would indicate meters.
try:
    nasion_point = next(d for d in dig if d['ident'] == mne.io.constants.FIFF.FIFFV_POINT_NASION)
    nasion_coord = nasion_point['r']
    # Calculate distance from origin (0,0,0)
    dist = np.linalg.norm(nasion_coord)
    print(f"Nasion coordinates: {nasion_coord}, Distance from origin: {dist:.4f}")
    assert dist < 1.0, "Digitization points appear to be in mm, not meters! Please scale them."
    print("Digitization points seem to be correctly scaled to meters.")
except (StopIteration, AssertionError) as e:
    print(f"Could not verify digitization point scale or assertion failed: {e}")

fiducials_dict = {}

for d in dig:
	if str(d['ident']) == '2 (FIFFV_POINT_NASION)':
		fiducials_dict['nasion'] = d['r']
	elif str(d['ident']) == '1 (FIFFV_POINT_LPA)':
		fiducials_dict['lpa'] = d['r']
	elif str(d['ident']) == '3 (FIFFV_POINT_RPA)':
		fiducials_dict['rpa'] = d['r']

# coregister standard MRI and our fiducials
coreg = mne.coreg.Coregistration(epochs.info,
                                 subject="",
                                 subjects_dir=fsaverage_dir,
                                 fiducials=fiducials_dict)

# fit using fiducials first
coreg.fit_fiducials(verbose=True)

# refine with ICP (Iterative Closest Point)
coreg.fit_icp(n_iterations=20, nasion_weight=10.0, verbose=True)

# get the final transform
trans = coreg.trans

if plot:
	mne.viz.plot_alignment(
	        epochs.info,  # EEG sensor information from raw data
	        subject = "",  # Subject for the fsaverage brain
	        subjects_dir = fsaverage_dir,  # Path to the fsaverage directory
	        trans = trans,  # Apply the transformation to align sensors with MRI
	        src = src,  # Source space for the brain
	        bem = bem,  # BEM model for the brain
	        show_axes = False,  # Show the axes in the plot
	        dig = True,  # Show digitization points
	        eeg = True) # show EEG sensors


# Compute the forward solution (leadfield)
fwd = mne.make_forward_solution(epochs.info,
                                trans = coreg.trans,
                                src = src,
                                bem = bem,
                                meg = False,
                                eeg = True,
                                mindist = 0.0,
                                n_jobs = -1)

# Compute the noise covariance from the baseline of the epochs
noise_cov = mne.compute_covariance(
    epochs,
    tmax=noise_tmax,  # Use the pre-stimulus baseline
    tmin=noise_tmin,  # Use the pre-stimulus baseline
    method='shrunk',
    rank='auto',  # Let MNE automatically determine the rank
    verbose=True
)

# Plot noise covariance
print("Plotting noise covariance matrix...")
if plot:
    mne.viz.plot_cov(noise_cov, epochs.info, show_svd=False)


# Create the inverse operator with the correct rank
print("Creating inverse operator...")
inverse_operator = mne.minimum_norm.make_inverse_operator(
    epochs.info,
    fwd,
    noise_cov,
    rank=None,  # Use the rank from the noise covariance
    loose=0.2,
    depth=0.8,
    verbose=True
)

# Apply the inverse solution to the epochs
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

print(f"Created {len(stcs)} source-space epochs (trials).")

print(f"Averaging {len(stcs)} source-space trials to create ERP...")
erp_stc = sum(stcs) / len(stcs)

# Plot the final source-space ERP
print("Plotting the source-space ERP...")
brain = erp_stc.plot(
    hemi='both',
    initial_time=None,
    clim=dict(kind="value", lims=[-1.5, 0, 1.5]),
    colormap='seismic',
    transparent=False,
    subjects_dir=fsaverage_dir,
    time_viewer=True,  # This opens the dedicated time course viewer
    backend="pyvistaqt",
    show_traces=True,
    cortex="classic",
    surface="white"
)

if save:
    # The documentation website's movie is generated with:
    brain.save_movie(tmin=-0.1, tmax=0.4, interpolation='linear',
                      time_dilation=20, framerate=5, time_viewer=True)

# ALTERNATIVE. LCMV beamformer
# make data cov
data_cov = mne.compute_covariance(epochs, tmin=0.01, tmax=0.25, method="empirical")
# make noise cov
noise_cov = mne.compute_covariance(epochs, tmin=noise_tmin, tmax=noise_tmax, method="empirical")

filters = make_lcmv(
    epochs.info,
    fwd,
    data_cov,
    reg=0.05,
    noise_cov=noise_cov,
    pick_ori="max-power",
    weight_norm="unit-noise-gain",
    rank=None,
)

# apply filter
stc = apply_lcmv_epochs(epochs, filters)

erp_stcs = sum(stc) / len(stc)

# plot
brain = erp_stcs.plot(
    hemi='both',
    initial_time=None,
    clim=dict(kind="value", lims=[-0.5, 0, 0.5]),
    colormap='seismic',
    transparent=False,
    subjects_dir=fsaverage_dir,
    time_viewer=True,  # This opens the dedicated time course viewer
    backend="pyvistaqt",
    show_traces=True,
    cortex="classic",
    surface="white"
)
if save:
    # The documentation website's movie is generated with:
    brain.save_movie(tmin=-0.1, tmax=0.4, interpolation='linear',
                      time_dilation=20, framerate=5, time_viewer=True)
