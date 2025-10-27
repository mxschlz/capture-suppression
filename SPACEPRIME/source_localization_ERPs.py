import SPACEPRIME
import mne
import os
mne.viz.set_3d_backend('pyvista')


# plotting switch
plot = False

# define subject
subject_id = 174
# load up epoched data
epochs = mne.read_epochs(f"{SPACEPRIME.get_data_path()}derivatives/epoching/sub-{subject_id}/eeg/sub-{subject_id}_task-spaceprime-epo.fif", preload=True)
epochs.set_eeg_reference('average', projection=True)  # We kind of need this as a projection I guess

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
                                mindist = 5.0,
                                n_jobs = -1)

# --- Compute noise covariance from the baseline of the epochs ---
# Define the baseline period for noise calculation (e.g., -200ms to 0ms)
# This should match the baseline used during epoching.
noise_tmax = 0
noise_tmin = -0.5

# --- Load excluded ICA components from file for accurate rank calculation ---
ica_labels_path = (
    f"{SPACEPRIME.get_data_path()}derivatives/preprocessing/sub-{subject_id}/eeg/"
    f"sub-{subject_id}_task-spaceprime_ica_labels.txt"
)

excl_components = []
try:
    with open(ica_labels_path, 'r') as f:
        excl_components = [int(line.strip()) for line in f if line.strip()]
    print(f"Successfully loaded {len(excl_components)} excluded ICA components from file: {excl_components}")
except FileNotFoundError:
    print(f"Warning: ICA labels file not found at '{ica_labels_path}'.\n"
          f"Assuming no ICA components were excluded.")
except Exception as e:
    print(f"An error occurred while reading the ICA labels file: {e}")

# Calculate the rank for the EEG data
n_eeg_channels = len(mne.pick_types(epochs.info, eeg=True, meg=False))
rank_eeg = n_eeg_channels - 1 - len(excl_components)

# Compute the noise covariance from the baseline of the epochs
noise_cov = mne.compute_covariance(
    epochs,
    tmax=noise_tmax,  # Use the pre-stimulus baseline
    method='shrunk',
    rank={'eeg': rank_eeg},
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
    rank={'eeg': rank_eeg},
    loose=0.2,
    depth=0.8,
    verbose=True
)

# Apply the inverse solution to the epochs
lambda2 = 1.0 / 3.0 ** 2
method = "dSPM"

# TODO: reduce epoch count to prevent RAM exhaustion
epochs = epochs[:500]

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
brain = erp_stc.plot(hemi = 'lh',
		             initial_time = 0.1,
		             #clim = dict(kind="value", lims = [-7, 0, 7]),
		             clim = dict(kind="value", lims = [-3, 0, 3]),
		             colormap = 'seismic', transparent = False,
		             subjects_dir = fsaverage_dir,
                     backend="matplotlib")
