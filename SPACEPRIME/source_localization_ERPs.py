import mne
import SPACEPRIME


# load up raw data
raw = mne.io.read_raw_fif("/home/maxschulz/IPSY1-Storage/Projects/ac/Experiments/running_studies/SPACEPRIME/derivatives/preprocessing/sub-174/eeg/sub-174_task-spaceprime_raw.fif")
montage = raw.get_montage()

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

# plot the alignment between MRI and our
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
    eeg=True)  # show EEG sensors

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

# I only want to analyse the data around dual-task block onsets: -10s–0 before and 0-10s after. If you have more data, that’s even better.
block_annotations = ['BL_m_on', '1back_d_m_on', '2back_d_m_on', 'vtask_main_on']

# get all annotations that are in block_annotations:
filtered_annotations = [annot for annot in annotations_list if annot['description'] in block_annotations]

# now cut data into blocks by using -10s to block onset (for estimating noise cov)
# and block onset to +10s (for estimating data cov)

for annot_idx, annot in enumerate(filtered_annotations):
    print(annot_idx)
    curr_onset_ts = annot["onset"]

    # cut block:
    curr_block_data = raw.copy().crop(tmin=curr_onset_ts,
                                      tmax=curr_onset_ts + 10)

    curr_block_noise = raw.copy().crop(tmin=curr_onset_ts - 10,
                                       tmax=curr_onset_ts)

    # append data to lists:
    if annot_idx == 0:
        data_covnoise = curr_block_noise.copy()
        data_covdata = curr_block_data.copy()
    else:
        data_covnoise = mne.concatenate_raws([data_covnoise, curr_block_noise])
        data_covdata = mne.concatenate_raws([data_covdata, curr_block_data])

# Now compute data chunk and noise chunk covariances:
noise_cov = mne.compute_raw_covariance(data_covnoise,
                                       method='auto',
                                       rank={'eeg': raw_source_space.info["nchan"] - 1 - len(excl_components)},
                                       tmin=0,
                                       tmax=None)
data_cov = mne.compute_raw_covariance(data_covdata,
                                      method='auto',
                                      rank={'eeg': raw_source_space.info["nchan"] - 1 - len(excl_components)},
                                      tmin=0,
                                      tmax=None)
