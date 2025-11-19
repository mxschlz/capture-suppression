import SPACEPRIME
import mne
mne.viz.set_3d_backend('pyvista')
from SPACEPRIME.subjects import subject_ids
import mne
import os


# plotting switch
plot = False
save = False

# define paradigm
paradigm = "spaceprime"

# Define the baseline period for noise calculation (e.g., -200ms to 0ms)
# This should match the baseline used during epoching.
noise_tmax = -0.1
noise_tmin = -0.3
signal_tmin = 0.0
signal_tmax = 0.25

# get standard fMRI head model
subjects_dir = mne.datasets.fetch_fsaverage(verbose=True)
# The fetcher might return a path ending in 'fsaverage', which can cause
# a double-folder issue (e.g., '.../fsaverage/fsaverage').
# We ensure we have the correct parent subjects directory.
if subjects_dir.name == "fsaverage":
    subjects_dir = subjects_dir.parent

# --- Create a directory for saving source estimates ---
stc_dir = os.path.join(SPACEPRIME.get_data_path(), "derivatives", "source_localization_dSPM")
os.makedirs(stc_dir, exist_ok=True)

# --- Set up a whole-brain source space ---
# For investigating attention networks, a uniform grid across the whole cortex is
# preferable to one constrained to a specific sensory area.
# 'ico4' provides a good balance of resolution and computational cost for EEG.
src = mne.setup_source_space(subject = "fsaverage",
                             spacing = "oct5", # → 1026 sources/hemi (2052 total)
                             subjects_dir = subjects_dir,
                             add_dist = False)
if plot:
    print("Plotting source space...")
    src.plot(subjects_dir=subjects_dir)

# --- Save the source space ---
src_dir = os.path.join(SPACEPRIME.get_data_path(), "derivatives")
os.makedirs(src_dir, exist_ok=True)
src_fname = os.path.join(src_dir, "fsaverage-oct5-src.fif")
mne.write_source_spaces(src_fname, src, overwrite=True)


# Create boundary element model (BEM) with 3 layers: scalp, skull and brain tissue
model = mne.make_bem_model(subject="fsaverage",
                           ico=4,  # icosahedral tessellation level (4 is medium)
                           conductivity=(0.3, 0.006, 0.3),  # scalp, skull, brain
                           subjects_dir=subjects_dir)

# Make the BEM solution based on the model
# This creates a solution to be used in forward calculations.
bem = mne.make_bem_solution(model)

if plot:
    mne.viz.plot_bem(subject="fsaverage",
                     subjects_dir=subjects_dir,
                     brain_surfaces="white",
                     orientation="sagittal",
                     slices=[50, 100, 150, 200])

# --- Loop through subjects for multi-subject analysis ---
processed_subjects = []

for subject_id in subject_ids:
    print(f"\n--- Processing Subject: {subject_id} ---")

    try:
        # Load epoched data
        epochs_path = f"{SPACEPRIME.get_data_path()}derivatives/epoching/sub-{subject_id}/eeg/sub-{subject_id}_task-{paradigm}-epo.fif"
        if not os.path.exists(epochs_path):
            print(f"Epochs file not found for subject {subject_id}. Skipping.")
            continue

        epochs = mne.read_epochs(epochs_path, preload=True)
        epochs.set_eeg_reference('average', projection=True)

        # Average epochs to create evoked data before applying inverse solution
        evoked = epochs.average()

        if plot:
            print("\n--- Plotting Sensor-Level Evoked Potentials (ERPs) ---")
            evoked.plot_joint()

        # use standard montage (the existing montage of the epochs is wrong)
        # Using a specific montage like 'easycap-M1' is more accurate if it matches
        # the recording hardware. We use it here for consistency with other scripts.
        montage = mne.channels.make_standard_montage("easycap-M1")
        epochs.set_montage(montage)

        # Coregistration
        dig = epochs.info['dig']
        try:
            fiducials_dict = {
                'nasion': next(d['r'] for d in dig if str(d['ident']) == '2 (FIFFV_POINT_NASION)'),
                'lpa': next(d['r'] for d in dig if str(d['ident']) == '1 (FIFFV_POINT_LPA)'),
                'rpa': next(d['r'] for d in dig if str(d['ident']) == '3 (FIFFV_POINT_RPA)'),
            }
        except StopIteration:
            print(f"Fiducials not found for subject {subject_id}. Skipping.")
            continue
            
        coreg = mne.coreg.Coregistration(epochs.info, "fsaverage", subjects_dir, fiducials=fiducials_dict)
        coreg.fit_fiducials(verbose=False)
        coreg.fit_icp(n_iterations=20, nasion_weight=10.0, verbose=False)
        trans = coreg.trans

        if plot:
            mne.viz.plot_alignment(
                epochs.info,
                subject="fsaverage",
                subjects_dir=subjects_dir,
                trans=trans,
                src=src,
                bem=bem,
                show_axes=True,
                dig=True,  # Only plot EEG channels from dig, not fiducials
                eeg=["original", "projected"]
            )

        # Forward solution
        # The 'mindist' parameter excludes sources that are too close to the inner skull boundary.
        # A value of 0.0 (no exclusion) can lead to spurious, high-amplitude activity in deep medial
        # wall sources that are unrealistically close to the skull model.
        # Setting this to 5mm is a standard practice to create a more robust forward model.
        fwd = mne.make_forward_solution(epochs.info, trans=trans, src=src, bem=bem, meg=False, eeg=True, mindist=5.0, n_jobs=-1)
        print(f"Forward solution created. Kept {fwd['nsource']} of {fwd['src'][0]['nuse'] + fwd['src'][1]['nuse']} sources.")

        # --- Create a directory for saving forward and inverse solutions ---
        solutions_dir = os.path.join(stc_dir, f"sub-{subject_id}")
        os.makedirs(solutions_dir, exist_ok=True)

        # Save forward solution
        fwd_fname = os.path.join(solutions_dir, f"sub-{subject_id}_task-{paradigm}-fwd.fif")
        mne.write_forward_solution(fwd_fname, fwd, overwrite=True)
        print(f"Saved forward solution for subject {subject_id} to {fwd_fname}")

        # Noise covariance
        # Using a regularized method like 'ledoit_wolf' is more robust than 'empirical',
        # especially when the amount of baseline data is limited. It provides a more
        # stable estimate of the noise by preventing overfitting.
        noise_cov = mne.compute_covariance(epochs, tmax=noise_tmax, tmin=noise_tmin, method='shrunk', rank=None, verbose=False)

        if plot:
            print("Plotting noise covariance matrix...")
            mne.viz.plot_cov(noise_cov, epochs.info, show_svd=False)

        # Inverse operator
        # The depth parameter compensates for the bias of MNE solutions towards superficial sources.
        # The default of 0.8 is optimized for MEG. For EEG, a value between 2.0 and 5.0 is recommended
        # due to the stronger smearing effect of the skull. We'll use 3.0 as a good starting point.
        depth_weighting = 0.0
        inverse_operator = mne.minimum_norm.make_inverse_operator(epochs.info, fwd, noise_cov, rank=None, loose="auto", depth=depth_weighting, verbose=False)
        
        # Save inverse operator
        inv_fname = os.path.join(solutions_dir, f"sub-{subject_id}_task-{paradigm}-inv.fif")
        mne.minimum_norm.write_inverse_operator(inv_fname, inverse_operator, overwrite=True)
        print(f"Saved inverse operator for subject {subject_id} to {inv_fname}")

        # --- Apply inverse solution with data-driven regularization ---
        # Instead of using a fixed SNR (e.g., 3.0), we can estimate it from the data.
        # This makes the regularization parameter lambda2 adaptive to each subject's data quality.
        snr = 3
        lambda2 = 1.0 / snr ** 2
        print(f"  Estimated SNR: {snr:.2f} -> lambda2: {lambda2:.2f}")
        method = "dSPM"
        erp_stc = mne.minimum_norm.apply_inverse(evoked, inverse_operator, lambda2=lambda2, method=method, pick_ori="normal", verbose=False)

        if plot:
            print(f"Plotting source ERP for subject {subject_id}...")
            brain = erp_stc.plot(
                hemi='both',
                initial_time=0.1,
                clim="auto",
                colormap='seismic',
                transparent=False,
                time_unit='s',
                subjects_dir=subjects_dir,
                time_viewer=True,
                backend="pyvistaqt",
                show_traces=True,
                cortex="classic",
                surface="inflated",
                smoothing_steps=10
            )

        # Save the individual subject's source ERP
        stc_filename = os.path.join(solutions_dir, f"sub-{subject_id}_task-{paradigm}_desc-dSPM-erp")
        erp_stc.save(stc_filename, overwrite=True)
        print(f"Saved source ERP for subject {subject_id} to {stc_filename}-stc.h5")

        processed_subjects.append(subject_id)

    except Exception as e:
        print(f"An error occurred while processing subject {subject_id}: {e}")

    finally:
        # RAM-friendly: delete large objects
        if 'epochs' in locals(): del epochs
        if 'coreg' in locals(): del coreg
        if 'trans' in locals(): del trans
        if 'fwd' in locals(): del fwd
        if 'noise_cov' in locals(): del noise_cov
        if 'inverse_operator' in locals(): del inverse_operator
        if 'erp_stc' in locals(): del erp_stc

# --- Grand-average source ERP ---
print("\n--- Computing Grand-Average Source ERP ---")
stcs_to_average = []
for subject_id in subject_ids:
    stc_filename = os.path.join(os.path.join(stc_dir, f"sub-{subject_id}"), f"sub-{subject_id}_task-{paradigm}_desc-dSPM-erp")
    stcs_to_average.append(mne.read_source_estimate(stc_filename))

grand_average_stc = sum(stcs_to_average) / len(stcs_to_average)

# Plot the grand-average source-space ERP
print("Plotting the grand-average source-space ERP...")
brain = grand_average_stc.plot(
    subject='fsaverage',  # Explicitly tell MNE to use the fsaverage brain
    hemi="split",
    clim="auto",
    subjects_dir=subjects_dir,
    backend="pyvistaqt",
    cortex="classic",
    surface="inflated",
    smoothing_steps=20,
    initial_time=0.1
)

if save:
    brain.save_movie(tmin=-0.1, tmax=0.4, interpolation='linear', time_dilation=20, framerate=5, time_viewer=True)


'''### ALTERNATIVE: LCMV beamformer ###
# make data cov
data_cov = mne.compute_covariance(epochs, tmin=signal_tmin, tmax=signal_tmax, method="empirical")
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

# Apply baseline correction to the beamformer source-space ERP
print(f"Applying baseline correction from {noise_tmin}s to {noise_tmax}s for LCMV...")
baseline_period = (noise_tmin, noise_tmax)
erp_stcs.apply_baseline(baseline=baseline_period)

# plot LCMV results
brain = erp_stcs.plot(
    hemi='both',
    initial_time=0.1, # Example: Start viewing around N1 peak
    clim=dict(kind="value", lims=[-0.5, 0, 0.5]),
    colormap='seismic',
    transparent=False,
    time_unit='s',
    subjects_dir=subjects_dir,
    time_viewer=True,  # This opens the dedicated time course viewer
    backend="pyvistaqt",
    show_traces=True,
    surface="inflated",
    smoothing_steps=10
)

if save:
    # The documentation website's movie is generated with:
    brain.save_movie(tmin=-0.1, tmax=0.4, interpolation='linear',
                      time_dilation=20, framerate=5, time_viewer=True)
'''
