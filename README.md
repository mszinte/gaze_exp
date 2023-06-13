# GAZE_EXP
---
We aim at assession gaze modulation of retinotopic areas in complete darknees.</br>
This repository contain all codes allowing analysis of this preliminary dataset [OpenNeuro:DS004271](https://openneuro.org/datasets/ds004271).

## Authors (alphabetic order): 
---
Margot CHEVILLARD, Tomas KNAPEN, Uriel LASCOMBES, Matthias NAU, Martin SZINTE

### Main dependencies
---
[dcm2niix](https://github.com/rordenlab/dcm2niix); 
[PyDeface](https://github.com/poldracklab/pydeface); 
[fMRIprep](https://fmriprep.org/en/stable/); 
[pRFpy](https://github.com/VU-Cog-Sci/prfpy); 
[FreeSurfer](https://surfer.nmr.mgh.harvard.edu/);
[FFmpeg](https://ffmpeg.org/);
[FSL](https://fsl.fmrib.ox.ac.uk);
[Inkscape](https://inkscape.org/)
</br>


### Experiment codes
---

#### Tasks:
_Calib / GazeCW / GazeCCW:_ Resting-state task with different gaze direction change [expLauncher.m](experiment_code/gazeField7T/main/expLauncher.m)</br>
_Calib / GazeLines / GazeColumns :_ Quick fixation change in complete darkness [expLauncher.m](experiment_code/pGFexp7T/main/expLauncher.m)</br>
_AttendFix_GazeCenterFS / AttendStime_GazeCenterFS :_ Retinotopy tasks [expLauncher.m](experiment_code/pRFexp7T/main/expLauncher.m)</br>

---
### Pre-processing

#### Data management with [data_management.ipydb](analysis_code/preproc/bids/data_management.ipynb)
- [x] Copy pRF data from gaze_prf project and 3T anatomy.
- [x] Copy event files from experiment code data to bids folder, rename sessions accordingly.
- [x] Convert par rec to nifti and separate phase from bold using [parrec_to_nii.py](analysis_code/preproc/bids/parrec_to_nii.py).
- [x] Nordic correction in matlab using [nordic_cor.m](analysis_code/preproc/bids/nordic_cor.m).
- [x] Correct error in header of nordicized files
- [x] Copy nordicized files in BIDS folder
- [x] Create bold and epi .json files, as well as particpant.tsv and .json and dataset_description.json and readme
- [x] Copy subject manually edited freesurfer segmentations
- [x] BIDS validations: change header time unit and duration
- [ ] Deface subjects anatomy (to do in post-pilot study)
 
#### Structural preprocessing
- [x] run MRIQC using [mriqc_sbatch.py](analysis_code/preproc/anatomical/mriqc_sbatch.py)
- [x] run fMRIPrep (anat-only) [fmriprep_sbatch.py](analysis_code/preproc/functional/fmriprep_sbatch.py)
- [x] create sagital view video before manual edit [sagital_view.py](analysis_code/preproc/anatomical/sagital_view.py)
- [x] manual edit of brain segmentation [pial_edits.sh](analysis_code/preproc/anatomical/pial_edits.sh)
- [x] FreeSurfer with new brainmask manually edited [freesurfer_pial.py](analysis_code/preproc/anatomical/freesurfer_pial.py)
- [x] create sagital view video before after edit [sagital_view.py](analysis_code/preproc/anatomical/sagital_view.py)
- [x] make cut in the brains for flattening [cortex_cuts.sh](analysis_code/preproc/anatomical/cortex_cuts.sh)
- [x] flatten the cut brains [flatten_sbatch.py](analysis_code/preproc/anatomical/flatten_sbatch.py)

#### Functional preprocessing
- [x] run fMRIprep [fmriprep_sbatch.py](analysis_code/preproc/functional/fmriprep_sbatch.py)
- [x] high-pass, z-score, average and leave-one-out average [preproc_end.py](analysis_code/preproc/functional/preproc_end.py)
- [x] Create pycortex database [pycortex_import.py](analysis_code/preproc/functional/pycortex_import.py)

---
### Post-processing

#### PRF analysis
- [x] create the visual matrix design [vdm_builder.py](analysis_code/postproc/prf/fit/vdm_builder.ipynb)
- [x] Execute [prf_fit.py](analysis_code/postproc/prf/fit/prf_fit.py) to fit pRF parameters (eccentricity, size, amplitude, baseline, rsquare): run only [submit_fit_jobs.py](analysis_code/postproc/prf/fit/submit_fit_jobs.py)
- [x] Compute pRF derivatives [compute_derivatives.py](analysis_code/postproc/prf/postfit/compute_derivatives.py)
    - [ ] add magnification factor see https://github.com/noahbenson/cortical-magnification-tutorial
- [x] make pycortex maps [pycortex_maps.py](analysis_code/postproc/prf/postfit/pycortex_maps.py)
- [x] draw ROIs using Inkscape
- [x] extract ROIs masks [roi_masks.ipynb](analysis_code/postproc/prf/postfit/roi_masks.ipynb)
- [X] make pdf files with the maps [pdf_maps.py](analysis_code/postproc/prf/postfit/pdf_maps.py)
- [x] make webgl with the pycortex dataset [pycortex_maps.py](analysis_code/postproc/prf/webgl/pycortex_webgl.py) 
- [x] send the files [send_data.sh](analysis_code/postproc/prf/webgl/send_data.sh)

#### Correlation analysis
- [x] extract all data as pickle files or tsv [make_tsv.ipynb](analysis_code/postproc/prf/postfit/make_tsv.ipynb)
- plot pRF parameters
    - [lines_size_r2_ecc.ipynb](analysis_code/postproc/correlations/lines_size_r2_ecc.ipynb)
    - [Violin_plot.ipynb](analysis_code/postproc/correlations/Violin_plot.ipynb)
    - [polar_angle.ipynb](analysis_code/postproc/correlations/polar_angle.ipynb)
    
- correlation maps between GazeCW/GazeCCW and GazeColumns/GazeLines using _postproc/cormaps.py_
- draw correlation maps on pycortex flatmaps using _postproc/pycortex_cormaps.ipynb_