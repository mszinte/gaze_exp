# GAZE_EXP

## About
---
Codes grouping analyses of two preliminary projects made to determine
gaze modulation effects on fMRI data.
<u>See dataset README.md for details. </u>
---
## Authors (alphabetic order): 
---
Margot CHEVILLARD, Tomas KNAPEN, Uriel LASCOMBES, Matthias NAU, Jan Patrick STELLMANN & Martin SZINTE<br/>
By :  Martin SZINTE

## Data analysis
---

### Pre-processing

#### BIDS
- [x] Data management with BIDS [data_management.ipynb](analysis_code/preproc/bids/data_management.ipynb)<br/>
        --> Get pRF data from gaze_prf project and 3T anatomy<br/>
            Copy _event files from experiment code data to bids folder, rename sessions accordingly<br/>
            Convert par rec to nifti and separate phase from bold<br/>
            Nordic correction in matlab using [nordic_cor.m](analysis_code/preproc/bids/nordic_cor.m)<br/>
            Correct error in header of nordicized files<br/>
            Copy nordicized files in BIDS folder<br/>
            Create bold and epi .json files, as well as particpant.tsv and .json and dataset_description.json and readme<br/>
            Copy subject manually edited freesurfer segmentations<br/>
            BIDS validations: change header time unit and duration<br/>



#### Structural preprocessing
- [x] fMRIprep with anat-only option [fmriprep_sbatch.py](analysis_code/preproc/functional/fmriprep_sbatch.py)

#### Functional preprocessing
- [x] MRIQC [mriqc_sbatch.py](analysis_code/preproc/functional/mriqc_sbatch.py)
- [x] fMRIprep [fmriprep_sbatch.py](analysis_code/preproc/functional/fmriprep_sbatch.py)
- [x] run pybest without pca analysis [pybest_sbatch.py](analysis_code/preproc/functional/pybest_sbatch.py)
- [x] high-pass, z-score, average and leave-one-out average [preproc_end.py](analysis_code/preproc/functional/preproc_end.py)
- [x] Load freesurfer and execute [pycortex_import.py](analysis_code/preproc/functional/pycortex_import.py): run only [freesurfer_import_pycortex.py](analysis_code/preproc/functional/freesurfer_import_pycortex.py)


### Post-processing

#### PRF analysis
- [x] create the visual matrix design [vdm_builder.py](analysis_code/postproc/prf/fit/vdm_builder.ipynb)
- [ ] Execute [prf_fit.py](analysis_code/postproc/prf/fit/prf_fit.py) to fit pRF parameters (eccentricity, size, amplitude, baseline, rsquare): run only [submit_fit_jobs.py](analysis_code/postproc/prf/fit/submit_fit_jobs.py)
- [ ] Compute pRF derivatives [compute_derivatives.py](analysis_code/postproc/prf/postfit/compute_derivatives.py)
    - [ ] add magnification factor see https://github.com/noahbenson/cortical-magnification-tutorial
- [ ] make pycortex maps [pycortex_maps.py](analysis_code/postproc/prf/postfit/pycortex_maps.py)
- [x] draw ROIs using Inkscape
- [ ] extract ROIs masks [roi_masks.ipynb](analysis_code/ROIs/roi_masks.ipynb)
- [ ] make pdf files with the maps [pdf_maps.py](analysis_code/postproc/prf/postfit/pdf_maps.py)
- [ ] make webgl with the pycortex dataset [pycortex_maps.py](analysis_code/postproc/prf/webgl/pycortex_webgl.py) 
- [ ] send the files [send_index.sh](analysis_code/postproc/prf/webgl/send_index.sh)


- correlation maps between GazeCW/GazeCCW and GazeColumns/GazeLines using _postproc/cormaps.py_
- draw correlation maps on pycortex flatmaps using _postproc/pycortex_cormaps.ipynb_

### Main analysis
- [ ] extract all data as pickle files or tsv
- [ ] think about the individual participants figures


