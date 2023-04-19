# GAZE_EXP
By :      Martin SZINTE<br/>
Projet :  gaze_exp<br/>
With :    Matthias NAU, Jan Patrick STELLMANN & Tomas KNAPEN
Version:  1.0<br/>

## Version description
Codes grouping analyses of two preliminary projects made to determine<br/>
gaze modulation effects on fMRI data. <br/>
<u>See dataset README.md for details. </u><br/>

## Data analysis

##### Data management see _preproc/data_management.ipynb_
 - Get pRF data from gaze_prf project and 3T anatomy<br/>
 - Copy _event files from experiment code data to bids folder, rename sessions accordingly<br/>
 - Convert par rec to nifti and separate phase from bold<br/>
 - Nordic correction in matlab using _preproc/nordic_cor.m_<br/>
 - Correct error in header of nordicized files<br/>
 - Copy nordicized files in BIDS folder<br/>
 - Create bold and epi .json files, as well as particpant.tsv and .json and dataset_description.json and readme<br/>
 - Copy subject manually edited freesurfer segmentations<br/>
 - BIDS validations: change header time unit and duration<br/>
 
 ##### Quality check and preprocessing
 - run MRIQC using _preproc/mriqc_sbatch.py_<br/>
 - run fMRIPrep using _preproc/fmriprep_sbatch.py<br/>_
 - run pybest without pca analysis using _preproc/pybest_sbatch.py_<br/>
 
##### First set of analysis
- correlation maps between GazeCW/GazeCCW and GazeColumns/GazeLines using _postproc/cormaps.py_
- draw correlation maps on pycortex flatmaps using _postproc/pycortex_cormaps.ipynb_