Node: funcMRIQC (ComputeIQMs (metadata (bids)
=============================================


 Hierarchy : workflow_enumerator.funcMRIQC.ComputeIQMs.metadata
 Exec ID : metadata.a21


Original Inputs
---------------


* bids_dir : None
* bids_validate : True
* in_file : /work_dir/gaze_exp/sub-001/ses-05/func/sub-001_ses-05_task-GazeLines_run-04_bold.nii.gz

Execution Inputs
----------------


* bids_dir : None
* bids_validate : True
* in_file : /work_dir/gaze_exp/sub-001/ses-05/func/sub-001_ses-05_task-GazeLines_run-04_bold.nii.gz


Execution Outputs
-----------------


* acquisition : <undefined>
* out_dict : {'AcquisitionMatrixPE': 112, 'CoilString': 'MULTI_COIL', 'DeviceSerialNumber': '00076', 'EchoTime': 0.016901, 'EchoTrainLength': 43, 'FlipAngle': 13, 'ImageType': ['ORIGINAL', 'PRIMARY', 'PHASE', 'MAP', 'P', 'FFE'], 'ImagingFrequency': 298.03, 'InstitutionAddress': 'Amsterdam', 'InstitutionName': 'Spinoza_Centre', 'InstitutionalDepartmentName': 'Spinoza_Centre', 'MRAcquisitionType': '3D', 'MagneticFieldStrength': 7, 'Manufacturer': 'Philips', 'ManufacturersModelName': 'Achieva', 'Modality': 'MR', 'NumberOfVolumesDiscardedByUser': 3, 'PatientPosition': 'HFS', 'PercentPhaseFOV': 100, 'PhaseEncodingDirection': 'i', 'PhaseEncodingSteps': 112, 'PhilipsRWVIntercept': -3142, 'PhilipsRWVSlope': 1.53455, 'PhilipsRescaleIntercept': -3142, 'PhilipsRescaleSlope': 1.53455, 'PhilipsScaleSlope': 651.74, 'PixelBandwidth': 1762, 'ReconMatrixPE': 112, 'RepetitionTime': 1.33, 'SAR': 1.96128, 'ScanOptions': 'FS', 'ScanningSequence': 'GR', 'SequenceVariant': 'SK', 'SeriesInstanceUID': '1.3.46.670589.11.76.5.0.6388.2022071316500147307', 'SeriesNumber': 601, 'SliceThickness': 1.8, 'SoftwareVersions': '5.1.7_5.1.7.0', 'SpacingBetweenSlices': 1.8, 'StationName': '7t-acq-multix', 'StudyID': '694967284', 'StudyInstanceUID': '1.3.46.670589.11.76.5.0.7360.2022071316280508000', 'TaskName': 'GazeLines', 'UsePhilipsFloatNotDisplayScaling': 1}
* reconstruction : <undefined>
* run : 4
* session : 05
* subject : 001
* suffix : <undefined>
* task : GazeLines


Runtime info
------------


* duration : 81.009606
* hostname : login01.cluster
* prev_wd : /home/mszinte/projects/gaze_exp/analysis_code/preproc/anatomical
* working_dir : /home/mszinte/projects/gaze_exp/analysis_code/preproc/anatomical/work/workflow_enumerator/funcMRIQC/ComputeIQMs/_in_file_..work_dir..gaze_exp..sub-001..ses-05..func..sub-001_ses-05_task-GazeLines_run-04_bold.nii.gz/metadata


Environment
~~~~~~~~~~~


* AFNI_IMSAVE_WARNINGS : NO
* AFNI_MODELPATH : /opt/afni/models
* AFNI_PLUGINPATH : /opt/afni/plugins
* AFNI_TTATLAS_DATASET : /opt/afni/atlases
* ANTSPATH : /usr/lib/ants
* APPTAINER_APPNAME : 
* APPTAINER_BIND : /work_dir
* APPTAINER_COMMAND : run
* APPTAINER_CONTAINER : /scratch/mszinte/data/gaze_exp/code/singularity/mriqc-0.15.1.simg
* APPTAINER_ENVIRONMENT : /.singularity.d/env/91-environment.sh
* APPTAINER_NAME : mriqc-0.15.1.simg
* CONDA_DEFAULT_ENV : mszinte
* CONDA_EXE : /home/mszinte/softwares/anaconda3/bin/conda
* CONDA_PREFIX : /home/mszinte/softwares/anaconda3/envs/mszinte
* CONDA_PREFIX_1 : /home/mszinte/softwares/anaconda3
* CONDA_PROMPT_MODIFIER : (mszinte) 
* CONDA_PYTHON_EXE : /home/mszinte/softwares/anaconda3/bin/python
* CONDA_SHLVL : 2
* CPATH : /usr/local/miniconda/include/:
* FIX_VERTEX_AREA : 
* FMRI_ANALYSIS_DIR : /home/mszinte/softwares/freesurfer/fsfast
* FREESURFER_HOME : /home/mszinte/softwares/freesurfer
* FSFAST_HOME : /home/mszinte/softwares/freesurfer/fsfast
* FSF_OUTPUT_FORMAT : nii.gz
* FSLDIR : /usr/share/fsl/5.0
* FSLGECUDAQ : cuda.q
* FSLLOCKDIR : 
* FSLMACHINELIST : 
* FSLMULTIFILEQUIT : TRUE
* FSLOUTPUTTYPE : NIFTI_GZ
* FSLREMOTECALL : 
* FSLTCLSH : /usr/bin/tclsh
* FSLWISH : /usr/bin/wish
* FS_OVERRIDE : 0
* FUNCTIONALS_DIR : /home/mszinte/softwares/freesurfer/sessions
* HISTCONTROL : ignoredups
* HISTSIZE : 1000
* HOME : /home/bidsapp
* HOSTNAME : login01.cluster
* LANG : C.UTF-8
* LC_ALL : C.UTF-8
* LD_LIBRARY_PATH : /usr/lib/fsl/5.0::/.singularity.d/libs
* LESSOPEN : ||/usr/bin/lesspipe.sh %s
* LOADEDMODULES : 
* LOCAL_DIR : /home/mszinte/softwares/freesurfer/local
* LOGNAME : mszinte
* LS_COLORS : rs=0:di=38;5;27:ln=38;5;51:mh=44;38;5;15:pi=40;38;5;11:so=38;5;13:do=38;5;5:bd=48;5;232;38;5;11:cd=48;5;232;38;5;3:or=48;5;232;38;5;9:mi=05;48;5;232;38;5;15:su=48;5;196;38;5;15:sg=48;5;11;38;5;16:ca=48;5;196;38;5;226:tw=48;5;10;38;5;16:ow=48;5;10;38;5;21:st=48;5;21;38;5;15:ex=38;5;34:*.tar=38;5;9:*.tgz=38;5;9:*.arc=38;5;9:*.arj=38;5;9:*.taz=38;5;9:*.lha=38;5;9:*.lz4=38;5;9:*.lzh=38;5;9:*.lzma=38;5;9:*.tlz=38;5;9:*.txz=38;5;9:*.tzo=38;5;9:*.t7z=38;5;9:*.zip=38;5;9:*.z=38;5;9:*.Z=38;5;9:*.dz=38;5;9:*.gz=38;5;9:*.lrz=38;5;9:*.lz=38;5;9:*.lzo=38;5;9:*.xz=38;5;9:*.bz2=38;5;9:*.bz=38;5;9:*.tbz=38;5;9:*.tbz2=38;5;9:*.tz=38;5;9:*.deb=38;5;9:*.rpm=38;5;9:*.jar=38;5;9:*.war=38;5;9:*.ear=38;5;9:*.sar=38;5;9:*.rar=38;5;9:*.alz=38;5;9:*.ace=38;5;9:*.zoo=38;5;9:*.cpio=38;5;9:*.7z=38;5;9:*.rz=38;5;9:*.cab=38;5;9:*.jpg=38;5;13:*.jpeg=38;5;13:*.gif=38;5;13:*.bmp=38;5;13:*.pbm=38;5;13:*.pgm=38;5;13:*.ppm=38;5;13:*.tga=38;5;13:*.xbm=38;5;13:*.xpm=38;5;13:*.tif=38;5;13:*.tiff=38;5;13:*.png=38;5;13:*.svg=38;5;13:*.svgz=38;5;13:*.mng=38;5;13:*.pcx=38;5;13:*.mov=38;5;13:*.mpg=38;5;13:*.mpeg=38;5;13:*.m2v=38;5;13:*.mkv=38;5;13:*.webm=38;5;13:*.ogm=38;5;13:*.mp4=38;5;13:*.m4v=38;5;13:*.mp4v=38;5;13:*.vob=38;5;13:*.qt=38;5;13:*.nuv=38;5;13:*.wmv=38;5;13:*.asf=38;5;13:*.rm=38;5;13:*.rmvb=38;5;13:*.flc=38;5;13:*.avi=38;5;13:*.fli=38;5;13:*.flv=38;5;13:*.gl=38;5;13:*.dl=38;5;13:*.xcf=38;5;13:*.xwd=38;5;13:*.yuv=38;5;13:*.cgm=38;5;13:*.emf=38;5;13:*.axv=38;5;13:*.anx=38;5;13:*.ogv=38;5;13:*.ogx=38;5;13:*.aac=38;5;45:*.au=38;5;45:*.flac=38;5;45:*.mid=38;5;45:*.midi=38;5;45:*.mka=38;5;45:*.mp3=38;5;45:*.mpc=38;5;45:*.ogg=38;5;45:*.ra=38;5;45:*.wav=38;5;45:*.axa=38;5;45:*.oga=38;5;45:*.spx=38;5;45:*.xspf=38;5;45:
* MAIL : /var/spool/mail/mszinte
* MINC_BIN_DIR : /home/mszinte/softwares/freesurfer/mni/bin
* MINC_LIB_DIR : /home/mszinte/softwares/freesurfer/mni/lib
* MKL_NUM_THREADS : 1
* MNI_DATAPATH : /home/mszinte/softwares/freesurfer/mni/data
* MNI_DIR : /home/mszinte/softwares/freesurfer/mni
* MNI_PERL5LIB : /home/mszinte/softwares/freesurfer/mni/share/perl5
* MODULEPATH : /trinity/shared/modules/groups/
* MODULESHOME : /usr/share/Modules
* MPLCONFIGDIR : /tmp/matplotlib-r36vlms0
* OLDPWD : /home/mszinte/projects/gaze_exp/analysis_code/preproc
* OMP_NUM_THREADS : 1
* OS : Linux
* PATH : /usr/local/miniconda/bin:/opt/afni:/usr/lib/ants:/usr/lib/fsl/5.0:/usr/lib/afni/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
* PERL5LIB : /home/mszinte/softwares/freesurfer/mni/share/perl5
* POSSUMDIR : /usr/share/fsl/5.0
* PROMPT_COMMAND : conda_auto_env;git_and_conda_prompt; PROMPT_COMMAND="${PROMPT_COMMAND%%; PROMPT_COMMAND=*}"; PS1="Singularity> "
* PS1 : Singularity> 
* PWD : /home/mszinte/projects/gaze_exp/analysis_code/preproc/anatomical
* PYTHONNOUSERSITE : 1
* QTDIR : /usr/lib64/qt-3.3
* QTINC : /usr/lib64/qt-3.3/include
* QTLIB : /usr/lib64/qt-3.3/lib
* QT_GRAPHICSSYSTEM_CHECKED : 1
* SHELL : /bin/bash
* SHLVL : 1
* SINGULARITY_BIND : /work_dir
* SINGULARITY_CONTAINER : /scratch/mszinte/data/gaze_exp/code/singularity/mriqc-0.15.1.simg
* SINGULARITY_ENVIRONMENT : /.singularity.d/env/91-environment.sh
* SINGULARITY_NAME : mriqc-0.15.1.simg
* SQUEUE_FORMAT : %.8i %.9P %.8j %.2t %.8u %.7a %.5C %.6D %.20R %.19S %L
* SSH_CLIENT : 78.47.166.135 46244 8822
* SSH_CONNECTION : 78.47.166.135 46244 193.51.217.200 8822
* SSH_TTY : /dev/pts/65
* SUBJECTS_DIR : /scratch/mszinte/data/PredictEye/deriv_data/fmriprep/freesurfer/
* TERM : xterm-256color
* USER : mszinte
* USER_PATH : /home/mszinte/softwares/fsl/bin:/home/mszinte/softwares/freesurfer/bin:/home/mszinte/softwares/freesurfer/fsfast/bin:/home/mszinte/softwares/freesurfer/tktools:/home/mszinte/softwares/freesurfer/mni/bin:/home/mszinte/softwares/anaconda3/envs/mszinte/bin:/home/mszinte/softwares/anaconda3/condabin:/usr/lib64/qt-3.3/bin:/usr/local/bin:/usr/bin:/usr/local/sbin:/usr/sbin:/opt/dell/srvadmin/bin:/home/mszinte/.local/bin:/home/mszinte/bin:/bin:/usr/bin:/sbin:/usr/sbin:/usr/local/bin:/usr/local/sbin
* XDG_RUNTIME_DIR : /run/user/1568
* XDG_SESSION_ID : c1137102
* _ : /usr/bin/singularity
* _CE_CONDA : 
* _CE_M : 

