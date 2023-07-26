% ----------------------------------------------------------------------
% nordic_cor
% ----------------------------------------------------------------------
% Goal of the script :
% Run nordic correction as in this paper
% https://www.nature.com/articles/s41467-021-25431-8
% ----------------------------------------------------------------------
% Function created by Martin SZINTE (martin.szinte@gmail.com)
% ----------------------------------------------------------------------

clear all

% define nifti folders localy and on server
server_nifti_dir = 'mszinte@login.mesocentre.univ-amu.fr:/scratch/mszinte/data/gaze_exp/sourcedata/nifti/';
local_nifti_dir = '/Users/martinszinte/Downloads/nifti/';

% load data from server to local
system(sprintf('rsync -avuz %s %s',server_nifti_dir,local_nifti_dir))

% get nordic function
dir_fun = which('nordic_cor');
addpath(dir_fun(1:end-12))

% Set nordic settings
ARG.temporal_phase=1;
ARG.phase_filter_width=10;

% get list of files
for subject_num = 1:2
    for session_num = 4:5
        
        subject = sprintf('sub-00%i',subject_num);
        session = sprintf('ses-0%i',session_num);
        
        phase_files = dir(sprintf('%s/%s_%s/*_phase*',local_nifti_dir, subject, session));
        magnitude_files = dir(sprintf('%s/%s_%s/*_magnitude*',local_nifti_dir, subject, session));
        
        phase_files_num = size(phase_files);
        if phase_files_num > 0
            for phase_file_num = 1:phase_files_num
                
                cd(sprintf('%s/%s_%s/',local_nifti_dir, subject, session))
                
                fn_magn_in = magnitude_files(phase_file_num).name;
                fn_phase_in = phase_files(phase_file_num).name;
                fn_out = sprintf('%s_nordic',fn_magn_in(1:end-7));
                
                % run nordic correction

                if ~exist(sprintf('%s.nii.gz',fn_out),'file')

                    fprintf(1,'\nCreate nordic correction %s\n\n',fn_out)
                    NIFTI_NORDIC(fn_magn_in,fn_phase_in,fn_out,ARG);
                    % gzip file
                    gzip(sprintf('%s.nii',fn_out))
                    delete(sprintf('%s.nii',fn_out))
                else
                    fprintf(1,'\nNordic correction %s already exist\n\n',fn_out)
                end

                
                
            end
        end
        
    end
end

% upload data from local to server
system(sprintf('rsync -avuz %s %s',local_nifti_dir,server_nifti_dir))
