function [const] = dirSaveFile(const)
% ----------------------------------------------------------------------
% [const] = dirSaveFile(const)
% ----------------------------------------------------------------------
% Goal of the function :
% Make directory and saving files name and fid.
% ----------------------------------------------------------------------
% Input(s) :
% const : struct containing constant configurations
% ----------------------------------------------------------------------
% Output(s):
% const : struct containing constant configurations
% ----------------------------------------------------------------------
% Function created by Martin SZINTE (martin.szinte@gmail.com)
% ----------------------------------------------------------------------

% Create data directory 
if ~isfolder(sprintf('data/%s/%s/func/',const.sjct,const.session))
    mkdir(sprintf('data/%s/%s/func/',const.sjct,const.session))
end

% Define directory
const.dat_output_file   =   sprintf('data/%s/%s/func/%s_%s_task-%s_%s',const.sjct,const.session,const.sjct,const.session,const.cond1_txt,const.run_txt);

% Behavioral data
const.behav_file        =   sprintf('%s_events.tsv',const.dat_output_file);
if const.expStart
    if exist(const.behav_file,'file')
        aswErase = upper(strtrim(input(sprintf('\n\tThis file allready exist, do you want to erase it ? (Y or N): '),'s')));
        if upper(aswErase) == 'N'
            error('Please restart the program with correct input.')
        elseif upper(aswErase) == 'Y'
        else
            error('Incorrect input => Please restart the program with correct input.')
        end
    end
end
const.behav_file_fid    =   fopen(const.behav_file,'w');

% Create additional info directory
if ~isfolder(sprintf('data/%s/%s/add/',const.sjct,const.session))
    mkdir(sprintf('data/%s/%s/add/',const.sjct,const.session))
end

% Define directory
const.add_output_file   =   sprintf('data/%s/%s/add/%s_%s_task-%s_%s',const.sjct,const.session,const.sjct,const.session,const.cond1_txt,const.run_txt);

% Define .mat saving file
const.mat_file          =   sprintf('%s_matFile.mat',const.add_output_file);

% Log file
const.log_file          =   sprintf('%s_logData.txt',const.add_output_file);
const.log_file_fid      =   fopen(const.log_file,'w');

end