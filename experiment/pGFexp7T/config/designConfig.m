function [expDes]=designConfig(const)
% ----------------------------------------------------------------------
% [expDes]=designConfig(const)
% ----------------------------------------------------------------------
% Goal of the function :
% Define experimental design
% ----------------------------------------------------------------------
% Input(s) :
% const : struct containing constant configurations
% ----------------------------------------------------------------------
% Output(s):
% expDes : struct containg experimental design
% ----------------------------------------------------------------------
% Function created by Martin SZINTE (martin.szinte@gmail.com)
% Last update : 09 / 02 / 2021
% Project :     pRFexp7T
% Version :     1.0
% ----------------------------------------------------------------------

%% Experimental random variables

% Cond 1 : task (1 modality)
% =======
expDes.oneC             =   [1,2,4];
expDes.txt_cond1        =   {'calib', 'main-lines', 'main-columns'};
% 01. calibration 9 points fixation most extreme (center, left, left-up, up, right-up, right, right-down, down, left-down)
% 02. main task 25 points to fixate in lines startin up-left and ending bottom right
% 02. main task 25 points to fixate in columns startin up-left and ending bottom right

% Var 1 : fixation directions (9 modalities)
% ======
expDes.oneV             =   [01,02,03,04,05,...
                             06,07,08,09,10,...
                             11,12,13,14,15,...
                             16,17,18,19,20,...
                             21,22,23,24,25];
expDes.txt_var1_calib   =   {'center','left','left-up','up','right-up','right','right-down','down','left-down'};
expDes.txt_var1         =   {'row1-col1','row1-col2','row1-col3','row1-col4','row1-col5',...
                             'row2-col1','row2-col2','row2-col3','row2-col4','row2-col5',...
                             'row3-col1','row3-col2','row3-col3','row3-col4','row3-col5',...
                             'row4-col1','row4-col2','row4-col3','row4-col4','row4-col5',...
                             'row5-col1','row5-col2','row5-col3','row5-col4','row5-col5'};
                         

% 01 02 03 04 05
% 06 07 08 09 10
% 11 12 13 14 15
% 16 17 18 19 20
% 21 22 23 24 25

%% Experimental configuration :
expDes.nb_cond = 3;
expDes.nb_var = 1;

%% Experimental loop
rng('default');rng('shuffle');
runT = const.runNum;

switch const.cond1
    case 1
        fix_num = const.calib_fix_num; 
        fix_order = const.calib_fix_order;
    case 2
        fix_num = const.main_fix_num;
        fix_order = const.main_lines_fix_order;
    case 3
        fix_num = const.main_fix_num;
        fix_order = const.main_columns_fix_order;
end

for t_trial = 1:fix_num
    cond1 = const.cond1;
    var1 = fix_order(t_trial);
    
    expDes.expMat(t_trial,:) = [runT, t_trial, cond1, var1, NaN, NaN];
    % col 01:   Run number
    % col 02:   Trial number
    % col 03:   Task condition
    % col 04:   Fixation direction
    % col 05:   Trial onset time
    % col 06:   Trial offset time
end
expDes.nb_trials = size(expDes.expMat,1);

end