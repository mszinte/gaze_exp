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
expDes.txt_cond1        =   {'calib', 'main-cw', 'main-ccw'};
% 01. calibration 9 points fixation most extreme (center, left, left-up, up, right-up, right, right-down, down, left-down)
% 02. main task 5 points clockwise fixation most extreme (center, left, up, right, down)
% 03. main task 5 points counter-clockwise fixation most extreme (center, left, up, right, down)

% Var 1 : fixation directions (9 modalities)
% ======
expDes.oneV             =   [1;2;3;4;5;6;7;8;9];
expDes.txt_var1         =   {'center','left','left-up','up','right-up','right','right-down','down','left-down'};
% 01 = center
% 02 = left
% 03 = left-up
% 04 = up
% 05 = right-up
% 06 = right
% 07 = right-down
% 08 = down
% 09 = left-down

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
        fix_order = const.main_cw_fix_order;
    case 3
        fix_num = const.main_fix_num;
        fix_order = const.main_ccw_fix_order;
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