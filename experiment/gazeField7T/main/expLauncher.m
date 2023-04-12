%% General experimenter launcher
%  =============================
% By :      Martin SZINTE
% Projet :  GazeField7T
% With :    Matthias NAU, Jan Patrick STELLMANN & Tomas Knapen

% Version description
% ===================
% Calibration with 9 positions, 13 sec each
% Calib: center, up, down, left, right, left-up, right-up, left-down, right-down

% Main task5 positions 1 min 28.4 sec  
% Two rotation order CW or CCW
% GazeCW : center -> right -> up -> left -> down
% GazeCCW : down -> left -> up -> right -> center

% Run sequence
% ------------
% Run 1 : task-Calib_run-01 :
% Run 2 : task-GazeCW_run-01 : 7 min 22 sec
% Run 3 : task-GazeCCW_run-01 : 7 min 22 sec
% Run 4 : task-GazeCW_run-02 : 7 min 22 sec
% Run 5 : task-GazeCCW_run-02 : 7 min 22 sec
% Run 6 : task-GazeCW_run-03 : 7 min 22 sec
% Run 7 : task-GazeCCW_run-03 : 7 min 22 sec

% To do before scanning:
% ---------------------

% 1. Turn on computer and Matlab
% 2. Turn on small power socket of Dancer Design amplifier
% 3. Plug mini-PTS headbox in scanner room, right socket-A (do not switch on)
% 4. Install subject
% 5. Strap a band on the right index and strap the vibrator to it
% 6. Plug vibrator ot socket 1 in mini-PTS headbox
% 7. Turn on vibrator
% 6. Make vibrator test
% 8. Start experiment

% To do after scanning:
% ---------------------
% 1. Turn off mini-PTS headbox
% 2. Unplug vibrator from mini-PTS headbox
% 3. Turn off small power socket of Dancer Design amplifier
% 4. Clean room and put back all equipment

% First settings
% --------------
clear all;clear mex;clear functions;close all;home;

% General settings
% ----------------
const.expStart          =   1;              % Start of a recording exp                          0 = NO  , 1 = YES
const.checkTrial        =   1;              % Print trial conditions (for debugging)            0 = NO  , 1 = YES

% External controls
% -----------------
const.scanner           =   1;              % run in MRI scanner                                0 = NO  , 1 = YES
const.scannerTest       =   0;              % run with T returned at TR time                    0 = NO  , 1 = YES
const.buzzer            =   1;              % Buzzer                                            0 = NO  , 1 = YES

% Run order and number per condition
% ----------------------------------
const.cond_run_order    =   [01;02;03;02;03;02;03];
const.cond_run_num      =   [01;01;01;02;02;03;03];

% Path
% ----
dir                     =   (which('expLauncher'));
cd(dir(1:end-18));

% Add Matlab path
% ---------------
addpath('config','main','trials','stim');

% Subject configuration
% ---------------------
[const] = sbjConfig(const);

% Main run
% --------
main(const);