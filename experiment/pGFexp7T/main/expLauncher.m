%% General experimenter launcher
%  =============================
% By :      Martin SZINTE
% Projet :  pGFexp7T
% With :    Matthias NAU, Jan Patrick STELLMANN & Tomas Knapen

% Version description
% ===================
% Determine gain field maps using an equivalent to pRF scanning design 
% but with gaze changes over a large visual portion of the visual field 
% in total darkness.

% Calibration with 9 positions, 13 sec each
% Directions : center, up, down, left, right, left-up, right-up, left-down, right-down

% GazeLines :   row1-col1 -> row1-col2 -> row1-col3 -> row1-col4 -> row1-col5
%               row2-col5 -> row2-col4 -> row2-col3 -> row2-col4 -> row2-col5
%               row3-col1 -> row3-col2 -> row3-col3 -> row3-col4 -> row3-col5
%               row4-col5 -> row4-col4 -> row4-col3 -> row4-col4 -> row4-col5
%               row5-col1 -> row5-col2 -> row5-col3 -> row5-col4 -> row5-col5
%               row5-col5 -> row5-col4 -> row5-col3 -> row5-col2 -> row1-col1
%               row4-col1 -> row4-col2 -> row4-col3 -> row4-col4 -> row2-col5
%               row3-col4 -> row3-col4 -> row3-col3 -> row3-col2 -> row3-col1
%               row2-col1 -> row2-col2 -> row2-col3 -> row2-col4 -> row4-col5
%               row1-col5 -> row1-col4 -> row1-col3 -> row1-col2 -> row5-col1

% GazeColumns : row1-col1 -> row2-col1 -> row3-col1 -> row4-col1 -> row5-col1
%               row5-col2 -> row4-col2 -> row3-col2 -> row2-col2 -> row1-col2
%               row1-col3 -> row2-col3 -> row3-col3 -> row4-col3 -> row5-col3
%               row5-col4 -> row4-col4 -> row3-col4 -> row2-col4 -> row1-col4
%               row1-col5 -> row2-col5 -> row3-col5 -> row4-col5 -> row5-col5
%               row5-col5 -> row4-col5 -> row3-col5 -> row2-col5 -> row1-col5
%               row1-col4 -> row2-col4 -> row3-col4 -> row4-col4 -> row5-col4
%               row5-col3 -> row4-col3 -> row3-col3 -> row2-col3 -> row1-col3
%               row1-col2 -> row2-col2 -> row3-col2 -> row4-col2 -> row5-col2
%               row5-col1 -> row4-col1 -> row3-col1 -> row2-col1 -> row1-col1

% Run sequence
% ------------
% Run  1 : task-Calib_run-01: 90TRs - 1 min 57 sec
% Run  2 : task-GazeLines_run-01 : 194TRs - 4 min 12.2 sec
% Run  3 : task-GazeColumns_run-01 : 194TRs - 4 min 12.2 sec
% Run  4 : task-GazeLines_run-02 : 194TRs - 4 min 12.2 sec
% Run  5 : task-GazeColumns_run-02 : 194TRs - 4 min 12.2 sec
% Run  6 : task-GazeLines_run-03 : 194TRs - 4 min 12.2 sec
% Run  7 : task-GazeColumns_run-03 : 194TRs - 4 min 12.2 sec
% Run  8 : task-GazeLines_run-04 : 194TRs - 4 min 12.2 sec
% Run  9 : task-GazeColumns_run-04 : 194TRs - 4 min 12.2 sec
% Run 10 : task-GazeLines_run-05 : 194TRs - 4 min 12.2 sec
% Run 11 : task-GazeColumns_run-05 : 194TRs - 4 min 12.2 sec

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
const.scannerTest       =   1;              % run with T returned at TR time                    0 = NO  , 1 = YES
const.buzzer            =   0;              % Buzzer                                            0 = NO  , 1 = YES

% Run order and number per condition
% ----------------------------------
const.cond_run_order    =   [01;02;03;02;03;02;03;02;03;02;03];
const.cond_run_num      =   [01;01;01;02;02;03;03;04;04;05;05];

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