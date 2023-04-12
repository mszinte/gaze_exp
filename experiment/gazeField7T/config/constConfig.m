function [const]=constConfig(const)
% ----------------------------------------------------------------------
% [const]=constConfig(const)
% ----------------------------------------------------------------------
% Goal of the function :
% Define all constant configurations
% ----------------------------------------------------------------------
% Input(s) :
% const : struct containing constant configurations
% ----------------------------------------------------------------------
% Output(s):
% const : struct containing constant configurations
% ----------------------------------------------------------------------
% Function created by Martin SZINTE (martin.szinte@gmail.com)
% ----------------------------------------------------------------------

% Randomization
rng('default');
rng('shuffle');

%% Time parameters
const.TR_dur            =   1.3;                                                                % repetition time
const.calib_fix_tr      =   10;                                                                 % calibration fixation duration in TR
const.calib_buzz_tr     =   1;                                                                  % calibration buzzer duration in TR
const.main_fix_tr       =   68;                                                                 % main task fixation duration in TR
const.main_buzz_tr      =   1;                                                                  % main task buzzer duration in TR

const.calib_fix_dur     =   (const.calib_fix_tr+const.calib_buzz_tr)*const.TR_dur;              % calibration fixation duration in seconds
const.main_fix_dur      =   (const.main_fix_tr+const.main_buzz_tr)*const.TR_dur;                % main task fixation duration in seconds

const.calib_on_dur      =   const.calib_buzz_tr*const.TR_dur;                                   % calibration on buzzer in seconds
const.calib_off_dur     =   const.calib_fix_tr*const.TR_dur;                                    % calibration off buzzer in seconds
const.main_on_dur       =   const.main_buzz_tr*const.TR_dur;                                    % main task on buzzer in seconds
const.main_off_dur      =   const.main_fix_tr*const.TR_dur;                                     % main task off buzzer in seconds

%% Stim parameters
const.calib_fix_num     =   9;                                                                  % number of fixations in calibration
const.main_fix_num      =   5;                                                                  % number of fixations in main task
const.buzzer_freq       =   120;                                                                % buzzer vibration frequency (hz)
const.buzzer_amplitude  =   2;                                                                  % buzzer amplitude (a.u.)

const.calib_signal      =   createSignal(const.buzzer_freq, const.buzzer_amplitude, ...
                                         const.calib_on_dur, const.calib_off_dur, 1);
const.main_signal       =   createSignal(const.buzzer_freq, const.buzzer_amplitude, ...
                                         const.main_on_dur, const.main_off_dur, 1);

const.calib_signals = [];
for main_fix = 1:const.calib_fix_num
    const.calib_signals = [const.calib_signals;const.calib_signal];
end

const.main_signals = [];
for main_fix = 1:const.calib_fix_num
    const.main_signals = [const.main_signals;const.main_signal];
end

% 01: 'center'; 02: 'left';        03: 'left-up'; 04: 'up'; 05: 'right-up'
% 06: 'right';  07: 'right-down';  08: 'down'     09: 'left-down'

const.calib_fix_order    = [1,2,3,4,5,6,7,8,9];                                                 % order of fixation for calibration
const.main_cw_fix_order  = [1,2,4,6,8];                                                         % order of fixation for main task clockwise
const.main_ccw_fix_order = [8,6,4,2,1];                                                         % order of fixation for main task counter-clockwise

%% Buzzer settings (Dancer design)
const.buzzer_output_number = 1;
const.buzzer_daq_name = 'cDAQ1Mod2';

%% Instructions
const.calib_fix_order_instr = 'Order: center => left => left-up => up => right-up => right => right-down => down => left-down';
const.main_cw_fix_order_instr = 'Order: center => left => up => right => down';
const.main_ccw_fix_order_instr = 'Order: down => right => up => left => center';

% define TR for scanner
if const.scanner
    const.TRs = 0;
    switch const.cond1
        case 1; const.TRs = const.calib_fix_num * const.calib_fix_tr;
        case 2; const.TRs = const.main_fix_num * const.main_fix_tr;
        case 3; const.TRs = const.main_fix_num * const.main_fix_tr;
    end
    
    fprintf(1,'\n\tScanner parameters; %1.0f TRs, %1.2f seconds, %s\n',const.TRs,const.TR_dur,datestr(seconds((const.TRs*const.TR_dur)),'MM:SS'));
end

end