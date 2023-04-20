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
const.main_fix_tr       =   2;                                                                  % main task fixation duration in TR
const.main_buzz_tr      =   1;                                                                  % main task buzzer duration in TR

const.calib_fix_dur     =   (const.calib_fix_tr+const.calib_buzz_tr)*const.TR_dur;              % calibration fixation duration in seconds
const.main_fix_dur      =   (const.main_fix_tr+const.main_buzz_tr)*const.TR_dur;                % main task fixation duration in seconds

const.calib_on_dur      =   const.calib_buzz_tr*const.TR_dur;                                   % calibration on buzzer in seconds
const.calib_off_dur     =   const.calib_fix_tr*const.TR_dur;                                    % calibration off buzzer in seconds
const.main_on_dur       =   const.main_buzz_tr*const.TR_dur;                                    % main task on buzzer in seconds
const.main_off_dur      =   const.main_fix_tr*const.TR_dur;                                     % main task off buzzer in seconds

%% Stim parameters
const.calib_fix_num     =   9;                                                                  % number of fixations in calibration
const.main_fix_num      =   97;                                                                 % number of fixations in main task
const.buzzer_freq       =   120;                                                                % buzzer vibration frequency (hz)
const.buzzer_amplitude  =   2;                                                                  % buzzer amplitude (a.u.)

const.calib_signal      =   createSignal(const.buzzer_freq, const.buzzer_amplitude, ...
                                         const.calib_on_dur, const.calib_off_dur, 1);
const.main_signal       =   createSignal(const.buzzer_freq, const.buzzer_amplitude, ...
                                         const.main_on_dur, const.main_off_dur, 1);

const.calib_signals = [];
for calib_fix = 1:const.calib_fix_num
    const.calib_signals = [const.calib_signals;const.calib_signal];
end

const.main_signals = [];
for main_fix = 1:const.main_fix_num
    const.main_signals = [const.main_signals;const.main_signal];
end

% Calibration mapping
% -------------------
% 03     04      05
%
% 02     01      06
%
% 09     08      07

const.calib_fix_order    = [1,2,3,4,5,6,7,8,9];                                                 % order of fixation for calibration

% Main task mapping
% -----------------
% 01   02   03   04   05
% 06   07   08   09   10
% 11   12   13   14   15
% 16   17   18   19   20
% 21   22   23   24   25

const.main_lines_fix_order = [01,02,03,04,05,...
                              10,09,08,07,06,...
                              11,12,13,14,15,...
                              20,19,18,17,16,...
                              21,22,23,24,25,...
                              24,23,22,21,...
                              16,17,18,19,20,...
                              15,14,13,12,11,...
                              06,07,08,09,10,...
                              05,04,03,02,01,...
                              02,03,04,05,...
                              10,09,08,07,06,...
                              11,12,13,14,15,...
                              20,19,18,17,16,...
                              21,22,23,24,25,...
                              24,23,22,21,...
                              16,17,18,19,20,...
                              15,14,13,12,11,...
                              06,07,08,09,10,...
                              05,04,03,02,01];                                                  % order of fixation for main task gaze lines
                          
const.main_columns_fix_order = [01,06,11,16,21,...
                                22,17,12,07,02,...
                                03,08,13,18,23,...
                                24,19,14,09,04,...
                                05,10,15,20,25,...
                                20,15,10,05,...
                                04,09,14,19,24,...
                                23,18,13,08,03,...
                                02,07,12,17,22,...
                                21,16,11,06,01,...
                                06,11,16,21,...
                                22,17,12,07,02,...
                                03,08,13,18,23,...
                                24,19,14,09,04,...
                                05,10,15,20,25,...
                                20,15,10,05,...
                                04,09,14,19,24,...
                                23,18,13,08,03,...
                                02,07,12,17,22,...
                                21,16,11,06,01];                                                % order of fixation for main task gaze lines

%% Buzzer settings (Dancer design)
const.buzzer_output_number = 1;
const.buzzer_daq_name = 'cDAQ1Mod2';

%% Instructions
const.calib_fix_order_instr = 'Order: center => left => left-up => up => right-up => right => right-down => down => left-down';
const.main_lines_order_instr = 'Gaze in lines: 2X top-left => bottom-right => top-left';
const.main_columns_order_instr = 'Gaze in columns: 2X top-left => bottom-right => top-left';

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