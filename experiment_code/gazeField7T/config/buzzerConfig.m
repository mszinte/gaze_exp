function [const] = buzzerConfig(const)
% ----------------------------------------------------------------------
% [const] = buzzerConfig(const)
% ----------------------------------------------------------------------
% Goal of the function :
% Configurate buzzer
% ----------------------------------------------------------------------
% Input(s) :
% const: struct containing constant of the experiment
% ----------------------------------------------------------------------
% Output(s):
% const: struct containing constant of the experiment
% ----------------------------------------------------------------------
% Function created by Martin SZINTE (martin.szinte@gmail.com)
% ----------------------------------------------------------------------

% Create session with NI devices
const.NiDaqDevice = daq.createSession('ni'); 

% Add one channel
addChannels(const.NiDaqDevice, const.buzzer_output_number, const.buzzer_daq_name);
aoCh = const.NiDaqDevice.Channels(const.buzzer_output_number);

% Create separate handles for analog and digital lines
const.aoTaskHandle = aoCh.TaskHandle;

% Configure handles (rate, number of scans)


% queue output data
switch const.cond1
    case 1
        NI_DAQmxCfgSampClkTiming(const.aoTaskHandle, const.NiDaqDevice.Rate, size(const.calib_signals, 1));
        NI_DAQmxWriteAnalogF64(const.aoTaskHandle, const.calib_signals);
    case 2
        NI_DAQmxCfgSampClkTiming(const.aoTaskHandle, const.NiDaqDevice.Rate, size(const.main_signals, 1));
        NI_DAQmxWriteAnalogF64(const.aoTaskHandle, const.main_signals);
    case 3
        NI_DAQmxCfgSampClkTiming(const.aoTaskHandle, const.NiDaqDevice.Rate, size(const.main_signals, 1));
        NI_DAQmxWriteAnalogF64(const.aoTaskHandle, const.main_signals);
end

end