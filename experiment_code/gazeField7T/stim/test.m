TR = 1.3;
freq = 120;
ampl = 2;
fix_trs = 9;%92;
buzz_trs = 1;
on = buzz_trs*TR;
off = (fix_trs-buzz_trs)*TR;
reps = 1;

signal = createSignal(freq, ampl, on, off, reps);
signals = [];
for gaze = 1:5
    signals = [signals;signal];
end

NiDaqDevice = daq.createSession('ni'); % Create session with NI devices
nrOutputs = 1;
daq1 = 'cDAQ1Mod2';
addChannels(NiDaqDevice, nrOutputs, daq1);
aoCh = NiDaqDevice.Channels(1);

% create separate handles for analog and digital lines
aoTaskHandle = aoCh.TaskHandle;

% Configure handles (rate, number of scans)
NI_DAQmxCfgSampClkTiming(aoTaskHandle, NiDaqDevice.Rate, size(signals, 1));

% queue output data
NI_DAQmxWriteAnalogF64(aoTaskHandle, signals);

% start the buzzer
NI_DAQmxStartTask(aoTaskHandle)
%NI_DAQmxStopTask(aoTaskHandle)