function main(const)
% ----------------------------------------------------------------------
% main(const)
% ----------------------------------------------------------------------
% Goal of the function :
% Launch all function of the experiment
% ----------------------------------------------------------------------
% Input(s) :
% const : struct containing a lot of constant configuration
% ----------------------------------------------------------------------
% Output(s):
% none
% ----------------------------------------------------------------------
% Function created by Martin SZINTE (martin.szinte@gmail.com)
% ----------------------------------------------------------------------

tic;

% File director
% -------------
[const] = dirSaveFile(const);

% Triggers and button configurations
% ----------------------------------
[my_key] = keyConfig;

% Experimental constant
% ---------------------
[const] = constConfig(const);

% Buzzer configuration
if const.buzzer
    [const] = buzzerConfig(const);
end

% Experimental design
% -------------------
[expDes] = designConfig(const);

% Trial runner
% ------------
[const] = runExp(const,expDes,my_key);

% End
% ---
overDone(const,my_key);

end