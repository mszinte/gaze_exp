function [const]=sbjConfig(const)
% ----------------------------------------------------------------------
% [const]=sbjConfig(const)
% ----------------------------------------------------------------------
% Goal of the function :
% Define subject configurations (initials, gender...)
% ----------------------------------------------------------------------
% Input(s) :
% const : struct containing constant configurations
% ----------------------------------------------------------------------
% Output(s):
% const : struct containing constant configurations
% ----------------------------------------------------------------------
% Function created by Martin SZINTE (martin.szinte@gmail.com)
% ----------------------------------------------------------------------

if const.expStart
    const.sjctNum           =  input(sprintf('\n\tParticipant number: '));
    if isempty(const.sjctNum)
        error('Incorrect participant number');
    end
    if const.sjctNum > 9
        const.sjct              =  sprintf('sub-0%i',const.sjctNum);
    else
        const.sjct              =  sprintf('sub-00%i',const.sjctNum);
    end
end


const.sesNum = input(sprintf('\n\tSession number: '));
if const.sesNum > 9
    const.session           =  sprintf('ses-%i',const.sesNum);
else
    const.session           =  sprintf('ses-0%i',const.sesNum);
end

const.runNum            =   input(sprintf('\n\tRun number: '));
if isempty(const.runNum)
    error('Incorrect run number');
end

if const.cond_run_num(const.runNum) > 9
    const.run_txt   =  sprintf('run-%i',const.cond_run_num(const.runNum));
else
    const.run_txt   =  sprintf('run-0%i',const.cond_run_num(const.runNum));
end

if const.expStart == 0
    const.cond1         =   1;
else
    const.cond1         =   const.cond_run_order(const.runNum,1);
end

switch const.cond1
    case 1; const.cond1_txt = 'Calib';
    case 2; const.cond1_txt = 'GazeLines';
    case 3; const.cond1_txt = 'GazeColumns';
end

fprintf(1,'\n\tTask: %s\n',const.cond1_txt);

if ~const.expStart
    const.sjct          =   'sub-0X';
end

end