function [expDes] = runTrials(const,expDes,my_key)
% ----------------------------------------------------------------------
% [expDes]=runTrials(const,expDes,my_key)
% ----------------------------------------------------------------------
% Goal of the function :
% Creat trials
% ----------------------------------------------------------------------
% Input(s) :
% const : struct containing constant configurations
% expDes : struct containg experimental design
% my_key : structure containing keyboard configurations
% ----------------------------------------------------------------------
% Output(s):
% resMat : experimental results (see below)
% expDes : struct containing all the variable design configurations.
% ----------------------------------------------------------------------
% Function created by Martin SZINTE (martin.szinte@gmail.com)
% ----------------------------------------------------------------------

% Cond1 : Task
cond1 = const.cond1;
if cond1 == 1
    fix_dur = const.calib_fix_dur;
else
    fix_dur = const.main_fix_dur;
end

first_trigger = 0;
trial = 1;
while trial <= expDes.nb_trials
        
    % Var 1 : Bar direction
    var1 = expDes.expMat(trial,4);
    
        
    % Wait first TR    
    while ~first_trigger
        if const.scanner == 0 || const.scannerTest
            first_trigger = 1;
            if const.buzzer 
                % run the buzzer thing
                NI_DAQmxStartTask(const.aoTaskHandle)
            end
        else
            keyPressed              =   0;
            keyCode                 =   zeros(1,my_key.keyCodeNum);
            for keyb = 1:size(my_key.keyboard_idx,2)
                [keyP, keyC]            =   KbQueueCheck(my_key.keyboard_idx(keyb));
                keyPressed              =   keyPressed+keyP;
                keyCode                 =   keyCode+keyC;
            end

            if keyPressed
                if keyCode(my_key.escape) && const.expStart == 0
                    overDone(const,my_key)
                elseif keyCode(my_key.tr)
                    first_trigger = 1;
                    if const.buzzer 
                        % run the buzzer thing
                        NI_DAQmxStartTask(const.aoTaskHandle)
                    end
                end
            end
        end
        if first_trigger
            trial_start = GetSecs;
            log_txt  =   sprintf('\ntrial_start %i at %i',trial,trial_start);
            fprintf(const.log_file_fid,log_txt);
            expDes.expMat(trial,5) = trial_start;
            if const.checkTrial
                fprintf(1,'\n\n\t========================  TRIAL %1.0f ========================\n', trial);
                fprintf(1,'\n\tCondition             =    %s',expDes.txt_cond1{cond1});
                if cond1 == 1
                    fprintf(1,'\n\tFixation direction    =    %s',expDes.txt_var1_calib{var1});
                else
                    fprintf(1,'\n\tFixation direction    =    %s',expDes.txt_var1{var1});
                end
            end
        end
    end
    
    % Clock timer across trial
    t_clock = GetSecs;
    if (t_clock-trial_start) >= fix_dur
        expDes.expMat(trial,6) = t_clock; % previous trial offset
        log_txt  =   sprintf('\ntrial_end %i at %i',trial,t_clock);
        fprintf(const.log_file_fid,log_txt);
        
        trial = trial +1;
        if trial <= expDes.nb_trials
            % log
            trial_start = t_clock;
            log_txt  =   sprintf('\ntrial_start %i at %i',trial,trial_start);
            fprintf(const.log_file_fid,log_txt);
            expDes.expMat(trial,5) = trial_start;

            % Var 1 : Bar direction
            var1 = expDes.expMat(trial,4);
            if const.checkTrial
                fprintf(1,'\n\n\t========================  TRIAL %1.0f ========================\n', trial);
                fprintf(1,'\n\tCondition             =    %s',expDes.txt_cond1{cond1});
                if cond1 == 1
                    fprintf(1,'\n\tFixation direction    =    %s',expDes.txt_var1_calib{var1});
                else
                    fprintf(1,'\n\tFixation direction    =    %s',expDes.txt_var1{var1});
                end
            end
        end
    end 

end