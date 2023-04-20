function [const] = runExp(const,expDes,my_key)
% ----------------------------------------------------------------------
% [const] = runExp(const,expDes,my_key)
% ----------------------------------------------------------------------
% Goal of the function :
% Launch experiement instructions and run trials
% ----------------------------------------------------------------------
% Input(s) :
% const : struct containing constant configurations
% expDes : struct containg experimental design
% my_key : structure containing keyboard configurations
% ----------------------------------------------------------------------
% Output(s):
% const : struct containing constant configurations
% ----------------------------------------------------------------------
% Function created by Martin SZINTE (martin.szinte@gmail.com)
% ----------------------------------------------------------------------

% Save all config at start of the block
% -------------------------------------
config.const = const;
config.expDes = expDes;
config.my_key = my_key;
save(const.mat_file,'config');

% First button config
% ------------------
if const.expStart
    for keyb = 1:size(my_key.keyboard_idx,2)
        KbQueueFlush(my_key.keyboard_idx(keyb));
    end
end
ListenChar(-1)

% Initial calibrations
% --------------------
% Task instructions 
fprintf(1,'\n\tTask instructions:');
switch const.cond1
    case 1; instr = const.calib_fix_order_instr;
    case 2; instr = const.main_lines_order_instr;
    case 3; instr = const.main_columns_order_instr;
end
fprintf(1,'\n\t%s',instr)
fprintf(1,'\n\tPress space to continue...')


push_button = 0;
while ~push_button    
    keyPressed = 0;
    keyCode = zeros(1,my_key.keyCodeNum);    
    for keyb = 1:size(my_key.keyboard_idx,2)
        [keyP, keyC] = KbQueueCheck(my_key.keyboard_idx(keyb));
        keyPressed = keyPressed+keyP;
        keyCode = keyCode+keyC;
    end

    if keyPressed
        if keyCode(my_key.space)
            push_button = 1;
        elseif keyCode(my_key.escape)
            if const.expStart == 0
                overDone(const,my_key)
            end
        end
    end
end
if const.scanner
    fprintf(1,'\n\tWaiting firt TR...\n')
end

% Clear button buffer
% -------------------
for keyb = 1:size(my_key.keyboard_idx,2)
    KbQueueFlush(my_key.keyboard_idx(keyb));
end

% Main trial loop
% ---------------
[expDes] = runTrials(const,expDes,my_key);

% Compute/Write mean/std behavioral data
% --------------------------------------
behav_txt_head{1}       =   'onset';                        behav_mat_res{1}        =   expDes.expMat(:,5);
behav_txt_head{2}       =   'duration';                     behav_mat_res{2}        = 	expDes.expMat(:,6)-expDes.expMat(:,5);
behav_txt_head{3}       =   'run_number';                   behav_mat_res{3}        = 	expDes.expMat(:,1);
behav_txt_head{4}       =   'trial_number';                 behav_mat_res{4}        = 	expDes.expMat(:,2);
behav_txt_head{5}       =   'task_condition';               behav_mat_res{5}        = 	expDes.expMat(:,3);
behav_txt_head{6}       =   'fix_direction';                behav_mat_res{6}        = 	expDes.expMat(:,4);

head_line               =   [];
for trial = 1:expDes.nb_trials
    % header line
    if trial == 1
        for tab = 1:size(behav_txt_head,2)
            if tab == size(behav_txt_head,2)
                head_line               =   [head_line,sprintf('%s',behav_txt_head{tab})];
            else
                head_line               =   [head_line,sprintf('%s\t',behav_txt_head{tab})];
            end
        end
        fprintf(const.behav_file_fid,'%s\n',head_line);
    end
    
	% trials line
    trial_line              =   [];
    for tab = 1:size(behav_mat_res,2)
        if tab == size(behav_mat_res,2)
            if isnan(behav_mat_res{tab}(trial))
                trial_line              =   [trial_line,sprintf('n/a')];
            else
                trial_line              =   [trial_line,sprintf('%1.10g',behav_mat_res{tab}(trial))];
            end
        else
            if isnan(behav_mat_res{tab}(trial))
                trial_line              =   [trial_line,sprintf('n/a\t')];
            else
                trial_line              =   [trial_line,sprintf('%1.10g\t',behav_mat_res{tab}(trial))];
            end
        end
    end
    fprintf(const.behav_file_fid,'%s\n',trial_line);
end

% End messages
% ------------
if const.runNum == size(const.cond_run_order,1)
    fprintf(1,'\n\n\tEnd of the experiment');
else
    fprintf(1,'\n\n\tEnd of the run');
end

% Save all config at the end of the block (overwrite start made at start)
% ---------------------------------------
config.const = const; config.expDes = expDes; config.my_key = my_key;
save(const.mat_file,'config');

end