function overDone(const,my_key)
% ----------------------------------------------------------------------
% overDone(const,my_key)
% ----------------------------------------------------------------------
% Goal of the function :
% Close screen and audio, transfer eye-link data, close files
% ----------------------------------------------------------------------
% Input(s) :
% const : struct containing constant configurations
% my_key : structure containing keyboard configurations
% ----------------------------------------------------------------------
% Output(s):
% none
% ----------------------------------------------------------------------
% Function created by Martin SZINTE (martin.szinte@gmail.com)
% ----------------------------------------------------------------------

% Stop recording the keyboard
% ---------------------------
for keyb = 1:size(my_key.keyboard_idx,2)
    KbQueueStop(my_key.keyboard_idx(keyb));
    KbQueueFlush(my_key.keyboard_idx(keyb));
end

% Close all fid
% ------------- 
fclose(const.behav_file_fid);
fclose(const.log_file_fid);

% Print block duration
% --------------------
timeDur=toc/60;
fprintf(1,'\n\tThis part of the experiment took : %2.0f min.\n\n',timeDur);

end