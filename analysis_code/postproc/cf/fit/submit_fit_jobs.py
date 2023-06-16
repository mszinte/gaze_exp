"""
-----------------------------------------------------------------------------------------
submit_fit_jobs.py
-----------------------------------------------------------------------------------------
Goal of the script:
Create jobscript to fit CF
-----------------------------------------------------------------------------------------
Input(s):
sys.argv[1]: main project directory
sys.argv[2]: project name (correspond to directory)
sys.argv[3]: subject name (e.g. sub-01)
sys.argv[4]: data permission group (e.g. 327)
sys.argv[5]: project name (e.g. b327)
-----------------------------------------------------------------------------------------
Output(s):
.sh file to execute in server
-----------------------------------------------------------------------------------------
To run:
1. cd to function
>> cd ~/projects/gaze_exp/analysis_code/postproc/
2. run python command
python submit_fit_jobs.py [main directory] [project name] [subject num] [group]
-----------------------------------------------------------------------------------------
Exemple:
python cf/fit/submit_fit_jobs.py /scratch/mszinte/data gaze_exp sub-001 327 b327
python cf/fit/submit_fit_jobs.py /scratch/mszinte/data gaze_exp sub-002 327 b327
-----------------------------------------------------------------------------------------
Written by Martin Szinte (mail@martinszinte.net)
-----------------------------------------------------------------------------------------
"""

# Stop warnings
import warnings
warnings.filterwarnings("ignore")

# General imports
import os
import sys
import ipdb
deb = ipdb.set_trace

# Inputs
main_dir = sys.argv[1]
project_dir = sys.argv[2]
subject = sys.argv[3]
group = sys.argv[4]
proj_name = sys.argv[5]

# Cluster settings
nb_procs = 1
memory_val = 100

# Define directories
pp_dir = "{}/{}/derivatives/pp_data".format(main_dir, project_dir)
cf_dir = "{}/{}/cf".format(pp_dir, subject)
os.makedirs(cf_dir, exist_ok=True)
cf_fit_dir = "{}/{}/cf/fit".format(pp_dir, subject)
os.makedirs(cf_fit_dir, exist_ok=True)
cf_jobs_dir = "{}/{}/cf/jobs".format(pp_dir, subject)
os.makedirs(cf_jobs_dir, exist_ok=True)
cf_logs_dir = "{}/{}/cf/log_outputs".format(pp_dir, subject)
os.makedirs(cf_logs_dir, exist_ok=True)

# define permission cmd
chmod_cmd = "\nchmod -Rf 771 {}/{}".format(main_dir, project_dir)
chgrp_cmd = "\nchgrp -Rf {} {}/{}".format(group, main_dir, project_dir)

# Across gaze directions
gaze_directions = ['center', 'left', 'right', 'up', 'down']
for gaze_direction in gaze_directions:

    # create job shell
    slurm_cmd = """\
#!/bin/bash
#SBATCH -p skylake
#SBATCH -A {proj_name}
#SBATCH --nodes=1
#SBATCH --cpus-per-task={nb_procs}
#SBATCH --time=1:00:00
#SBATCH --mem={memory_val}gb
#SBATCH -e {log_dir}/{sub}_fit_{gaze_dir}_%N_%j_%a.err
#SBATCH -o {log_dir}/{sub}_fit_{gaze_dir}_%N_%j_%a.out
#SBATCH -J {sub}_fit_{gaze_dir}\n\n""".format(
        nb_procs=nb_procs, log_dir=cf_logs_dir, memory_val=memory_val,
        sub=subject, gaze_dir=gaze_direction ,proj_name=proj_name)

    # define fit cmd
    fit_cmd = "python cf/fit/cf_fit.py {} {} {} {}".format(main_dir, project_dir, subject, gaze_direction)

    # create sh
    sh_fn = "{}/jobs/{}_cf_fit-{}.sh".format(cf_dir,subject,gaze_direction)
    of = open(sh_fn, 'w')
    of.write("{}{}{}{}".format(slurm_cmd, fit_cmd, chmod_cmd, chgrp_cmd))
    of.close()

    # Submit jobs
    print("Submitting {} to queue".format(sh_fn))
    os.system("sbatch {}".format(sh_fn))