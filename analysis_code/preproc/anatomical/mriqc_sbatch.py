"""
-----------------------------------------------------------------------------------------
mriqc_sbatch.py
-----------------------------------------------------------------------------------------
Goal of the script:
Run frmiqc on mesocentre using job mode
-----------------------------------------------------------------------------------------
Input(s):
sys.argv[1]: main project directory
sys.argv[2]: project name (correspond to directory)
sys.argv[3]: bids subject name (e.g. sub-01)
sys.argv[5]: server nb of hour to request (e.g 10)
sys.argv[6]: email account
sys.argv[7]: data group (e.g. 327)
sys.argv[8]: project name (e.g. b327)
-----------------------------------------------------------------------------------------
Output(s):
QC html files
-----------------------------------------------------------------------------------------
To run:
1. cd to function
>> cd ~/projects/gaze_exp/analysis_code/preproc/anatomical
2. run python command
python mriqc_sbatch.py [main directory] [project name] [subject name] 
                       [nb proc.] [hour proc.] [email] [group] [project name)
-----------------------------------------------------------------------------------------
Exemple:
python mriqc_sbatch.py /scratch/mszinte/data gaze_exp sub-001 20 martin.szinte 327 b327
-----------------------------------------------------------------------------------------
Written by Martin Szinte (martin.szinte@gmail.com)
-----------------------------------------------------------------------------------------
"""

# imports modules
import sys
import os
import time
import ipdb

# inputs and settings
main_dir = sys.argv[1]
project_dir = sys.argv[2]
subject = sys.argv[3]
sub_num = subject[-3:]
job_dur = int(sys.argv[4])
email = sys.argv[5]
group = sys.argv[6]
proj_name = sys.argv[7]
nb_procs = 32
memory_val = 100
singularity_dir = "{}/{}/code/singularity/mriqc-0.15.1.simg".format(main_dir, project_dir)
log_dir = "{}/{}/derivatives/mriqc/log_outputs".format(main_dir, project_dir)
jobs_dir = "{}/{}/derivatives/mriqc/jobs".format(main_dir, project_dir)

# create job and log output folders
os.makedirs(jobs_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

# define SLURM command
slurm_cmd = """\
#!/bin/bash
#SBATCH --mail-user={email}
#SBATCH -p skylake
#SBATCH -A {proj_name}
#SBATCH --nodes=1
#SBATCH --mem=100gb
#SBATCH --cpus-per-task={nb_procs}
#SBATCH --time={job_dur}:00:00
#SBATCH -e {log_dir}/{subject}_mriqc_%N_%j_%a.err
#SBATCH -o {log_dir}/{subject}_mriqc_%N_%j_%a.out
#SBATCH -J {subject}_mriqc
#SBATCH --mail-type=BEGIN,END\n\n""".format(email=email, 
                    proj_name=proj_name, 
                    nb_procs=nb_procs, log_dir=log_dir,
                    job_dur=job_dur, subject=subject)

# define singularity cmd
singularity_cmd = "singularity run --bind {main_dir}:/work_dir {simg} /work_dir/{project_dir}/ /work_dir/{project_dir}/derivatives/mriqc/ participant --participant_label {sub_num} --verbose-reports --mem_gb {memory_val} -m bold T1w T2w --no-sub".format(main_dir=main_dir, project_dir=project_dir, simg=singularity_dir, sub_num=sub_num, memory_val=memory_val)

# define permission cmd
chmod_cmd = "\nchmod -Rf 771 {}/{}".format(main_dir, project_dir)
chgrp_cmd = "\nchgrp -Rf {} {}/{}".format(group, main_dir, project_dir)

sh_fn = "{}/{}/derivatives/mriqc/jobs/{}_mriqc.sh".format(main_dir,project_dir, subject)
of = open(sh_fn, 'w')
of.write("{}{}{}{}".format(slurm_cmd, singularity_cmd, chmod_cmd, chgrp_cmd))
of.close()

# Submit jobs
print("Submitting {} to queue".format(sh_fn))
os.chdir(log_dir)
os.system("sbatch {}".format(sh_fn))
