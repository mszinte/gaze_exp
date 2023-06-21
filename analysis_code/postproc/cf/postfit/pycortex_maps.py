"""
-----------------------------------------------------------------------------------------
pycortex_maps.py
-----------------------------------------------------------------------------------------
Goal of the script:
Create flatmap plots and dataset
-----------------------------------------------------------------------------------------
Input(s):
sys.argv[1]: main project directory
sys.argv[2]: project name (correspond to directory)
sys.argv[3]: subject name (e.g. sub-01)
sys.argv[4]: save map in svg (y/n)
-----------------------------------------------------------------------------------------
Output(s):
Pycortex flatmaps figures
-----------------------------------------------------------------------------------------
To run:
0. TO RUN ON INVIBE SERVER (with Inkscape)
1. cd to function
>> cd ~/disks/meso_H/projects/gaze_exp/analysis_code/postproc/cf/postfit/
2. run python command
>> python pycortex_maps.py [main directory] [project name] [subject num] [gaze_direction]
-----------------------------------------------------------------------------------------
Exemple:
python pycortex_maps.py ~/disks/meso_S/data gaze_exp sub-001 center
python pycortex_maps.py ~/disks/meso_S/data gaze_exp sub-001 left
python pycortex_maps.py ~/disks/meso_S/data gaze_exp sub-001 right
python pycortex_maps.py ~/disks/meso_S/data gaze_exp sub-001 up
python pycortex_maps.py ~/disks/meso_S/data gaze_exp sub-001 down
-----------------------------------------------------------------------------------------
Written by Martin Szinte (mail@martinszinte.net)
-----------------------------------------------------------------------------------------
"""

# Stop warnings
import warnings
warnings.filterwarnings("ignore")

# General imports
import cortex
import importlib
import ipdb
import json
import matplotlib.pyplot as plt
import nibabel as nb
import numpy as np
import os
import sys
sys.path.append("{}/../../../utils".format(os.getcwd()))
from pycortex_utils import draw_cortex, set_pycortex_config_file
deb = ipdb.set_trace

# Inputs
main_dir = sys.argv[1]
project_dir = sys.argv[2]
subject = sys.argv[3]
gaze_direction = sys.argv[4]

# Define analysis parameters
with open('../../../settings.json') as f:
    json_s = f.read()
    analysis_info = json.loads(json_s)
xfm_name = analysis_info["xfm_name"]
task = "GazeCWCCW-{}".format(gaze_direction)
high_pass_type = analysis_info['high_pass_type']
    
# Define directories and fn
cortex_dir = "{}/{}/derivatives/pp_data/cortex".format(main_dir, project_dir)
cf_fit_dir = "{}/{}/derivatives/pp_data/{}/cf/fit".format(main_dir, project_dir, subject)
flatmaps_loo_avg_dir = '{}/{}/derivatives/pp_data/{}/cf/pycortex/flatmaps_loo_avg'.format(main_dir, project_dir, subject)
datasets_loo_avg_dir = '{}/{}/derivatives/pp_data/{}/cf/pycortex/datasets_loo_avg'.format(main_dir, project_dir, subject)
os.makedirs(flatmaps_loo_avg_dir, exist_ok=True)
os.makedirs(datasets_loo_avg_dir, exist_ok=True)

deriv_prf_surf_fn = "{}/{}_task-prf_fmriprep_{}_bold_loo_avg_prf-deriv.gii".format(cf_fit_dir, subject, high_pass_type)
deriv_cf_surf_fn = "{}/{}_task-{}_fmriprep_{}_bold_loo_avg_CFprf-deriv.gii".format(cf_fit_dir, subject, task, high_pass_type)
deriv_fn_labels = ['cf_loo_avg']

# Set pycortex db and colormaps
set_pycortex_config_file(cortex_dir)
importlib.reload(cortex)

# Maps settings
rsq_idx, rsq_loo_idx, ecc_idx, polar_real_idx, polar_imag_idx , size_idx, \
    amp_idx, baseline_idx, x_idx, y_idx, cf_size_idx, cf_rsq_idx = 0,1,2,3,4,5,6,7,8,9,11,12
cmap_polar, cmap_uni, cmap_ecc_size = 'hsv', 'Reds', 'Spectral'
col_offset = 1.0/14.0
cmap_steps = 255

# plot scales
rsq_scale = [0, 0.6]
ecc_scale = [0, 10]
size_scale = [0, 10]
cf_size_scale = [0, 3]

print('Creating flatmaps...')

deriv_fn_label = 'cf_loo_avg'
save_svg = False
description_end = ' (CF leave-one-out fit)'
flatmaps_dir = flatmaps_loo_avg_dir
datasets_dir = datasets_loo_avg_dir
maps_names = []

# load data
prf_deriv_mat = nb.load(deriv_prf_surf_fn).agg_data()
cf_deriv_mat = nb.load(deriv_cf_surf_fn).agg_data()
    
# threshold data
prf_deriv_mat_th = prf_deriv_mat
rsqr_th_down = prf_deriv_mat_th[rsq_loo_idx,:] >= analysis_info['rsqr_th'][0]
rsqr_th_up = prf_deriv_mat_th[rsq_loo_idx,:] <= analysis_info['rsqr_th'][1]
prf_all_th = np.array((rsqr_th_down,rsqr_th_up)) 
prf_deriv_mat[rsq_loo_idx, np.logical_and.reduce(prf_all_th)==False]=0

cf_deriv_mat_th = cf_deriv_mat
amp_down =  cf_deriv_mat_th[amp_idx,:] > 0
size_th_down = cf_deriv_mat_th[size_idx,:] >= analysis_info['size_th'][0]
size_th_up = cf_deriv_mat_th[size_idx,:] <= analysis_info['size_th'][1]
ecc_th_down = cf_deriv_mat_th[ecc_idx,:] >= analysis_info['ecc_th'][0]
ecc_th_up = cf_deriv_mat_th[ecc_idx,:] <= analysis_info['ecc_th'][1]
cf_all_th = np.array((amp_down,size_th_down,size_th_up,ecc_th_down,ecc_th_up)) 
cf_deriv_mat[rsq_loo_idx, np.logical_and.reduce(cf_all_th)==False]=0

# compute alpha
alpha_data = prf_deriv_mat[rsq_loo_idx,:]
alpha_range = analysis_info["cf_alpha_range"]
alpha = (alpha_data - alpha_range[0])/(alpha_range[1]-alpha_range[0])
alpha[alpha>1]=1

# r-square
rsq_data = cf_deriv_mat[cf_rsq_idx,:]
param_rsq = {'data': rsq_data, 'cmap': cmap_uni, 'alpha': alpha, 
             'vmin': rsq_scale[0], 'vmax': rsq_scale[1], 'cbar': 'discrete', 
             'cortex_type': 'VertexRGB','description': '{} CF R2'.format(task),
             'curv_brightness': 1, 'curv_contrast': 0.1, 'add_roi': save_svg,
             'cbar_label': 'pRF R2', 'with_labels': True}
maps_names.append('rsq')

# polar angle
pol_comp_num = cf_deriv_mat[polar_real_idx,:] + 1j * cf_deriv_mat[polar_imag_idx,:]
polar_ang = np.angle(pol_comp_num)
ang_norm = (polar_ang + np.pi) / (np.pi * 2.0)
ang_norm = np.fmod(ang_norm + col_offset,1)
param_polar = {'data': ang_norm, 'cmap': cmap_polar, 'alpha': alpha, 
               'vmin': 0, 'vmax': 1, 'cmap_steps': cmap_steps, 'cortex_type': 'VertexRGB',
               'cbar': 'polar', 'col_offset': col_offset, 
               'description': '{} polar:{:3.0f} steps{}'.format(task, cmap_steps, description_end), 
               'curv_brightness': 0.1, 'curv_contrast': 0.25, 'add_roi': save_svg, 
               'with_labels': True}
exec('param_polar_{cmap_steps} = param_polar'.format(cmap_steps = int(cmap_steps)))
exec('maps_names.append("polar_{cmap_steps}")'.format(cmap_steps = int(cmap_steps)))

# eccentricity
ecc_data = cf_deriv_mat[ecc_idx,:]
param_ecc = {'data': ecc_data, 'cmap': cmap_ecc_size, 'alpha': alpha,
             'vmin': ecc_scale[0], 'vmax': ecc_scale[1], 'cbar': 'ecc', 'cortex_type': 'VertexRGB',
             'description': '{} pRF eccentricity{}'.format(task,description_end), 'curv_brightness': 1,
             'curv_contrast': 0.1, 'add_roi': save_svg, 'with_labels': True}
maps_names.append('ecc')

# size
size_data = cf_deriv_mat[size_idx,:]
param_size = {'data': size_data, 'cmap': cmap_ecc_size, 'alpha': alpha, 
              'vmin': size_scale[0], 'vmax': size_scale[1], 'cbar': 'discrete', 
              'cortex_type': 'VertexRGB', 'description': '{} pRF size{}'.format(task, description_end), 
              'curv_brightness': 1, 'curv_contrast': 0.1, 'add_roi': False, 'cbar_label': 'pRF size (dva)',
              'with_labels': True}
maps_names.append('size')

# CF size
cf_size_data = cf_deriv_mat[cf_size_idx,:]
param_cf_size = {'data': cf_size_data, 'cmap': cmap_ecc_size, 'alpha': alpha, 
                  'vmin': cf_size_scale[0], 'vmax': cf_size_scale[1], 'cbar': 'discrete', 
                  'cortex_type': 'VertexRGB', 'description': '{} CF size{}'.format(task, description_end), 
                  'curv_brightness': 1, 'curv_contrast': 0.1, 'add_roi': False, 'cbar_label': 'CF size (mm)',
                  'with_labels': True}
maps_names.append('cf_size')

# draw flatmaps
volumes = {}
for maps_name in maps_names:

    # create flatmap
    roi_name = '{}_{}'.format(task, maps_name)
    roi_param = {'subject': subject, 'xfmname': xfm_name, 'roi_name': roi_name}
    print(roi_name)
    exec('param_{}.update(roi_param)'.format(maps_name))
    exec('volume_{maps_name} = draw_cortex(**param_{maps_name})'.format(maps_name = maps_name))
    exec("plt.savefig('{}/{}_task-{}_{}_{}.pdf')".format(flatmaps_dir, subject, task,  maps_name, deriv_fn_label))
    plt.close()

    # save flatmap as dataset
    exec('vol_description = param_{}["description"]'.format(maps_name))
    exec('volume = volume_{}'.format(maps_name))
    volumes.update({vol_description:volume})

# save dataset
dataset_file = "{}/{}_task-{}_{}.hdf".format(datasets_dir, subject, task, deriv_fn_label)
dataset = cortex.Dataset(data=volumes)
dataset.save(dataset_file)
