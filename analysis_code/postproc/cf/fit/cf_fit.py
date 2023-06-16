"""
-----------------------------------------------------------------------------------------
cf_fit.py
-----------------------------------------------------------------------------------------
Goal of the script:
CF fit code run by submit_fit.py
-----------------------------------------------------------------------------------------
Input(s):
sys.argv[1]: main project directory
sys.argv[2]: project name (correspond to directory)
sys.argv[3]: subject (e.g. sub-01)
sys.argv[4]: gaze_direction (center/left/right/up/down)
-----------------------------------------------------------------------------------------
Output(s):
Nifti image files with fit parameters for a z slice
-----------------------------------------------------------------------------------------
To run :
>> cd to function directory
cd ~/projects/gaze_exp/analysis_code/postproc/
>> python cf/fit/cf_fit.py [main_dir] [project] [subject] [gaze_direction]
-----------------------------------------------------------------------------------------
Written by Martin Szinte (martin.szinte@gmail.com)
-----------------------------------------------------------------------------------------
"""

# Stop warnings
import warnings
warnings.filterwarnings("ignore")

# General imports
import sys, os
import numpy as np
import glob
import datetime
import json
import importlib
import ipdb
deb = ipdb.set_trace

# MRI analysis imports
import nibabel as nb
import cortex
sys.path.append("{}/../utils".format(os.getcwd()))
from cf_utils import *
from pycortex_utils import set_pycortex_config_file

# Get inputs
start_time = datetime.datetime.now()
main_dir = sys.argv[1]
project_dir = sys.argv[2]
subject = sys.argv[3]
gaze_direction = sys.argv[4]

# Analysis parameters
with open('../settings.json') as f:
    json_s = f.read()
    analysis_info = json.loads(json_s)
TR = analysis_info['TR']
grid_cf_nr = analysis_info['grid_cf_nr']
max_cf_size = analysis_info['max_cf_size']
xfm_name = analysis_info['xfm_name']
high_pass_type = analysis_info['high_pass_type']

# Define directories
cortex_dir = "{}/{}/derivatives/pp_data/cortex".format(main_dir, project_dir)
avg_dir = "{}/{}/derivatives/pp_data/{}/func/fmriprep_{}_avg".format(main_dir, project_dir, subject, high_pass_type)
prf_fit_dir = "{}/{}/derivatives/pp_data/{}/prf/fit".format(main_dir, project_dir, subject)
cf_fit_dir = "{}/{}/derivatives/pp_data/{}/cf/fit".format(main_dir, project_dir, subject)
os.makedirs(cf_fit_dir, exist_ok=True)

# Set pycortex db and colormaps
set_pycortex_config_file(cortex_dir)
importlib.reload(cortex)

# Define data filenames
gaze_cw_fn = "{}/{}_task-GazeCW_fmriprep_{}_bold_avg.nii.gz".format(avg_dir, subject, high_pass_type)
gaze_ccw_fn = "{}/{}_task-GazeCCW_fmriprep_{}_bold_avg.nii.gz".format(avg_dir, subject, high_pass_type)
deriv_avg_loo_fn = "{}/{}_task-prf_fmriprep_{}_bold_loo_avg_prf-deriv.nii.gz".format(prf_fit_dir, subject, high_pass_type)
deriv_cf_fn = "{}/{}_task-GazeCWCCW-{}_fmriprep_{}_bold_loo_avg_CFprf-deriv.nii.gz.nii.gz".format(cf_fit_dir, subject, gaze_direction, high_pass_type)

# Load data in volumes
gaze_cw_vol = nb.load(gaze_cw_fn)
gaze_ccw_vol = nb.load(gaze_ccw_fn)

# Convert volumes to surfaces
# Average data of CW and CCW for each gaze directions
run_trs = [[3, 70], [71, 138], [139, 206], [207, 274], [275, 342]]
cw_center_tr, ccw_center_tr = run_trs[0], run_trs[4]
cw_left_tr, ccw_left_tr = run_trs[1], run_trs[3]
cw_up_tr, ccw_up_tr = run_trs[2], run_trs[2]
cw_right_tr, ccw_right_tr = run_trs[3], run_trs[1]
cw_down_tr, ccw_down_tr = run_trs[4], run_trs[0]

mapper = cortex.get_mapper(subject, xfm_name, 'line_nearest', recache=True)
print("load {}".format(gaze_cw_fn))
gaze_cw_surf = mapper(cortex.Volume(gaze_cw_vol.get_fdata().T, subject, xfm_name)).data
exec("gaze_{gaze_dir}_cw_surf = gaze_cw_surf[cw_{gaze_dir}_tr[0]:cw_{gaze_dir}_tr[1]]".format(gaze_dir = gaze_direction))
exec("del gaze_cw_surf")
print("end loading 1")

print("load {}".format(gaze_ccw_fn))
gaze_ccw_surf = mapper(cortex.Volume(gaze_ccw_vol.get_fdata().T, subject, xfm_name)).data
exec("gaze_{gaze_dir}_ccw_surf = gaze_ccw_surf[ccw_{gaze_dir}_tr[0]:ccw_{gaze_dir}_tr[1]]".format(gaze_dir = gaze_direction))
exec("del gaze_ccw_surf")
print("end loading 2")

exec("train_data = np.nanmean((gaze_{gaze_dir}_cw_surf, gaze_{gaze_dir}_ccw_surf),axis=0)".format(gaze_dir = gaze_direction))
exec("del gaze_{gaze_dir}_cw_surf".format(gaze_dir = gaze_direction))
exec("del gaze_{gaze_dir}_ccw_surf".format(gaze_dir = gaze_direction))
print("end averaging")

# Define source region for CF models
surfs = [cortex.polyutils.Surface(*d) for d in cortex.db.get_surf(subject, "flat")]
surf_lh, surf_rh = surfs[0], surfs[1]
lh_vert_num, rh_vert_num = surf_lh.pts.shape[0], surf_rh.pts.shape[0]
vert_num = lh_vert_num + rh_vert_num
source_roi = 'V1'
roi_verts = cortex.get_roi_verts(subject, mask=True)
maskL = roi_verts[source_roi][:lh_vert_num]
maskR = roi_verts[source_roi][lh_vert_num:]
myv1surf = subsurface(subject,[maskL,maskR])
myv1surf.create()

# Make a connective field 'stimulus'
train_stim = CFStimulus(data=train_data.T, vertinds=myv1surf.subsurface_verts, distances=myv1surf.distance_matrix)

# Define model
model = CFGaussianModel(train_stim)

# Define fitter and perform 'quick' grids using the CF sizes options
print("{} CF quick grid fit with CF size grid of {} elements".format(gaze_direction, grid_cf_nr))
gf = CFFitter(data=train_data.T, model=model)
exec("del train_data")
cf_sizes = max_cf_size*np.linspace(0.1,1,grid_cf_nr)**2
gf.quick_grid_fit(cf_sizes)

# Load pRF paramameters for projection
print("Load PRF data")
deriv_img = nb.load(deriv_avg_loo_fn)
deriv_mat = deriv_img.get_fdata()

# Convert pRF parameters in vertices
params_num = 10
rsq_idx, rsq_loo_idx, ecc_idx, polar_real_idx, polar_imag_idx , size_idx, \
    amp_idx, baseline_idx, x_idx, y_idx = 0,1,2,3,4,5,6,7,8,9
vert_rsq_data = mapper(cortex.Volume(deriv_mat[...,rsq_idx].T, subject, xfm_name)).data
vert_rsq_loo_data = mapper(cortex.Volume(deriv_mat[...,rsq_loo_idx].T, subject, xfm_name)).data
vert_ecc_data = mapper(cortex.Volume(deriv_mat[...,ecc_idx].T, subject, xfm_name)).data
vert_polar_real_data = mapper(cortex.Volume(deriv_mat[...,polar_real_idx].T, subject, xfm_name)).data
vert_polar_imag_data = mapper(cortex.Volume(deriv_mat[...,polar_imag_idx].T, subject, xfm_name)).data
vert_size_data = mapper(cortex.Volume(deriv_mat[...,size_idx].T, subject, xfm_name)).data
vert_amp_data = mapper(cortex.Volume(deriv_mat[...,amp_idx].T, subject, xfm_name)).data
vert_baseline_data = mapper(cortex.Volume(deriv_mat[...,baseline_idx].T, subject, xfm_name)).data
vert_x_data = mapper(cortex.Volume(deriv_mat[...,x_idx].T, subject, xfm_name)).data
vert_y_data = mapper(cortex.Volume(deriv_mat[...,y_idx].T, subject, xfm_name)).data

# CF projected parameters on surface
vert_cf_rsq_data = vert_rsq_data[gf.quick_vertex_centres]
vert_cf_rsq_loo_data = vert_rsq_loo_data[gf.quick_vertex_centres]
vert_cf_ecc_data = vert_ecc_data[gf.quick_vertex_centres]
vert_cf_polar_real_data = vert_polar_real_data[gf.quick_vertex_centres]
vert_cf_polar_imag_data = vert_polar_imag_data[gf.quick_vertex_centres]
vert_cf_size_data = vert_size_data[gf.quick_vertex_centres]
vert_cf_amp_data = vert_amp_data[gf.quick_vertex_centres]
vert_cf_baseline_data = vert_baseline_data[gf.quick_vertex_centres]
vert_cf_x_data = vert_x_data[gf.quick_vertex_centres]
vert_cf_y_data = vert_y_data[gf.quick_vertex_centres]

# Sava in pRF derivative file projected back to volume
deriv_cf_mat = np.zeros(deriv_img.shape)
deriv_cf_mat[...,rsq_idx] = mapper.backwards(vert_cf_rsq_data).T
deriv_cf_mat[...,rsq_loo_idx] = mapper.backwards(vert_cf_rsq_loo_data).T
deriv_cf_mat[...,ecc_idx] = mapper.backwards(vert_cf_ecc_data).T
deriv_cf_mat[...,polar_real_idx] = mapper.backwards(vert_cf_polar_real_data).T
deriv_cf_mat[...,polar_imag_idx] = mapper.backwards(vert_cf_polar_imag_data).T
deriv_cf_mat[...,size_idx] = mapper.backwards(vert_cf_size_data).T
deriv_cf_mat[...,amp_idx] = mapper.backwards(vert_cf_amp_data).T
deriv_cf_mat[...,baseline_idx] = mapper.backwards(vert_cf_baseline_data).T
deriv_cf_mat[...,x_idx] = mapper.backwards(vert_cf_x_data).T
deriv_cf_mat[...,y_idx] = mapper.backwards(vert_cf_y_data).T

deriv_cf_img = nb.Nifti1Image(dataobj=deriv_cf_mat, affine=deriv_img.affine, header=deriv_img.header)
deriv_cf_img.to_filename(deriv_cf_fn)

# Print duration
end_time = datetime.datetime.now()
print("\nStart time:\t{start_time}\nEnd time:\t{end_time}\nDuration:\t{dur}".format(
    start_time=start_time, end_time=end_time, dur=end_time - start_time))