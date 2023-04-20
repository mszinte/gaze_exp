Node: funcMRIQC (ComputeIQMs (provenance (utility)
==================================================


 Hierarchy : workflow_enumerator.funcMRIQC.ComputeIQMs.provenance
 Exec ID : provenance.a20


Original Inputs
---------------


* function_str : def _add_provenance(in_file, settings):
    from mriqc import __version__ as version
    from nipype.utils.filemanip import hash_infile
    out_prov = {
        'md5sum': hash_infile(in_file),
        'version': version,
        'software': 'mriqc',
        'webapi_url': settings.pop('webapi_url'),
        'webapi_port': settings.pop('webapi_port'),
    }

    if settings:
        out_prov['settings'] = settings

    return out_prov

* in_file : /work_dir/gaze_exp/sub-001/ses-05/func/sub-001_ses-05_task-GazeLines_run-03_bold.nii.gz
* settings : {'fd_thres': 0.2, 'hmc_fsl': False, 'webapi_url': 'https://mriqc.nimh.nih.gov/api/v1', 'webapi_port': None}

