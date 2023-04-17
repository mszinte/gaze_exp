#/usr/bin/env python
# convert all parrec files in a directory to nifti

import argparse
from glob import glob
import os
import sys
import ipdb
import nibabel as nib


def arg_parser():
    parser = argparse.ArgumentParser(description='convert all parrec images in a directory to nifti')
    parser.add_argument('img_dir', type=str)
    parser.add_argument('out_dir', type=str)
    parser.add_argument('phase_included', type=int)
    return parser


def split_filename(filepath):
    path = os.path.dirname(filepath)
    filename = os.path.basename(filepath)
    base, ext = os.path.splitext(filename)
    if ext == '.gz':
        base, ext2 = os.path.splitext(base)
        ext = ext2 + ext
    return path, base, ext


def main():
    try:
        args = arg_parser().parse_args()
        fns = glob(os.path.join(args.img_dir, '*.par'))
        
        if len(fns) == 0:
            fns = glob(os.path.join(args.img_dir, '*.PAR'))
        if len(fns) == 0:
            raise Exception('Could not find .par files in {}'.format(args.img_dir))
        for fn in fns:

            print('Converting image: {}'.format(fn))
            img = nib.load(fn)
            _, base, _ = split_filename(fn)

            out_fn = os.path.join(args.out_dir, base + '.nii.gz')
            nifti = nib.Nifti1Image(img.dataobj, img.affine, header=img.header)
            nifti.set_data_dtype('<f4')
            
            
            if args.phase_included == 1:
                phase_out_fn = os.path.join(args.out_dir, base + '_phase.nii.gz')
                nifti_mat = nifti.get_fdata()
                nifti_bold = nifti_mat[...,:int(nifti_mat.shape[3]/2)]
                nifti_phase = nifti_mat[...,int(nifti_mat.shape[3]/2):]
                nib.Nifti1Image(nifti_phase, nifti.affine).to_filename(phase_out_fn)
                print('Saved to: {}'.format(phase_out_fn))
                out_fn = os.path.join(args.out_dir, base + '_magnitude.nii.gz')
            else:
                nifti_mat = nifti.get_fdata()
                nifti_bold = nifti_mat
                
            nib.Nifti1Image(nifti_bold, nifti.affine).to_filename(out_fn)

            print('Saved to: {}'.format(out_fn))
        return 0
    except Exception as e:
        print(e)
        return 1

if __name__ == "__main__":
    sys.exit(main())