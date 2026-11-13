#!/bin/sh
# module load Miniconda3/24.7.1-0
# conda deactivate
# source activate pydafiV3
SENpy_location='/home/wajordan/SENpy'
run_dir='/home/wajordan/SENSEI_CASES/parabola_test_param_sweep'
subdir_pattern1='*case_*'
subdir_pattern2='*'
error_file='error_norms.h5'
force_file='forces.h5'
verbose='True'
# output_hdf5_dict(dic,filename):
# parametric_refinement_forces(run_dir,subdir_pattern1="*",subdir_pattern2="*",verbose=True):
# parametric_refinement_error_norms(run_dir,subdir_pattern1="*",subdir_pattern2="*",verbose=True):
my_python_command1=\
"import sys; sys.path.append('$SENpy_location'); \
from error_norm_parsing import parametric_refinement_error_norms, output_hdf5_dict; \
DE = parametric_refinement_error_norms( '$run_dir', \
subdir_pattern1='$subdir_pattern1', \
subdir_pattern2='$subdir_pattern2', \
verbose=$verbose ); \
output_hdf5_dict(DE,'$error_file')"

my_python_command2=\
"import sys; sys.path.append('$SENpy_location'); \
from error_norm_parsing import parametric_refinement_forces, output_hdf5_dict; \
F = parametric_refinement_forces( '$run_dir', \
subdir_pattern1='$subdir_pattern1', \
subdir_pattern2='$subdir_pattern2', \
verbose=$verbose ); \
output_hdf5_dict(F,'$force_file')"

echo "$my_python_command1"
python3 -c "$my_python_command1"

echo "$my_python_command2"
python3 -c "$my_python_command2"