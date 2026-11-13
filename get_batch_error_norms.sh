#!/bin/sh
# module load Miniconda3/24.7.1-0
# source activate pydafiV3
SENpy_location='/home/wajordan/SENpy'
run_dir='/home/wajordan/SENSEI_CASES/parabola_test_param_sweep'
subdir_pattern1='*case_*'
subdir_pattern2='*'
subdir_exclude1='[]'
subdir_exclude2='[]'
error_file='error_norms.h5'
force_file='forces.h5'
force_error_file='force_errors.h5'
verbose='True'
Cx='None'
Cy='None'
Cx_p='None'
Cy_p='None'
Axial_imin='None'
Axial_imax='None'
Axial_jmin='None'
Axial_jmax='None'
Normal_imin='None'
Normal_imax='None'
Normal_jmin='None'
Normal_jmax='None'

my_python_command1=\
"import sys; sys.path.append('$SENpy_location'); \
from error_norm_parsing import parametric_refinement_error_norms, output_hdf5_dict; \
DE = parametric_refinement_error_norms( '$run_dir', \
subdir_pattern1='$subdir_pattern1', \
subdir_pattern2='$subdir_pattern2', \
subdir_exclude1=$subdir_exclude1, \
subdir_exclude2=$subdir_exclude2, \
verbose=$verbose ); \
output_hdf5_dict(DE,'$error_file')"

my_python_command2=\
"import sys; sys.path.append('$SENpy_location'); \
from error_norm_parsing import parametric_refinement_forces, output_hdf5_dict; \
F = parametric_refinement_forces( '$run_dir', \
subdir_pattern1='$subdir_pattern1', \
subdir_pattern2='$subdir_pattern2', \
subdir_exclude1=$subdir_exclude1, \
subdir_exclude2=$subdir_exclude2, \
verbose=$verbose ); \
output_hdf5_dict(F,'$force_file')"

my_python_command3=\
"import sys; sys.path.append('$SENpy_location'); \
from error_norm_parsing import parametric_refinement_force_errors, read_hdf5_dict, output_hdf5_dict; \
F = read_hdf5_dict('$run_dir$force_file'); \
E = parametric_refinement_force_errors( F, \
Cx=$Cx, Cy=$Cy, Cx_p=$Cx_p, Cy_p=$Cy_p, \
Axial_imin=$Axial_imin, Axial_imax=$Axial_imax, Axial_jmin=$Axial_jmin, Axial_jmax=$Axial_jmax, \
Normal_imin=$Normal_imin, Normal_imax=$Normal_imax, Normal_jmin=$Normal_jmin, Normal_jmax=$Normal_jmax ); \
output_hdf5_dict(E,'$run_dir$force_error_file')"

echo "$my_python_command1"
python3 -c "$my_python_command1"

echo "$my_python_command2"
python3 -c "$my_python_command2"

echo "$my_python_command3"
python3 -c "$my_python_command3"