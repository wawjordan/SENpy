#!/bin/sh
# module load Miniconda3/24.7.1-0
# module load tecplot360/2023r2
# conda deactivate
# source activate pydafiV3
SENpy_location='/home/wajordan/SENpy'
run_dir='/home/wajordan/SENSEI_CASES/parabola_test_param_sweep'
subdir_pattern1='*case_*'
subdir_pattern2='*'
save_backups='False'
verbose='True'
# output_hdf5_dict(dic,filename):
# parametric_refinement_forces(run_dir,subdir_pattern1="*",subdir_pattern2="*",verbose=True):
# parametric_refinement_error_norms(run_dir,subdir_pattern1="*",subdir_pattern2="*",verbose=True):
my_python_command1=\
"import sys; sys.path.append('$SENpy_location'); \
from sensei_batch_run_helper import convert_dat_files_parametric_study; \
convert_dat_files_parametric_study( '$run_dir', \
subdir_pattern1='$subdir_pattern1', \
subdir_pattern2='$subdir_pattern2', \
save_backups=$save_backups, \
verbose=$verbose)"

echo "$my_python_command1"
python3 -c "$my_python_command1"