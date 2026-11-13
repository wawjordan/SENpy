#!/bin/sh
# module load Miniconda3/24.7.1-0
# module load tecplot360/2023r2
# source activate pydafiV3
SENpy_location='/home/wajordan/SENpy'
run_dir='/home/wajordan/SENSEI_CASES/parabola_test_param_sweep/'
out_dir='/home/wajordan/SENSEI_CASES/parabola_test_param_sweep/'
rel_dir='/home/wajordan/SENSEI_CASES/parabola_test_param_sweep/'
out_file='compressed_data.zip'
subdir_pattern1='*case_*'
subdir_pattern2='*'
subdir_exclude1='[]'
subdir_exclude2='[]'
verbose='True'
kwargs=\
""


my_python_command=\
"import sys; sys.path.append('$SENpy_location'); \
from pathlib import Path; \
from error_norm_parsing import parametric_refinement_compress_data; \
parametric_refinement_compress_data( '$run_dir', \
(Path('$out_dir') / '$out_file').resolve(), \
rel_dir='$rel_dir'
subdir_pattern1='$subdir_pattern1', \
subdir_pattern2='$subdir_pattern2', \
subdir_exclude1=$subdir_exclude1, \
subdir_exclude2=$subdir_exclude2, \
verbose=$verbose, \
$kwargs )"

echo "$my_python_command"
python3 -c "$my_python_command"