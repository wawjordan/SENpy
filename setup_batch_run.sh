#!/bin/sh
# module load Miniconda3/24.7.1-0
# source activate pydafiV3
SENpy_location='/home/wajordan/SENpy'
executable_name='/home/wajordan/SENSEI_DEV/HO-dev/build-release/bin/SENSEI'
src_nml='/home/wajordan/tmp/sensei.nml'
src_grids='/home/wajordan/SENSEI_PRACTICE/SVF_11032026/grids/'
job_name='svf-param-sweep'
output_dir='/home/wajordan/tmp/tmp/'
task_list_out=$output_dir'task_list.txt'
key_file=$output_dir'key_list.txt'
username='wajordan@vt.edu'
base_mem_size=1
array_throttle=0.5
base_name='case'
folder_pattern='*'

namelist_args=\
"CONSTANT_DENSITY_MS=('EXACT',[False,True]), \
ENFORCE_MMS_CONSTRAINT=('RECONSTRUCTION',[False,True]), \
ENFORCE_SLIP_WALL_CONSTRAINT=('RECONSTRUCTION',[False,True]), \
CHECK_BOUNDARY_CHARACTERISTICS=('DE_ESTIMATION',[False,True])"

my_python_command=\
"import sys; sys.path.append('$SENpy_location'); \
from sensei_batch_run_helper \
import generate_namelist_vals, create_case_dirs; \
namelist_vals = generate_namelist_vals( '$src_nml', \
key_file='$key_file', \
base_name='$base_name', \
$namelist_args); \
create_case_dirs( '$src_grids', namelist_vals, \
executable_name='$executable_name', \
job_name='$job_name', \
base_mem_size=$base_mem_size, \
array_throttle=$array_throttle, \
username='$username', \
dir='$output_dir', \
base_name='$base_name', \
folder_pattern='$folder_pattern', \
out_file_name='$task_list_out' \
)"

echo "$my_python_command"

# python -c "$my_python_command"