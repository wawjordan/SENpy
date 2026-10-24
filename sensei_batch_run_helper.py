# standard library imports
import os
import shutil
import subprocess
import multiprocessing
from multiprocessing import Pool, cpu_count
from pathlib import Path
from datetime import datetime
from sys import stderr, stdout
import zipfile
import psutil
import time
import argparse

# third party imports
import pdb
import numpy as np
# import yaml
# import h5py
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
# TENSORDIM = 9
# find . -maxdepth 1 -mindepth 1 -type d -exec touch {}/filename.rst \;
# find ./*/ -type f -name "*.nml" -delete


def convert_dat_files(run_dir,save_backups=True,verbose=False):
  run_dir = Path(run_dir)
  # ".dat" files that aren't formatted for tecplot
  file_patterns = ( "-rec-err_primitive.dat",
                    "-te_ext.dat",
                    "-te_est.dat",
                    "-te_err.dat",
                    "-de_primal_primitive_1.dat",
                    "-de_primal_primitive.dat",
                    "-de_primal_primitive_offset.dat",
                    "-de_primal_conserved_1.dat",
                    "-de_primal_conserved.dat",
                    "-de_primal_conserved_offset.dat",
                    "-de_corrected_primitive_1.dat",
                    "-de_corrected_primitive.dat",
                    "-de_corrected_primitive_offset.dat",
                    "-de_corrected_primitive_ic.dat",
                    "-de_corrected_primitive_ic_offset.dat",
                    "-de_corrected_conserved_1.dat",
                    "-de_corrected_conserved.dat",
                    "-de_primal_conserved_offset.dat",
                    "-de_corrected_conserved_ic.dat",
                    "-de_corrected_conserved_ic_offset.dat",
                    "-press_force.dat",
                    "-press_force_curv.dat",
                    "-press_force_rec.dat",
                    "-press_force_curv_rec.dat",
                    "-visc_force.dat",
                    "-visc_force_curv.dat",
                    "-visc_force_rec.dat",
                    "-visc_force_curv_rec.dat",
                    "-de_primal_primitive_flagged",
                    "-de_primal_conserved_flagged",
                    "-de_corrected_primitive_flagged",
                    "-de_corrected_conserved_flagged",
                  )
  dat_file_list = [file_path for file_path in run_dir.glob("*.dat")
                   if file_path.name.endswith(file_patterns)]

  # create log file to hold output from tec360 batch.log output files
  log_file = run_dir / "tec360_batch.log"
  tmp_log_file = run_dir / "batch.log"
  with open(log_file, "w", encoding="utf-8") as file:
    file.close()

  # loop over the dat files
  for file_path in dat_file_list:
    try:
      # preprocess (replace NaN's with 1e69)
      bash_command = ["sed","-i.bak","s/\\bNaN\\b/1e69/gI", str(file_path.resolve())]
      subprocess.run(bash_command,cwd=run_dir.resolve(),check=True)
    except subprocess.CalledProcessError as e:
      if verbose:
        print(f"sed failed with exit code {e.returncode}")
      print(f"Error output: {e.stderr}")
    #  tec360 conversion to szplt
    try:
      old_file = file_path.resolve()
      new_file = (file_path.with_suffix(".szplt")).resolve()
      # strip out unnecessary addons
      bash_command = ["tec360",
                      "-nostdaddons",
                      "-loadaddon","tecutildataio_szlfile",
                      "-convert",f"{old_file}",
                      "-o", f"{new_file}"]
      if verbose:
        print(f"Writing: {new_file}")
      subprocess.run(bash_command,cwd=run_dir.resolve(),capture_output=True,text=True,check=True)
      
      # copy contents of single batch.log file to the cumulative tec360_batch.log file
      with open(log_file, 'a') as f_target, open(tmp_log_file, 'r') as f_source:
        shutil.copyfileobj(f_source, f_target)
    except subprocess.CalledProcessError as e:
      if verbose:
        print(f"tec360 -convert failed with exit code {e.returncode}")
      print(f"Error output: {e.stderr}")

  # remove all (non-excluded) .dat files
  for file_path in dat_file_list:
    file_path.unlink()   # delete the file
  # change .dat.bak back to .dat, add to archive, and delete
  for file_path in run_dir.glob("*.dat.bak"):
    file_path.rename(file_path.with_suffix(""))
  bkp_file = run_dir / "dat_backup.zip"
  if save_backups:
    with zipfile.ZipFile(bkp_file, "w", compression=zipfile.ZIP_DEFLATED) as zipf:
      for file_path in dat_file_list:
        if verbose:
          print(f"Zipping: {file_path}")
        zipf.write(file_path)
  # delete the dat files
  for file_path in dat_file_list:
    if verbose:
      print(f"Deleting: {file_path}")
    file_path.unlink() # delete the file
      


def get_meta_data_path(executable_path):
  # assumes that executable is inside of build_dir/bin/
  build_dir = Path(executable_path).resolve().parent.parent
  metadata = (build_dir / "metadata.log").resolve()
  if metadata.is_file():
    return metadata
  raise FileNotFoundError(f"metadata.log not found in {build_dir}")

def has_sensei_nml(run_dir):
  namelist = (run_dir / "sensei.nml" ).resolve()
  if namelist.is_file():
    return True
  raise FileNotFoundError(f"sensei.nml not found in {run_dir}")

def pre_sensei(executable_path,run_dir,clean=False,restart=False):
  err_string = ""
  if has_sensei_nml(run_dir):
    try:
      if clean:
        clean_sensei(run_dir=run_dir,verbose=False,clean_restarts=restart,clean_metadata=True)
      metadata = get_meta_data_path(executable_path)
      try:
        shutil.copy2(metadata,run_dir)
      except Exception as e:
        print(e, file=stderr)
        print(f"An error occured when copying metadata.log",file=stdout)
    except FileNotFoundError as e:
      print(e, file=stderr)

def create_case_dirs_4_param(src_dir,src_file,dir=Path.cwd(),base_name="case",folder_pattern="*",out_file_name="task_list.txt"):
  width = 4
  out_file = Path(out_file_name)
  with open(out_file, "w", encoding="utf-8") as file:
    file.close()
  directory_list = [] # all directory strings
  str_list = []       # unique substrings
  for i in range(2**width):
    binary_str = f"{i:0{width}b}"
    # print(f"{binary_str}")
    vals = [char == '1' for char in binary_str]
    name = base_name + "_" + binary_str
    dir_path = ( Path(dir) / name ).resolve()
    dir_path.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src_dir, dir_path, dirs_exist_ok=True)
    change_namelist_4_param(src_file,dir_path,CD=vals[0],MB=vals[1],SB=vals[2],CC=vals[3])
    copy_nml_to_subfolders(run_dir=dir_path,file_name=Path(src_file).name)
    target_directories = [x for x in dir_path.glob(folder_pattern) if x.is_dir()]
    for directory in sorted(target_directories):
      directory_list.append(directory)
      if directory.name not in str_list:
        str_list.append(directory.name)
      with open(out_file, "a", encoding="utf-8") as file:
        file.write(f"{directory}\n")
        
  for i, str_ in enumerate(str_list):
    out_file1 = out_file.with_stem(out_file.stem + f"{i:02d}")
    with open(out_file1, "w", encoding="utf-8") as file:
      for item in directory_list:
        if str_ in str(item):
          file.write(f"{item}\n")


def change_namelist_4_param(src_file,dst,CD=False,MB=False,SB=False,CC=False):
  patterns = [
    "CONSTANT_DENSITY_MS",
    "ENFORCE_MMS_CONSTRAINT",
    "ENFORCE_SLIP_WALL_CONSTRAINT",
    "CHECK_BOUNDARY_CHARACTERISTICS",
  ]
  # convert to T/F
  values = [CD,MB,SB,CC]
  vals = []
  for val in values:
    vals.append("T" if val else "F")
  filename = Path(src_file)
  new_file = ( Path(dst) / filename.name ).resolve()
  if filename.is_file():
    try:
      shutil.copy2(src_file,dst)
    except Exception as e:
      print(e, file=stderr)
      print(f"An error occured when copying {src_file}",file=stdout)
    for i, pattern in enumerate(patterns):
      val = "T" if val else "F"
      bash_command = ["sed","-i",f's/\({pattern}[[:space:]]*=[[:space:]]*\)[TF]/\\1{vals[i]}/',f"{new_file}"]
      subprocess.run(bash_command)
  else:
    raise FileNotFoundError(f"{src_file} not found")


  
def run_sensei_serial(run_dir,executable_path):
    output = subprocess.run(executable_path, cwd=run_dir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,text=True)
    log_file = "log_" + datetime.now().strftime("%Y_%m_%d__%H_%M_%S") + ".txt"
    with open(log_file, "w", encoding="utf-8") as file:
      file.write(output.stdout)
    print(f"Finished!")

def clean_sensei(run_dir=Path.cwd(),verbose=True,clean_restarts=True,clean_metadata=True):
  patterns = [
    "nohup.out",
    "*.dat",
    "*.txt",
    "*.tar",
    "sensei.nml.*",
    "sensei_turb.nml.*",
    "*.deexact*",
    "fort.0",
    "*-coarse.bc",
    "*-agglom.bc",
    "*-coarse.grd",
    "*-fine.grd",
    "*.hdf5.out",
  ]
  if clean_restarts:
    patterns.append("*.rst")
  if clean_metadata:
    patterns.append("metadata.log")
  for pattern in patterns:
    for file_path in run_dir.glob(pattern):
        if file_path.is_file():  # make sure it's not a folder
            if verbose:
              print(f"Deleting: {file_path}")
            try:  
              file_path.unlink()   # delete the file
            except Exception as e:
              print(f"Error deleting {file_path}: {e}",file=stderr)
            
def copy_nml_to_subfolders(run_dir=Path.cwd(),folder_pattern="*",file_name="sensei.nml"):
  namelist = (run_dir / file_name ).resolve()
  target_directories = [x for x in Path(run_dir).glob(folder_pattern) if x.is_dir()]
  for directory in target_directories:
    try:
      shutil.copy2(namelist,directory)
    except Exception as e:
      print(e, file=stderr)

# Run SENSEI Cases
    
# parallel = multiprocessing.Pool(self.ncpu) 

# _ = parallel.starmap(_run_sensei, inputs_run)
# parallel.close()
# print(f'    Finished SENSEI Cases ... {time.time()-ts:.2f}s')

# parallel = multiprocessing.Pool(self.ncpu)  
# sample_dirs=[]

# for ii in range(self.nsamples):
#     if broken_sims[ii]==0:
#         sample_dirs.append(self.results_dir+"/sample_"+str(ii)+"/iteration_"+str(int(self.da_iteration))+'_'+self.sensei_casename[case])
# if case ==0:
#     inputs_run = [(sample_dirs[i],n_decomps) for i in range(len(sample_dirs))]
# else:
#     inputs_run = inputs_run+[(sample_dirs[i],n_decomps) for i in range(len(sample_dirs))]

# # Run SENSEI Cases

# _ = parallel.starmap(_run_sensei, inputs_run)
# parallel.close()


# # Changes directory and runs sensei from that directory
# def _run_sensei(run_dir,ncpu):
#     my_dir=os.getcwd()
#     os.chdir(run_dir)
#     bash_command= 'rm sensei_turb.nml.* sensei.nml.*'
#     subprocess.call(bash_command,shell=True)
#     bash_command=f'mpirun -np {ncpu} --bind-to none /projects/mlsttrphase2/SENSEI_ML_Training_Bradshaw/BC-dev-dev/build-mpi-release/bin/SENSEI > test.out'
#     subprocess.call(bash_command,shell=True)
#     os.chdir(my_dir)