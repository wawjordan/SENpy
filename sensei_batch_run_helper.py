from pathlib import Path
from sys import stderr, stdout
import shutil
import subprocess
import zipfile
import json
import re
import numpy as np

# import psutil
# import time
# import argparse
# import os
# import pdb
# from multiprocessing import Pool, cpu_count
# import yaml
# import h5py
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
# TENSORDIM = 9
# find . -maxdepth 1 -mindepth 1 -type d -exec touch {}/filename.rst \;
# find ./*/ -type f -name "*.nml" -delete

def write_batch_script( filename,
                        task_list,
                        executable,
                        logname='log',
                        job_name='test',
                        account='ete_cfd',
                        partition='normal_q',
                        nodes=1,
                        ntasks_per_node=1,
                        cpus_per_task=1,
                        mem='1G',
                        time=144.0,
                        array_size=1,
                        array_throttle=None,
                        user=None,
                        mail_type='ARRAY_TASKS'):
  from datetime import timedelta
  with open(filename,"w",encoding="utf-8") as f:
    f.write(r'#!/bin/sh'+'\n')
    f.write(rf'#SBATCH --job-name={job_name}'+'\n')
    f.write(rf'#SBATCH --account={account}'+'\n')
    f.write(rf'#SBATCH --partition={partition}'+'\n')
    f.write(rf'#SBATCH --nodes={nodes:d} --ntasks-per-node={ntasks_per_node:d} --cpus-per-task={cpus_per_task:d}'+'\n')
    f.write(rf'#SBATCH --mem={mem}'+'\n')
    hours, remainder = divmod(timedelta(hours=time).total_seconds(), 3600)
    minutes, seconds = divmod(remainder, 60)
    f.write(rf'#SBATCH --time=0-{int(hours)}:{int(minutes):02d}:{int(seconds):02d}'+'\n')
    if array_throttle is None:
      array_throttle = array_size
    f.write(rf'#SBATCH --array=1-{array_size:d}%{array_throttle:d}'+'\n')
    if user is not None:
      f.write(rf'#SBATCH --mail-user={user}'+'\n')
      f.write(rf'#SBATCH --mail-type={mail_type}'+'\n')
    f.write(rf'#SBATCH --output={logname}/%x_%A_%a.out')
    f.write('\n')
    f.write(r'cd "$(sed -n "${SLURM_ARRAY_TASK_ID}p"' + rf' {task_list})" || exit 1'+'\n')
    f.write(rf'{executable} 2>&1 | tee log_$(date +"%Y-%m-%d_%H:%M%S").txt'+'\n')


def read_namelist(text):
  """Parse a Fortran namelist string into a list of (name, dict) groups."""
  def convert_scalar(token):
    # Try int, then float, otherwise fall back to a string
    try:
      return int(token)
    except ValueError:
      pass
    try:
      return float(token)
    except ValueError:
      pass
    return token
  
  def read_value(value_str):
    # Convert a namelist value string into a Python list of dicts
    # Remove a trailing comma, e.g. "100000.0," -> "100000.0"
    value_str = value_str.rstrip(',').strip()
    if value_str.startswith('"') or value_str.startswith("'"):
      return value_str.strip('"\'').strip()   # quoted string
    if value_str.upper() in ('T', 'F'):       #  boolean
      return value_str.upper() == 'T'
    parts = [p.strip() for p in value_str.split(',') if p.strip()]
    values = [convert_scalar(p) for p in parts]
    # return a scalar if appropriate
    return values[0] if len(values) == 1 else values
  
  groups = []     # the result: one entry per &GROUP
  current = None  # the current group
  current_key = None # the current key
  for raw_line in text.splitlines():
    line = raw_line.split('!')[0].strip()   # trim comments and whitespace
    if not line:
      continue                              # skip blank lines
    if line.startswith('&'):                # start of a new group
      current = (line[1:].strip(), {})      # initialize with name and empty dict
      current_key = None                    # initialize to None
      groups.append(current)
    elif line.startswith('/'):              # end of the group
      current = None
      current_key = None
    elif current is not None:
      if '=' in line: # new assignment
        key, _, value_str = line.partition('=')
        current[1][key.strip()] = read_value(value_str.strip())
        current_key = key
      elif current_key is not None:
        # assume we are continuing assignment from previous line
        # append the new entries to the list
        current[1][current_key.strip()].extend( read_value(line.strip()) )
    else: # assume that we are done and continue
      current_key = None # set key to none
  return groups


def namelist_fmt(v):
    if v is True:  return 'T'
    if v is False: return 'F'
    if isinstance(v, (list, tuple)):                 # comma-separated list
        return ','.join(namelist_fmt(x) for x in v)
    if isinstance(v, str) and not (v and v[0] in "'\""):
        return f'"{v}"'                              # auto-quote plain strings
    return str(v)


def write_namelist(groups,
                   filename):
  def namelist_fmt(v): # formats values for printing to namelist
    if v is True:  return 'T' # boolean .true.
    if v is False: return 'F' # boolean .false.
    if isinstance(v, (list, tuple)): # comma-separated list
        return ','.join(namelist_fmt(x) for x in v)
    if isinstance(v, str) and not (v and v[0] in "'\""):
        return f'"{v}"' # add appropriate quotes to strings
    return str(v)

  with open(filename, 'w') as f:
    # name is the name of the respective group
    for name, entry in groups:
      # write the group header
      f.write(f"&{name.upper()}\n")
      # write each entry of this group
      for key, value in entry.items():
        f.write(f" {key.upper()}={namelist_fmt(value)},\n")
      # close the group
      f.write('/\n')


def change_namelist_params(src_file,
                           new_file,
                           **kwargs):
  KEY  = r'[A-Za-z][A-Za-z0-9_]*'

  # One "atom" of a value: either a quoted string (single or double, which may
  # contain commas, newlines, '=', '/'), or any single non-quote character.
  ATOM = r"(?:'[^']*'|\"[^\"]*\"|[^'\"])"

  PATTERN = re.compile(
    rf'(?P<name>{KEY})\s*=\s*'
    rf'(?P<value>{ATOM}*?)'              # lazy: stop at first valid terminator
    rf'(?=\s*/|\s*{KEY}\s*=|$)'          # next key, '/', or end of file
  )

  def substitute(m):
    key = m.group('name').upper()
    if key in replacement_dict:
      # preserve the trailing separator (comma / whitespace) the
      # original value ended with
      trailing = re.search(r'[\s,]*$', m.group('value')).group(0)
      return f"{m.group('name')}={replacement_dict[key]}{trailing}"
    return m.group(0) # leave unmatched entries alone
  
  replacement_dict = {k.upper(): namelist_fmt(v) for k, v in kwargs.items()}
  filename = Path(src_file)
  newfilename = Path(new_file)
  if filename.is_file():
    try:
      with open(filename, 'r') as f:
        content = f.read()
      new_content = PATTERN.sub(substitute, content)
      # Warn about keys that were never found in the file
      found = {m.group('name').upper() for m in PATTERN.finditer(content)}
      missing = set(replacement_dict) - found
      if missing:
        print(f"Warning: keys not found in file: {sorted(missing)}")
      with open(newfilename, 'w') as f:
        f.write(new_content)
    except Exception as e:
      print(e, file=stderr)
      print(f"An error occured when reading {src_file}",file=stdout)
  else:
    raise FileNotFoundError(f"{src_file} not found")


def hash_namelist(groups,
                  out_file_loc=None):
  # generates a hash for the unique .json of the namelist content
  import hashlib
  def json_dumps(var): # specify defaults for repeatability
    return json.dumps(
      var,
      skipkeys=False,
      check_circular=True,
      allow_nan=True,
      cls=None,
      default=None,
      ensure_ascii=True,
      sort_keys=True,
      indent=None,
      separators=(',', ':'),
    )
  def json_dump_file(var,file): # writes the json to file with pretty printing
    json.dump(
      var,
      file,
      skipkeys=False,
      check_circular=True,
      allow_nan=True,
      cls=None,
      default=None,
      ensure_ascii=True,
      sort_keys=True,
      indent=4,
      separators=(',', ':'),
    )
  # generate the string from hash in hexidecimal form
  hash_string = hashlib.md5(json_dumps(groups).encode('utf-8')).digest().hex()
  # if out_file_loc is specified, write the .json file there
  # using the hash string in the name
  if out_file_loc is not None:
    dir_path = Path(out_file_loc)
    if dir_path.is_dir():
      file_str = hash_string + ".json"
      out_file = dir_path / file_str
      with open(out_file.resolve(),'w') as f:
        json_dump_file(groups,f)
    else:
      print(f"Warning: Specified location not found: {out_file_loc}")
  return hash_string


def hash_namelist_file(namelist_file,
                       out_file_loc=None):
  # generates a hash for the unique .json of the namelist content
  with open(namelist_file,'r') as f:
    groups = read_namelist(f.read())
  return hash_namelist(groups,out_file_loc=out_file_loc)


def copy_nml_to_subfolders(run_dir=Path.cwd(),
                           folder_pattern="*",
                           file_name="sensei.nml"):
  namelist = (run_dir / file_name ).resolve()
  target_directories = [x for x in Path(run_dir).glob(folder_pattern) if x.is_dir()]
  for directory in target_directories:
    try:
      shutil.copy2(namelist,directory)
    except Exception as e:
      print(e, file=stderr)


def generate_namelist_vals(src_namelist,
                           key_file=None,
                           base_name="case",
                           **kwargs):
  import itertools
  from copy import deepcopy
  # list of lists:
  # replacement_dict = {k.upper(): v for k, v in kwargs.items()}
  groups = []
  replacement_dict = {}
  for k, v in kwargs.items():
    groups.append(v[0])
    replacement_dict[k] = v[1]
  arg_list = list(replacement_dict.values())
  keys     = list(replacement_dict.keys())
  combinations = list(itertools.product(*arg_list))
  if key_file is not None:
    # write out a key for what parameters correspond to the case number
    with open(Path(key_file).resolve(),'w',encoding="utf-8") as file:
      for j, item in enumerate(combinations):
        name = base_name + "_" + f"{j:03d}"
        file.write(f"{name}: ")
        for i, k in enumerate(keys):
          file.write(f" {k}={namelist_fmt(item[i])},")
        file.write('\n')
  with open(src_namelist,'r') as f:
    base = read_namelist(f.read())
  group_idx = []
  for group in groups:
    tmp_list = [j for j,val in enumerate(base) if group == val[0]]
    if len(tmp_list) > 0:
      group_idx.append(tmp_list[0])
      if len(tmp_list) > 1:
        print(f"Warning: multiple matching groups in namelist; using first match")
    else:
      raise ValueError("Error: group not found in namelist")
  # list of output lists
  # Note the need for deepcopy here; otherwise the list just points to the same item
  namelist_vals = [deepcopy(base) for _ in range(len(combinations))]
  for i, vals in enumerate(combinations):
    for j, key in enumerate(keys):
      namelist_vals[i][group_idx[j]][1][key] = vals[j]

  return namelist_vals


def create_case_dirs( src_dir, 
                      namelist_vals,
                      executable_name='echo',
                      job_name='test',
                      array_throttle=0.5,
                      base_mem_size=1,
                      username=None,
                      dir=Path.cwd(),
                      base_name="case",
                      folder_pattern="*",
                      out_file_name="task_list.txt"):
  # namelist_vals is a list of lists of dictionaries       |
  #                           |--> these contain each case |
  out_file = Path(out_file_name)

  if array_throttle > 1.0: array_throttle = 1.0

  # create the file that will accumulate the case directories
  with open(out_file, "w", encoding="utf-8") as file:
    file.close()
    
  directory_list = [] # all directory strings
  str_list = []       # unique substrings

  # loop over the different cases
  for i, nml_val in enumerate(namelist_vals):
    # simply generate name from enumeration
    name = base_name + "_" + f"{i:03d}"

    # path to this specific refinement study
    dir_path = ( Path(dir) / name ).resolve()
    dir_path.mkdir(parents=True, exist_ok=True)

    # copy all of the source folders 
    # (the case directories containing .grd and .bc files)
    shutil.copytree(src_dir, dir_path, dirs_exist_ok=True)

    # write the parameters in nml_val to a namelist
    nml_path = (dir_path / "sensei.nml").resolve()
    write_namelist(nml_val,nml_path)

    # output a json file with the namelist information as well
    hash_namelist_file(nml_path,out_file_loc=dir_path)

    # now copy the namelist to each subfolder
    # in the refinement study directory
    copy_nml_to_subfolders(run_dir=dir_path,file_name=nml_path)

    # now, we will generate full task_list
    # based on subfolders in each refinement study that match folder_pattern
    target_directories = [x for x in dir_path.glob(folder_pattern) if x.is_dir()]
    for directory in sorted(target_directories):
      directory_list.append(directory)
      if directory.name not in str_list:
        # unique patterns (should match to the different grid sizes)
        str_list.append(directory.name)
      with open(out_file, "a", encoding="utf-8") as file:
        file.write(f"{directory}\n")

  # create bash script to submit all jobs
  batchfile_all = (Path(dir) / "run_all.sh").resolve()
  with open(batchfile_all, "w", encoding="utf-8") as run_all_file:
    run_all_file.write(r'#!/bin/sh'+'\n')
  # now write out separate task lists for each unique pattern
  # We do this to have a task list for the set of all cases of a given size
  # That way we can specify RAM allocation according to the grid size 
  # while still using job arrays
  for i, str_ in enumerate(str_list):
    out_file1 = out_file.with_stem(out_file.stem + f"{i:02d}")
    cnt = 0
    with open(out_file1, "w", encoding="utf-8") as file:
      for item in directory_list:
        if str_ in str(item):
          cnt += 1
          file.write(f"{item}\n")
    batchfile = (Path(dir) / f"run{i:02d}.sh").resolve()
    # append the new script to the run_all script
    with open(batchfile_all, "a", encoding="utf-8") as run_all_file:
        run_all_file.write(f"echo 'running: {batchfile}'\n")
        run_all_file.write(f"sbatch {batchfile}\n")
    write_batch_script( batchfile,
                        out_file1,
                        executable_name,
                        logname=f"log{i:02d}",
                        job_name=job_name,
                        mem=f"{base_mem_size*2**i:d}G",
                        array_size=cnt,
                        array_throttle=int(np.ceil( array_throttle * cnt)),#-(cnt//-2),
                        user=username,
                        )


def convert_dat_files(run_dir,
                      save_backups=True,
                      verbose=False):
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
                   if not file_path.name.endswith(file_patterns)]

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


def convert_dat_files_parametric_study(run_dir,
                                       subdir_pattern1="*",
                                       subdir_pattern2='*',
                                       save_backups=True,
                                       verbose=False):
  for i, subdir in enumerate(sorted(Path(run_dir).glob(f"{subdir_pattern1}"))):
    if verbose:
      print(f"{subdir}")
    for subsubdir in sorted(Path(subdir).glob(f"{subdir_pattern2}")):
      if subsubdir.is_dir():
        if verbose:
          print(f"{subsubdir}")
        convert_dat_files(subsubdir,save_backups=save_backups,verbose=verbose)


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


def pre_sensei(executable_path,
               run_dir,
               clean=False,
               restart=False):
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


def run_sensei_serial(run_dir,
                      executable_path):
  from datetime import datetime
  output = subprocess.run(executable_path, cwd=run_dir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,text=True)
  log_file = "log_" + datetime.now().strftime("%Y_%m_%d__%H_%M_%S") + ".txt"
  with open(log_file, "w", encoding="utf-8") as file:
    file.write(output.stdout)
  print(f"Finished!")


def clean_sensei(run_dir=Path.cwd(),
                 verbose=True,
                 clean_restarts=True,
                 clean_metadata=True):
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



if __name__ == "__main__":
  src_file ='/home/wajordan/tmp/sensei.nml'
  dst      ='/home/wajordan/tmp/sensei_updated.nml'
  dst2      ='/home/wajordan/tmp/sensei_updated2.nml'
  dst3      ='/home/wajordan/tmp/sensei.nml.json'
  out_loc   ='/home/wajordan/tmp/tmp/'
  # change_namelist_params( src_file, dst,
  #                         check_boundary_characteristics=False,
  #                         N_LAYERS =(-1,-1, 0,-1,-1, 0,),
  #                         )
  # with open(src_file,'r') as f:
  #   groups = read_namelist(f.read())
  # write_namelist(groups,dst2)
  # print(hash_namelist_file(src_file,out_file_loc=out_loc))
  # print(hash_namelist_file(dst2,out_file_loc=out_loc))

  # write_batch_script( Path(out_loc) / 'run.sh',
  #                     'task_list.txt',
  #                     'executable',
  #                       logname='log',
  #                       job_name='test',
  #                       account='ete_cfd',
  #                       partition='normal_q',
  #                       nodes=1,
  #                       ntasks_per_node=1,
  #                       cpus_per_task=1,
  #                       mem='1G',
  #                       time=144.0,
  #                       array_size=1)

  namelist_vals = generate_namelist_vals( src_file, key_file='/home/wajordan/pySENSEI/key.txt',
                                          CONSTANT_DENSITY_MS=('EXACT',[False,True]),
                                          ENFORCE_MMS_CONSTRAINT=('RECONSTRUCTION',[False,True]),
                                          ENFORCE_SLIP_WALL_CONSTRAINT=('RECONSTRUCTION',[False,True]),
                                          CHECK_BOUNDARY_CHARACTERISTICS=('DE_ESTIMATION',[False,True]) )
  # print(len(namelist_vals))

  create_case_dirs( '/home/wajordan/SENSEI_PRACTICE/SVF_11032026/grids/',
                    namelist_vals,
                    executable_name='/home/wajordan/SENSEI_DEV/HO-dev/build-release/bin/SENSEI',
                    job_name='svf-param-sweep',
                    base_mem_size=1,
                    username='wajordan@vt.edu',
                    dir=out_loc,
                    base_name="case",
                    folder_pattern="*",
                    out_file_name=Path(out_loc) / "task_list.txt")
