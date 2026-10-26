# standard library imports
import shutil
import subprocess
from pathlib import Path
from sys import stderr, stdout
import zipfile
import warnings
import pdb
import re
from math import prod

# third party imports

import numpy as np

# import yaml
# import h5py
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
# TENSORDIM = 9
# find . -maxdepth 1 -mindepth 1 -type d -exec touch {}/filename.rst \;
# find ./*/ -type f -name "*.nml" -delete


def output_hdf5_dict(dic,filename):
  import h5py
  with h5py.File(Path(filename).resolve(), "w") as f:
    save_dict_to_hdf5(dic, f)

def save_dict_to_hdf5(dic, h5file_or_group):
  """
  Recursively saves a dictionary to an open h5py File or Group object.
  """
  for key, item in dic.items():
    if isinstance(item, dict):
      # Create a group for a nested dictionary
      subgroup = h5file_or_group.create_group(key)
      save_dict_to_hdf5(item, subgroup)
    else:
      # Ensure data is in a format h5py accepts (like a numpy array)
      h5file_or_group.create_dataset(key, data=np.asarray(item))

def read_hdf5_dict(filename):
  import h5py
  with h5py.File(Path(filename).resolve(), 'r') as f:
    dic = h5_to_dict(f)
  return dic

def h5_to_dict(h5_object):
  import h5py
  """
  Recursively converts an h5py File or Group into a Python dictionary.
  """
  dic = {}
  for key, item in reversed(h5_object.items()):
    if isinstance(item, h5py.Dataset):
      # Load dataset as a NumPy array
      dic[key] = item[:]  # or item[:]
    elif isinstance(item, h5py.Group):
      # Recursively read the subgroup
      dic[key] = h5_to_dict(item)
  return dic

def parametric_refinement_error_norms(run_dir,subdir_pattern1="*",subdir_pattern2="*",verbose=True):
  error_norms = [] # initialize temporary list
  foldernames = []
  for i, subdir in enumerate(sorted(Path(run_dir).glob(f"{subdir_pattern1}"))):
    print(f"{subdir}")
    foldernames.append( f"{subdir.parent.name}:{subdir.name}")
    error_norms.append(refinement_error_norms(subdir,subdir_pattern=subdir_pattern2,verbose=verbose))
  # make a new dictionary, stacking all of the error norms for each error type
  DE = dict.fromkeys([key for key in error_norms[0]])
  DE["N1"] = error_norms[0]["N1"]
  DE["N_layers"] = {key : None for key in foldernames}
  DE["N_iterations"] = {key : None for key in foldernames}
  DE["t"] = {key : None for key in foldernames}
  for i in range(len(error_norms)):
    DE["t"][foldernames[i]] = error_norms[i]["t"]
    DE["N_layers"][foldernames[i]] = error_norms[i]["N_layers"]
    DE["N_iterations"][foldernames[i]] = error_norms[i]["N_iterations"]
  for key in error_norms[0]: # note: this won't work if they are different sizes
    if key=="t": continue
    if key=="N1": continue
    if key=="N_layers": continue
    if key=="N_iterations": continue
    arrays = [d[key] for d in error_norms]
    max_shape = np.max([arr.shape for arr in arrays], axis=0)
    padded_arrays = []
    for arr in arrays:
      pad_width = [(0, max_shape[i] - arr.shape[i]) for i in range(len(max_shape))]
      padded_arr = np.pad(arr, pad_width, mode='constant', constant_values=np.NaN)
      padded_arrays.append(padded_arr)
    DE[key] = np.stack(padded_arrays,axis=0)
  return DE
  


def refinement_error_norms(run_dir,subdir_pattern="*",verbose=False):
  error_norms = [] # initialize temporary list
  # append error norm dicts from each subdirectory
  for subdir in sorted(Path(run_dir).glob(f"{subdir_pattern}")):
    if subdir.is_dir():
      error_norms.append(single_case_error_norms(subdir,verbose=verbose))
  is_sorted = False
  keys = [key for key in error_norms[0]] # they should all have the same keys
  special_key = keys[0] # initialize
  for key in keys: # (hopefully) find a key that is fully populated across all grid levels
    tmp = [item[key]["t"] for item in error_norms if (np.size(item.get(key,{}).get("t",{})) != 0)]
    if len(tmp) == len(error_norms):
      special_key = key
      break

  for key in keys: # (hopefully) find a key that is fully populated across all grid levels
    tmp = [item[key]["N"] for item in error_norms if (np.size(item.get(key,{}).get("N",{})) != 0)]
    if len(tmp) == len(error_norms):
      error_norms.sort(key = lambda d: d[key]["N"])
      is_sorted = True
      break
  if not is_sorted:
    raise ValueError("Could not find data set that matched across all subdirectories.")

  # make a new dictionary, stacking all of the error norms for each error type
  DE = dict.fromkeys([key for key in error_norms[0]])
  for key in error_norms[0]:
    DE[key] = np.stack([d[key]["E"] for d in error_norms])

  DE["N1"] = np.array([d[special_key]["N"] for d in error_norms])
  DE["N_iterations"] = np.array([d[special_key]["N_iterations"] for d in error_norms])
  DE["N_layers"] = np.array([d[special_key]["N_layers"] for d in error_norms])
  DE["t"] = [d[special_key]["t"] for d in error_norms]

  return DE


def single_case_error_norms(run_dir,verbose=False):
  run_dir = Path(run_dir)
  keys = ["Reconstruction_error",
          "TE_exact",
          "TE_estimated",
          "TE_error",
          "DE_primal_primitive",
          "DE_primal_primitive_flagged",
          "DE_primal_conserved",
          "DE_primal_conserved_flagged",
          "DE_corrected_primitive",
          "DE_corrected_primitive_flagged",
          "DE_corrected_conserved",
          "DE_corrected_conserved_flagged",
        ]
  filepatterns = { "-rec-err_primitive.dat",
                    "-te_ext.dat",
                    "-te_est.dat",
                    "-te_err.dat",
                    "-de_primal_primitive_offset.dat",
                    "-de_primal_primitive.dat",
                    "-de_primal_primitive_1.dat",
                    "-de_primal_primitive_flagged.dat",
                    "-de_primal_conserved_offset.dat",
                    "-de_primal_conserved.dat",
                    "-de_primal_conserved_1.dat",
                    "-de_primal_conserved_flagged.dat",
                    "-de_corrected_primitive_ic_offset.dat",
                    "-de_corrected_primitive_ic.dat",
                    "-de_corrected_primitive_offset.dat",
                    "-de_corrected_primitive.dat",
                    "-de_corrected_primitive_1.dat",
                    "-de_corrected_primitive_flagged.dat",
                    "-de_corrected_conserved_ic_offset.dat",
                    "-de_corrected_conserved_ic.dat",
                    "-de_corrected_conserved_offset.dat",
                    "-de_corrected_conserved.dat",
                    "-de_corrected_conserved_1.dat",
                    "-de_corrected_conserved_flagged.dat",
                    }
  filefromkey = { "Reconstruction_error"           : ["-rec-err_primitive.dat"],
                  "TE_exact"                       : ["-te_ext.dat"],
                  "TE_estimated"                   : ["-te_est.dat"],
                  "TE_error"                       : ["-te_err.dat"],
                  "DE_primal_primitive"            : ["-de_primal_primitive_offset.dat",
                                                      "-de_primal_primitive.dat",
                                                      "-de_primal_primitive_1.dat"],
                  "DE_primal_primitive_flagged"    : ["-de_primal_primitive_flagged.dat"],
                  "DE_primal_conserved"            : ["-de_primal_conserved_offset.dat",
                                                      "-de_primal_conserved.dat",
                                                      "-de_primal_conserved_1.dat"],
                  "DE_primal_conserved_flagged"    : ["-de_primal_conserved_flagged.dat"],
                  "DE_corrected_primitive"         : ["-de_corrected_primitive_ic_offset.dat",
                                                      "-de_corrected_primitive_ic.dat",
                                                      "-de_corrected_primitive_offset.dat",
                                                      "-de_corrected_primitive.dat",
                                                      "-de_corrected_primitive_1.dat"],
                  "DE_corrected_primitive_flagged" : ["-de_corrected_primitive_flagged.dat"],
                  "DE_corrected_conserved"         : ["-de_corrected_conserved_ic_offset.dat",
                                                      "-de_corrected_conserved_ic.dat",
                                                      "-de_corrected_conserved_offset.dat",
                                                      "-de_corrected_conserved.dat",
                                                      "-de_corrected_conserved_1.dat"],
                  "DE_corrected_conserved_flagged" : ["-de_corrected_conserved_flagged.dat"],
                }
  empty_error_norm  = {key: [] for key in ["N","t","N_iterations","N_layers","E"]}
  empty_error_norm['E'] = np.zeros((0,0,0,0))
  error_norms = {key: None for key in keys}
  file_list = [file_path for file_path in run_dir.glob("*.dat")
                     if file_path.name.endswith(tuple(filepatterns))]
  for key in keys:
    for pat in filefromkey[key]: # try each option in sequence until one (hopefully) works
      first_match = next((f for f in file_list if f.name.endswith(pat)), None)
      if first_match is not None:
        try:
          error_norms[key] = read_err_norm_file(first_match.resolve(),verbose=verbose)
          break
        except:
          warnings.warn(f"Error reading from {first_match}")
    if error_norms[key] is None:
      error_norms[key] = empty_error_norm

  return error_norms

# def calc_OOA(DE, r_fac):
#   DE[0]["OOA"] = np.full_like(DE[0]["E"], np.nan)   # (analagous to matlab nan*DE(1).E)
#   with np.errstate(invalid='ignore', divide='ignore'): # suppress NaN warnings for this calc
#     for i in range(1, len(DE)):
#         DE[i]["OOA"] = np.log( DE[i-1]["E"] / DE[i]["E"] ) / np.log(r_fac)
#         DE[i]["OOA"] = np.nan_to_num( DE[i]["OOA"], nan=np.nan, posinf=np.nan, neginf=np.nan)
#   return DE

def read_err_norm_file(file_name,verbose=False):
  file_name = Path(file_name).resolve()
  with open(file_name, "r") as file:
    text_lines = file.read().splitlines()
  for i,text_line in enumerate(text_lines):
    text_lines[i] = text_line.split()
  if verbose:
    print(f"Reading {file_name}")
  if len(text_lines[0]) != 1: # No offsets, no iterative correction
    dat, N, t, N_iterations, N_layers = read_err_norms_basic(text_lines)
  elif len(text_lines[1]) != 1: # offsets, no iterative correction
    dat, N, t, N_iterations, N_layers = read_err_norms_offset(text_lines)
  elif len(text_lines[2]) != 1: # iterative correction, no offsets
    dat, N, t, N_iterations, N_layers = read_err_norms_ic(text_lines)
  else:
    dat, N, t, N_iterations, N_layers = read_err_norms_ic_offset(text_lines)
  return {
            "N": N,
            "t": t,
            "N_iterations": N_iterations,
            "N_layers": N_layers,
            "E": dat,
           }

def read_err_norms_basic(text_lines):
  N = int(text_lines[0][0]) # this line should have grid size etc.
  t = []
  N_iterations = []
  N_layers = []
  if len(text_lines[0]) == 3:
    t = float(text_lines[0][2])
  N_eqns = len(text_lines[1]) # should be first line with error norms
  dat = np.zeros((N_eqns,3,1,1))
  cnt = 0
  for i in range(3): # norm loop
    cnt += 1
    dat[:,i,0,0] = np.array([float(item) for item in text_lines[cnt]]) # eqns
  return dat, N, t, N_iterations, N_layers

def read_err_norms_offset(text_lines):
  N_layers = int(text_lines[0][0]) # assume that this integer is the number of excluded layers (+1)
  N        = int(text_lines[1][0]) # this line should have grid size etc.
  t = []
  N_iterations = []
  if len(text_lines[1]) == 3:
    t = float(text_lines[1][2])
  N_eqns = len(text_lines[2]) # should be first line with error norms
  dat = np.zeros((N_eqns,3,1,N_layers+1))
  cnt = 0                     # skip header information (number of layers)
  for k in range(N_layers+1): # layer loop
    cnt += 1                  # skip header information (grid/time info)
    for i in range(3):        # norm loop
      cnt += 1
      dat[:,i,0,k] = np.array([float(item) for item in text_lines[cnt]]) # eqns
  return dat, N, t, N_iterations, N_layers


def read_err_norms_ic(text_lines):
  N_iterations = int(text_lines[0][0]) # assume that this integer is the number of iterative corrections
  # N_layers   = int(text_lines[1][0])
  N            = int(text_lines[2][0]) # this line should have grid size etc.
  t = []
  N_layers = []
  has_t = False
  if len(text_lines[2]) == 3:
    has_t = True

  N_eqns = len(text_lines[3]) # should be first line with error norms
  dat = np.zeros((N_eqns,3,N_iterations+1,1))
  cnt = 0
  for j in range(N_iterations+1): # iterative correction loop
    cnt += 1                      # skip header information (iteration counter)
    if cnt>len(text_lines): break
    cnt += 1                      # skip header information (grid/time info)
    if cnt>len(text_lines): break
    if has_t:
      t.append(float(text_lines[cnt][2]))
    for i in range(3):          # norm loop
      cnt += 1
      if cnt>len(text_lines): break
      dat[:,i,j,0] = np.array([float(item) for item in text_lines[cnt]]) # eqns
  return dat, N, t, N_iterations, N_layers


def read_err_norms_ic_offset(text_lines):
  N_iterations = int(text_lines[0][0]) # assume that this integer is the number of iterative corrections
  N_layers     = int(text_lines[1][0]) # assume that this integer is the number of excluded layers (+1)
  # skip 1 more (iteration coutner)
  N            = int(text_lines[3][0]) # this line should have grid size etc.
  t = []
  has_t = False
  if len(text_lines[3]) == 3:
    has_t = True
  N_eqns = len(text_lines[4]) # should be first line with error norms
  dat = np.zeros((N_eqns,3,N_iterations+1,N_layers+1))
  cnt = 0
  cnt += 1                        # skip header information (number of iterative corrections)
  for j in range(N_iterations+1): # iterative correction loop
    cnt += 1                      # skip redundant header information (iterations)
    if cnt>len(text_lines): break
    for k in range(N_layers+1):   # layer loop
      cnt += 1                    # skip redundant header information (number of layers)
      if cnt>len(text_lines): break
      if has_t and k==0:
        t.append(float(text_lines[cnt][2]))
      for i in range(3):          # norm loop
        cnt += 1
        if cnt>len(text_lines): break
        dat[:,i,j,k] = np.array([float(item) for item in text_lines[cnt]]) # eqns
  return dat, N, t, N_iterations, N_layers

def parametric_refinement_forces(run_dir,subdir_pattern1="*",subdir_pattern2="*",verbose=True):
  Ftmp = [] # initialize temporary list
  foldernames = []
  for i, subdir in enumerate(sorted(Path(run_dir).glob(f"{subdir_pattern1}"))):
    print(f"{subdir}")
    foldernames.append( f"{subdir.parent.name}:{subdir.name}")
    Ftmp.append(refinement_forces(subdir,subdir_pattern=subdir_pattern2,verbose=verbose))
  # make a new dictionary, stacking all of the error norms for each error type
  F = dict.fromkeys([key for key in Ftmp[0]])
  F["N1"] = Ftmp[0]["N1"]
  for key in Ftmp[0]: # note: this won't work if they are different sizes
    if key=="N1": continue
    arrays = [d[key] for d in Ftmp]
    max_shape = np.max([arr.shape for arr in arrays], axis=0)
    padded_arrays = []
    for arr in arrays:
      pad_width = [(0, max_shape[i] - arr.shape[i]) for i in range(len(max_shape))]
      padded_arr = np.pad(arr, pad_width, mode='constant', constant_values=np.NaN)
      padded_arrays.append(padded_arr)
    F[key] = np.stack(padded_arrays,axis=0)
  return F


def refinement_forces(run_dir,subdir_pattern="*",verbose=False):
  Ftmp = [] # initialize temporary list for force dictionaries
  N1   = [] # initialize temporary list for cell count
  # append error norm dicts from each subdirectory
  for subdir in sorted(Path(run_dir).glob(f"{subdir_pattern}")):
    if subdir.is_dir():
      Ftmp.append(single_case_forces(subdir,verbose=verbose))
      N1.append(get_grid_size(subdir,verbose=verbose))

  Ftmp = [val for _, val in sorted(zip(N1, Ftmp))]

  # make a new dictionary, stacking all of the forces 

  F = {key: {blk: {var: [] for var in Ftmp[0][key][blk]} for blk in Ftmp[0][key]} for key in Ftmp[0]}

  # zones[zone][var] = data
  for key in Ftmp[0]:
    for blk in Ftmp[0][key]:
      for var in Ftmp[0][key][blk]:
        F[key][blk][var] = np.stack([d[key][blk][var] for d in Ftmp])
  F['N1'] = np.array(N1)
  return F

def get_grid_size(run_dir,verbose=False):
  dir = Path(run_dir).resolve()
  patterns = ("")
  file_list = [file_path for file_path in run_dir.glob("*.grd")
               if not file_path.name.endswith("coarse.grd")]
  if len(file_list) < 1:
    raise RuntimeError(f".grd file not found in {run_dir}")
  elif len(file_list) > 1:
    if verbose:
      warnings.warn(f"multiple .grd files found, using first (alphabetical)")
  file = sorted(file_list)[0]
  try:
    ginfo = read_P3D_header(Path(file).resolve())
  except:
    raise RuntimeError(f"Error reading {file}")
  # read namelist to get grid_skip (if present)
  n_cells = ginfo['n_cells']
  try:
    pattern = re.compile(r"GRID_NSKIP", re.IGNORECASE)
    with open( ( run_dir / "sensei.nml" ).resolve(),"r" ) as f:
      for line in f:
        if pattern.search(line):
          sz = re.findall(r'\d+', line)
          sz = np.array([int(val) for val in sz])
          v  = np.prod(sz[sz>0])
          n_cells = n_cells/v
          return n_cells
  except:
    return n_cells

def read_P3D_header(filename):
  n_dim    = 0
  n_blocks = 0
  n_nodes  = 0
  n_cells  = 0
  sizes    = []
  idx      = []
  with open(filename,"r") as f:
    numbers = [int(a) for a in f.readline().split()]
    if len(numbers)==1:
      n_blocks = numbers[0]
      numbers = [int(a) for a in f.next(f).split()]
      n_dim=len(numbers)
      sizes.append((numbers))
      n_nodes += np.prod(numbers)
      n_cells += np.prod([n-1 for n in numbers if n>1])
      idx.append(n_nodes)
      for i in range(1,n_blocks):
        numbers = [int(a) for a in next(f).split()]
        sizes.append((numbers))
        n_nodes += np.prod(numbers)
        n_cells += np.prod([n-1 for n in numbers if n>1])
        idx.append(n_nodes)
    else:
      n_blocks = 1
      n_dim=len(numbers)
      sizes.append((numbers))
      n_nodes += np.prod(numbers)
      n_cells += np.prod([n-1 for n in numbers if n>1])
      idx.append(n_nodes)
  return {
          "n_blocks": n_blocks,
          "n_dim": n_dim,
          "sizes": sizes,
          "n_nodes": n_nodes,
          "n_cells": n_cells,
          "idx": idx,
          }

def single_case_forces(run_dir,verbose=False):
  run_dir = Path(run_dir)
  file_list = [file_path for file_path in run_dir.glob("*-force_history.dat")]
  F = {}
  for file_path in file_list:
    key = file_path.name.removesuffix("*-force_history.dat")
    F[key] = read_force_history_file(file_path,verbose=verbose)
  return F

def read_force_history_file(file_name,verbose=False):
  file_name = Path(file_name).resolve()
  with open(file_name, "r") as file:
    text_lines = file.read().splitlines()
  # for i,text_line in enumerate(text_lines):
  #   text_lines[i] = text_line.split()
  if verbose:
    print(f"Reading {file_name}")
  var_line = ""
  # First line should be variables
  for line in text_lines:
    if line.strip().casefold().startswith(("variables").casefold()):
      try:
        variables = re.findall(r'"([^"]*)"', line)
      except:
        raise RuntimeError(f"Error reading variable list from {file_name} on line {i}")
      break
  # now find zones
  zone_lines = []
  zones      = []
  for i, line in enumerate(text_lines):
    if line.strip().casefold().startswith(("zone").casefold()):
      try:
        zone_str = re.findall(r'"([^"]*)"', line)
        if len(zone_str) != 1:
          warnings.warn(f"unrecognized formatting for zone string on line {i}")
          zones.append(f"zone_{i:02d}") # there should only be one match
        else:
          zones.append(zone_str[0]) # there should only be one match
        for j in range(i,len(text_lines)): # find where data starts
          if text_lines[j].strip()[0].isdigit():
            zone_lines.append(j)
            break
          if j==len(text_lines):
            raise RuntimeError(f"No data found for zone {zones[-1]}")
      except:
        raise RuntimeError(f"Error reading zone name {file_name} on line {i}")
  zone_lines.append(len(text_lines))
  var_dict = {var: [] for var in variables}
  zones = {zone: var_dict for zone in zones}

  for i, zone in enumerate(zones):
    data = np.stack([[float(item) for item in text_lines[j].split()] for j in range(zone_lines[i],zone_lines[i+1])])
    for j, var in enumerate(var_dict):
      zones[zone][var] = data[:,j]
  return zones

# if __name__ == "__main__":
#   DE = refinement_error_norms('/mnt/c/Users/wajordan/Desktop/CASES/COPY_parabola_test_NEW_AR_1_BD_div_8_P2_CBC=F_NO_BC_2026-10-14_19.21.19',verbose=False)
#   output_hdf5_dict(DE,'test.h5')
# if __name__ == "__main__":
#   DE = parametric_refinement_error_norms('/mnt/c/Users/wajordan/Desktop/CASES/',subdir_pattern1="*ms1_test_09172026_2026-09-17*",subdir_pattern2="*")
#   output_hdf5_dict(DE,'test.h5')
# if __name__ == "__main__":
#   Z = read_force_history_file('/home/wajordan/SENSEI_Practice/DEBUG/airfoil/_kt0129x0033_HO-dev/kt-001-ic-force_history.dat',verbose=True)
#   output_hdf5_dict(Z,'test_forces.h5')

if __name__ == "__main__":
  # Z = single_case_forces('/home/wajordan/pySENSEI/JOUKOWSKY_C_GRID/_kt0129x0033',verbose=True)
  # Z = refinement_forces('/home/wajordan/pySENSEI/JOUKOWSKY_C_GRID',verbose=True)
  # output_hdf5_dict(Z,'test_forces.h5')
  # checking with h5diff
  # Z2 = read_hdf5_dict('test_forces.h5')
  # output_hdf5_dict(Z2,'test_forces2.h5')
  F = parametric_refinement_forces('/mnt/c/Users/wajordan/Desktop/CASES/',subdir_pattern1="*ms1_test_09172026_2026-09-17*",subdir_pattern2="*")
  output_hdf5_dict(F,'test_forces.h5')
  
  