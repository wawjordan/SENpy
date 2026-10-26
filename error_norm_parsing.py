# standard library imports
import shutil
import subprocess
from pathlib import Path
from sys import stderr, stdout
import zipfile

# third party imports
import pdb
import numpy as np
import warnings
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

# def parametric_refinement_error_norms(run_dir,subdir_pattern1="*",subdir_pattern2="*"):
#   error_norms = [] # initialize temporary list
#   for subdir in sorted(Path(run_dir).glob(f"{subdir_pattern1}")):
#     error_norms.append(refinement_error_norms(subdir,subdir_pattern=subdir_pattern2))
#   # make a new dictionary, stacking all of the error norms for each error type
#   DE = dict.fromkeys([key for key in error_norms[0]])
#   DE["t"] = []
#   for key in error_norms[0]: # note: this won't work if they are different sizes
#     DE[key] = np.stack([d[key] for d in error_norms])
#   breakpoint()
#   for i in range(len(error_norms)):
#     DE["t"].append( error_norms[i]["t"] )
#     breakpoint()
#   DE["N1"] = error_norms[0]["N1"]
#   DE["N_layers"] = error_norms[0]["N_layers"]


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
          error_norms[key] = read_err_norm_file_to_dict(first_match.resolve(),verbose=verbose)
          break
        except:
          warnings.warn(f"Error reading from {first_match}")
    if error_norms[key] is None:
      error_norms[key] = empty_error_norm

  return error_norms



def read_err_norm_folder(folder, pattern, dim, r_fac,verbose=False):
  DE = []
  for file_name in sorted(Path(folder).glob(f"**/{pattern}")):
      dat, N, t, N_layers, N_iterations = read_err_norm_file(file_name,verbose=verbose)
      DE.append({
          "N1": N,
          "N2": float(N) ** (1 / dim),
          "t": t,
          "N_iterations": N_iterations,
          "N_layers": N_layers,
          "E": dat,
      })
  DE.sort(key=lambda d: d["N1"]) # sort list of dicts by key "N1" ascending
  DE = calc_OOA(DE, r_fac)
  return DE

def calc_OOA(DE, r_fac):
  DE[0]["OOA"] = np.full_like(DE[0]["E"], np.nan)   # (analagous to matlab nan*DE(1).E)
  with np.errstate(invalid='ignore', divide='ignore'): # suppress NaN warnings for this calc
    for i in range(1, len(DE)):
        DE[i]["OOA"] = np.log( DE[i-1]["E"] / DE[i]["E"] ) / np.log(r_fac)
        DE[i]["OOA"] = np.nan_to_num( DE[i]["OOA"], nan=np.nan, posinf=np.nan, neginf=np.nan)
  return DE

def to_columnar(DE):
    # re-arrange to be a dict of lists/arrays
    return {
        "N1":           np.array([d["N1"] for d in DE]),
        "N2":           np.array([d["N2"] for d in DE]),
        "t":            [d["t"] for d in DE],
        "N_iterations": np.array([d["N_iterations"] for d in DE]),
        "N_layers":     np.array([d["N_layers"] for d in DE]),
        "E":            np.stack([d["E"] for d in DE]),
        "OOA":          np.stack([d["OOA"] for d in DE]),
    }

def read_err_norm_file_to_dict(file_name,verbose=False):
  dat, N, t, N_iterations, N_layers = read_err_norm_file(file_name,verbose=verbose)
  return {
          "N": N,
          "t": t,
          "N_iterations": N_iterations,
          "N_layers": N_layers,
          "E": dat,
         }

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
  return dat, N, t, N_iterations, N_layers

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


# if __name__ == "__main__":
#   DE = refinement_error_norms('/mnt/c/Users/wajordan/Desktop/CASES/COPY_parabola_test_NEW_AR_1_BD_div_8_P2_CBC=F_NO_BC_2026-10-14_19.21.19',verbose=False)
#   output_hdf5_dict(DE,'test.h5')