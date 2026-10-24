# standard library imports
import shutil
import subprocess
from pathlib import Path
from sys import stderr, stdout
import zipfile

# third party imports
import pdb
import numpy as np
# import yaml
# import h5py
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
# TENSORDIM = 9
# find . -maxdepth 1 -mindepth 1 -type d -exec touch {}/filename.rst \;
# find ./*/ -type f -name "*.nml" -delete

def read_err_norm_file(file_name):
  file_name = Path(file_name).resolve()
  with open(file_name, "r") as file:
    text_lines = file.read().splitlines()
  for i,text_line in enumerate(text_lines):
    text_lines[i] = text_line.split()

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



def collect_norm_files(run_dir,verbose=False):
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
                    "-de_primal_primitive_flagged.dat",
                    "-de_primal_conserved_flagged.dat",
                    "-de_corrected_primitive_flagged.dat",
                    "-de_corrected_conserved_flagged.dat",
                  )
  dat_file_list = [file_path for file_path in run_dir.glob("*.dat")
                   if file_path.name.endswith(file_patterns)]

  # create log file to hold output from tec360 batch.log output files
  log_file = run_dir / "tec360_batch.log"
  tmp_log_file = run_dir / "batch.log"
  with open(log_file, "w", encoding="utf-8") as file:
    file.close()

  # loop over the dat files
  # for file_path in dat_file_list:



def read_err_norm_folder(folder, pattern, dim, r_fac):
  DE = []
  for file_name in sorted(Path(folder).glob(f"**/{pattern}")):
      dat, N, t, N_layers, N_iterations = read_err_norm_file(file_name)
      DE.append({
          "N1": N,
          "N2": float(N) ** (1 / dim),
          "t": t,
          "N_iterations": N_iterations,
          "N_layers": N_layers,
          "E": dat,
      })
  DE.sort(key=lambda d: d["N1"])
  DE = calc_OOA(DE, r_fac)
  return DE

def calc_OOA(DE, r_fac):
  N_grids = len(DE)
  DE[0]["OOA"] = np.full_like(DE[0]["E"], np.nan)   # nan*DE(1).E
  for i in range(1, N_grids):
      DE[i]["OOA"] = np.where( DE[i]["E"] > 0,
                               np.log(Etmp) / np.log(r_fac),
                               np.nan)
  return DE