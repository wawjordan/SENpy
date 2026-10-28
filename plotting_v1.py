# standard library imports
from pathlib import Path
from sys import stderr, stdout
import zipfile
import warnings
import pdb
import re
import itertools

# third party imports
import numpy as np
import h5py
import matplotlib.pyplot as plt

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


def plot_error_norm_line( N: np.ndarray, DE_slice: np.ndarray, ndim=None, ax=None, **kwargs ):
  # N: grid refinement measure (e.g. (N_cells)^(1/2)) (slice of ndarray)
  # DE_slice: slice of ndarray
  # ndim: number of spatial dimensions for the simulation
  # ax: axes object
  if ax is None:
    ax = plt.gca()
  DE = DE_slice.flatten()
  DE[DE==0] = np.nan
  if ndim is None:
    ax.plot( N.flatten(), DE, **kwargs)
  else:
    ax.plot( N.flatten()**(1./ndim), DE, **kwargs)
  return ax

def plot_ooa_line( N: np.ndarray, DE_slice: np.ndarray, ndim=None, r=2, ax=None, **kwargs ):
  # N: grid refinement measure (e.g. (N_cells)^(1/2)) (slice of ndarray)
  # DE_slice: slice of ndarray
  # ndim: number of spatial dimensions for the simulation
  # r: refinement factor; assumed as 2 unless specified
  # ax: axes object
  if ax is None:
    ax = plt.gca()
  DE = DE_slice.flatten()
  OOA = np.full_like(DE, np.nan)
  with np.errstate(invalid='ignore', divide='ignore'): # suppress NaN warnings for this calc
    for i in range(1, len(DE)):
      OOA[i] = np.log( DE[i-1] / DE[i] ) / np.log(r)
      OOA[i] = np.nan_to_num( OOA[i], nan=np.nan, posinf=np.nan, neginf=np.nan)

  if ndim is None:
    ax.plot( N.flatten(), OOA, **kwargs)
  else:
    ax.plot( N.flatten()**(1./ndim), OOA, **kwargs)
  return ax


def preplot_from_param_refine_DE_dict(DE_dict,DE_key=None,cases=None, levels=None, vars=0,norms=0,iters=0,layers=0):
  # preprocessing arguments for plotting DE norms or OOA
  if DE_key is None:
    DE_key_ = ['DE_primal_primitive']
  else:
    DE_key_ = DE_key if isinstance(DE_key, list) else [DE_key]

  n_cases  = DE_dict[DE_key_[0]].shape[0]
  n_levels = DE_dict[DE_key_[0]].shape[1]
  n_iters = 1
  for key in DE_key_:
    n_iters = max( n_iters, DE_dict[key].shape[4] )
  if cases is None:
    cases_ = np.arange(n_cases)
  else:
    cases_ = cases if isinstance(cases, list) else [cases]

  if levels is None:
    levels_ = np.arange(n_levels)
  else:
    levels_ = levels if isinstance(levels, list) else [levels]
  
  vars_   = vars if isinstance(vars, list) else [vars]
  norms_  = norms if isinstance(norms, list) else [norms]
  layers_ = layers if isinstance(layers, list) else [layers]

  iters_tmp = []
  if not isinstance(iters, list): # simplest case, a single scalar input
    iters_tmp = [[iters] for _ in DE_key_]
  else:
    if len(iters)!=len(DE_key_):
      raise ValueError(f"number of iteration inputs should be scalar or match the number of DE_keys")
    if not isinstance(iters[0], list): # if it is a single level list
      iters_tmp = [[iter] for iter in iters]
    else:
      if isinstance(iters[0][0],list): # if it is a list of list of lists
        raise ValueError(f"iteration inputs may be at most doubly nested")
      iters_tmp = iters # otherwise, should be fine
  iters_tmp2 = [ np.arange(n_iters) if (-2 in iter) else iter for iter in iters_tmp ]

  iters_ = [ [0] if 'primal' in key else iters_tmp2[i] for i, key in enumerate(DE_key_)]

  return DE_key_, cases_, levels_, vars_, norms_, iters_, layers_

def collect_legend_info( DE_key, cases, vars, norms, iters, layers ):
  prim_var_text = { 0 : "$\rho$",
                      1 : "$u$",
                      2 : "$v$",
                      3 : "$w$",
                      4 : "$P$",
                      5 : "$\tilde{\nu}$",
                    }
  cons_var_text = { 0 : "$\rho$",
                    1 : "$\rho u$",
                    2 : "$\rho v$",
                    3 : "$\rho w$",
                    4 : "$\rho e$",
                    5 : "$\rho \tilde{\nu}$",
                  }
  type_info = {"Reconstruction_error"           : ["rec. error",prim_var_text],
                "TE_exact"                       : ["TE (exact)",cons_var_text],
                "TE_estimated"                   : ["TE (est.)",cons_var_text],
                "TE_error"                       : ["TE (error)",cons_var_text],
                "DE_primal_primitive"            : ["primal DE",prim_var_text],
                "DE_primal_primitive_flagged"    : ["primal DE (flagged)",prim_var_text],
                "DE_primal_conserved"            : ["primal DE",cons_var_text],
                "DE_primal_conserved_flagged"    : ["primal DE (flagged)",cons_var_text],
                "DE_corrected_primitive"         : ["corr. DE",prim_var_text],
                "DE_corrected_primitive_flagged" : ["corr. DE (flagged)",prim_var_text],
                "DE_corrected_conserved"         : ["corr. DE",cons_var_text],
                "DE_corrected_conserved_flagged" : ["corr. DE (flagged)",cons_var_text],
              }
  norm_text = { 0 : "Norm: $L_{1}$",
                1 : "Norm: $L_{2}$",
                2 : "Norm: $L_{\infty}$",
              }
  colors = ["#1171BE","#DD5400","#EDB120","#8516D1","#3BAA32","#2FBEEF","#D1048B"]
  markers = ['o', 's', 'd', '^', 'v', '<', '>', '1', '2', '3', '4', 'p', '*', 'h', '+', 'x']
  lines   = ['-','--','-.',':']
  alphas = [ np.full_like(iter, 1.,dtype='float') for iter in iters ]
  for j,a in enumerate(alphas):
    if a.size > 2:
      alphas[j][1:-2] = 0.5

  type_map = lines
  cycle_dict = { "color"  : colors,
                 "line"   : lines,
                 "marker" : markers,
                 "null"   : [None]
              }
  dim_dict  = { "group" : "color",
                "case"  : "marker",
                "var"   : "null",
                "norm"  : "line",
                "layer" : "null",
                "iter"  : "null",
              }
  
  # labels    = []
  # cyc1 = itertools.cycle(cycle_dict[dim_dict["group"]])
  # for j,key in enumerate(DE_key): # group loop
  #   c1 = next(cyc1)
  #   cyc2 = itertools.cycle(cycle_dict[dim_dict["case"]])
  #   for case_num, _ in enumerate(cases): # case loop
  #     c2 = next(cyc2)
  #     cyc3 = itertools.cycle(cycle_dict[dim_dict["var"]])
  #     for var in vars: # var loop
  #       c3 = next(cyc3)
  #       cyc4 = itertools.cycle(cycle_dict[dim_dict["norm"]])
  #       for norm in norms: # norm loop
  #         c4 = next(cyc4)
  #         cyc5 = itertools.cycle(cycle_dict[dim_dict["iter"]])
  #         for i, iter in enumerate(iters[j]): # iter loop
  #           c5 = next(cyc5)
  #           cyc6 = itertools.cycle(cycle_dict[dim_dict["layer"]])
  #           for layer in layers: # layer loop
  #             c6 = next(cyc6)
  #             labels.append({ "key_name"   : key,
  #                             "case_num"   : case_num,
  #                             "type_name"  : type_info[key][0],
  #                             "var_name"   : type_info[key][1][var],
  #                             "norm_name"  : norm_text[norm],
  #                             "iteration"  : iter,
  #                             "layer_num"  : layer,
  #                             "color"      : color,
  #                             "marker"     : marker,
  #                             "line_style" : line,
  #                             "line_alpha" : alphas[j][i]
  #                           } )
  labels    = []
  line_cyle = itertools.cycle(lines)
  for j,key in enumerate(DE_key): # group loop
    line = next(line_cyle)
    marker_cycle = itertools.cycle(markers[0:min(len(cases),len(markers))])
    for case_num, _ in enumerate(cases): # case loop
      marker = next(marker_cycle)
      color_cycle  = itertools.cycle(colors[0:len(vars)])
      for var in vars: # var loop
        color  = next(color_cycle)
        for norm in norms: # norm loop
          for i, iter in enumerate(iters[j]): # iter loop
            for layer in layers: # layer loop
              labels.append({ "key_name"   : key,
                              "case_num"   : case_num,
                              "type_name"  : type_info[key][0],
                              "var_name"   : type_info[key][1][var],
                              "norm_name"  : norm_text[norm],
                              "iteration"  : iter,
                              "layer_num"  : layer,
                              "color"      : color,
                              "marker"     : marker,
                              "line_style" : line,
                              "line_alpha" : alphas[j][i]
                            } )
  return labels

def plot_ERR_from_param_refine_dict(DE_dict,DE_key=None,r=2,ooa=False,ndim=None,cases=None, levels=None, vars=0,norms=0,iters=0,layers=0,ax=None,**kwargs):
  # DE_dict: dictionary from the function 'parametric_refinement_error_norms'
  # DE_key: which field in DE_dict will be plotted (defaults to 'DE_primal_primitive')
  # ndim: number of spatial dimensions for the simulation
  # cases: which cases to plot (default to all)
  # levels: which grid levels to plot (default to all)
  # vars: which variables to plot (default to all)
  # norms: which norms to plot (default to 1)
  # iters: which iterative corrections to plot (default to last)
  # layers: which number of excluded layers (default to no excluded layers)
  # n_dim: number of spatial dimensions for the simulation
  # r: refinement factor; assumed as 2 unless specified
  # ax: axes object
  DE_key_, cases_, levels_, vars_, norms_, iters_, layers_ = preplot_from_param_refine_DE_dict( 
                             DE_dict,DE_key=DE_key,cases=cases, levels=levels, vars=vars,norms=norms,iters=iters,layers=layers )
  labels = collect_legend_info( DE_key_, cases_, vars_, norms_, iters_, layers_ )
  # labels_ = find_unique_subsets(labels)
  DE = [np.array(DE_dict[key]) for key in DE_key_]
  N  = np.array(DE_dict['N1'])
  cnt = -1
  for j,de in enumerate(DE):
    for case in cases_:
      for var in vars_:
        for norm in norms_:
          for iter in iters_[j]:
            for layer in layers_:
              cnt += 1
              if ooa:
                ax = plot_ooa_line( N, 
                                    de[case,levels_,var,norm,iter,layer], 
                                    r=r, 
                                    ndim=ndim, 
                                    ax=ax, 
                                    color=labels[cnt]["color"], 
                                    marker=labels[cnt]["marker"], 
                                    linestyle=labels[cnt]["line_style"], 
                                    alpha=labels[cnt]["line_alpha"],
                                    **kwargs )
              else:
                ax = plot_error_norm_line(N, 
                                          de[case,levels_,var,norm,iter,layer], 
                                          ndim=ndim, 
                                          ax=ax, 
                                          color=labels[cnt]["color"], 
                                          marker=labels[cnt]["marker"], 
                                          linestyle=labels[cnt]["line_style"], 
                                          alpha=labels[cnt]["line_alpha"],
                                          **kwargs )
  ax.set_xscale('log')
  if not ooa:
    ax.set_yscale('log')
  return ax, labels

def find_unique_subsets(data_list):
    unique_subsets = set()
    result = []
    
    for d in data_list:
        # Sort items so that {'a': 1, 'b': 2} matches {'b': 2, 'a': 1}
        fs = frozenset(d.items())
        if fs not in unique_subsets:
            unique_subsets.add(fs)
            result.append(dict(fs))
            
    return result

if __name__ == "__main__":
  filename ='/mnt/c/Users/Will/Downloads/test.h5'
  DE_dict = read_hdf5_dict(filename)
  DE_key = ['DE_primal_primitive','DE_corrected_primitive']
  cases  = 4 #[i for i in range(5)]
  norms  = [0,1,2]
  iters  = [[0],[0,-1]]
  ndim   = 2
  markersize = 8
  fillstyle='none'
  fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(7.5, 5.5))

  for i,ooa in enumerate([False,True]):
    axs[i], labels = plot_ERR_from_param_refine_dict( DE_dict, 
                                              DE_key=DE_key,
                                              cases=cases,
                                              norms=norms,
                                              iters=iters,
                                              ndim=ndim,
                                              markersize=markersize,
                                              fillstyle=fillstyle,
                                              ooa=ooa,
                                              ax=axs[i] )
  axs[0].set_xlim(10,1000)
  axs[1].set_xlim(10,1000)

  axs[0].set_ylim(10.**np.array([-11,-2]))
  axs[1].set_ylim(0,6)
  # ax.legend(loc='upper right')
  plt.show(block=False)
  dumb = 1