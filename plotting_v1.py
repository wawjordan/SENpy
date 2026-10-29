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
import matplotlib.ticker as ticker

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

def collect_legend_info( DE_key, cases, vars, norms, iters, layers, mapping_dict ):
  # mapping_dict : provides a map for [colors/lines/markers] to data
  # Example:
  # mapping_dict  = { "group" : "color",
  #                   "case"  : "marker",
  #                   "var"   : "null",
  #                   "norm"  : "line",
  #                   "layer" : "null",
  #                   "iter"  : "null",
  #                 }
  prim_var_text = { 0 : r"$\rho$",
                    1 : r"$u$",
                    2 : r"$v$",
                    3 : r"$w$",
                    4 : r"$P$",
                    5 : r"$\tilde{\nu}$",
                    }
  cons_var_text = { 0 : r"$\rho$",
                    1 : r"$\rho u$",
                    2 : r"$\rho v$",
                    3 : r"$\rho w$",
                    4 : r"$\rho e$",
                    5 : r"$\rho \tilde{\nu}$",
                  }
  type_info = { "Reconstruction_error"           : ["rec. error",prim_var_text],
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

  cycle_dict = { "color"     : colors,
                 "linestyle" : lines,
                 "marker"    : markers,
                 "null"      : [None]
              }

  # invert the mapping_dict
  # this takes a line style and outputs a the dimension it maps to
  style_dict = { key: [None] for key in cycle_dict }
  for key, value in mapping_dict.items():
    style_dict[value] = key

  # copy mapping_dict to be container for the cycle components
  map_dict = mapping_dict.copy()
  
  labels    = []
  cyc1 = itertools.cycle(cycle_dict[mapping_dict["group"]])
  for j,key in enumerate(DE_key): # group loop
    map_dict["group"] = next(cyc1)
    cyc2 = itertools.cycle(cycle_dict[mapping_dict["case"]])
    for case_num, _ in enumerate(cases): # case loop
      map_dict["case"] = next(cyc2)
      cyc3 = itertools.cycle(cycle_dict[mapping_dict["var"]])
      for var in vars: # var loop
        map_dict["var"] = next(cyc3)
        cyc4 = itertools.cycle(cycle_dict[mapping_dict["norm"]])
        for norm in norms: # norm loop
          map_dict["norm"] = next(cyc4)
          cyc5 = itertools.cycle(cycle_dict[mapping_dict["iter"]])
          for i, iter in enumerate(iters[j]): # iter loop
            map_dict["iter"] = next(cyc5)
            cyc6 = itertools.cycle(cycle_dict[mapping_dict["layer"]])
            for layer in layers: # layer loop
              map_dict["layer"] = next(cyc6)
              labels.append({ "key_name"   : key,
                              "case_num"   : case_num,
                              "type_name"  : type_info[key][0],
                              "var_name"   : type_info[key][1][var],
                              "norm_name"  : norm_text[norm],
                              "iteration"  : iter,
                              "layer_num"  : layer,
                              "color"      : map_dict[style_dict["color"]],
                              "marker"     : map_dict[style_dict["marker"]],
                              "line_style" : map_dict[style_dict["linestyle"]],
                              "line_alpha" : alphas[j][i]
                            } )
  lbls = []
  if mapping_dict["group"] != "null":
    cyc1 = itertools.cycle(cycle_dict[mapping_dict["group"]])
    for j,key in enumerate(DE_key): # group loop
      lbls.append({type_info[key][0] : {mapping_dict["group"]:next(cyc1)} })
  if mapping_dict["case"] != "null":
    cyc2 = itertools.cycle(cycle_dict[mapping_dict["case"]])
    for case_num in cases: # case loop
      lbls.append({f"case {case_num}" : {mapping_dict["case"]:next(cyc2)} })
  if mapping_dict["var"] !="null":
    unique_vars = list(dict.fromkeys(d["var_name"] for d in labels if "var_name" in d))
    cyc3 = itertools.cycle(cycle_dict[mapping_dict["var"]])
    for i, var in enumerate(vars): # var loop
      lbls.append({unique_vars[i] : {mapping_dict["var"]:next(cyc3)} })
  if mapping_dict["norm"] !="null":
    cyc4 = itertools.cycle(cycle_dict[mapping_dict["norm"]])
    for norm in norms: # norm loop
      lbls.append({norm_text[norm] : {mapping_dict["norm"]:next(cyc4)} })
  if mapping_dict["iter"] !="null":
    cyc5 = itertools.cycle(cycle_dict[mapping_dict["iter"]])
    for i, iter in enumerate(iters[j]): # iter loop
      if iter==-1:
        lbls.append({"last iter" : {mapping_dict["iter"]:next(cyc5)} })
      else:
        lbls.append({f"iter {iter}" : {mapping_dict["iter"]:next(cyc5)} })
  if mapping_dict["layer"] !="null":
    cyc6 = itertools.cycle(cycle_dict[mapping_dict["layer"]])
    for layer in layers: # layer loop
      lbls.append({f"layer {layer}" : {mapping_dict["layer"]:next(cyc6)} })
  return labels, lbls

def plot_ERR_from_param_refine_dict(DE_dict,mapping_dict,DE_key=None,r=2,ooa=False,ndim=None,cases=None, levels=None, vars=0,norms=0,iters=0,layers=0,ax=None,**kwargs):
  # DE_dict: dictionary from the function 'parametric_refinement_error_norms'
  # mapping_dict: dictionary that specifies what linestyle element applies to which variables
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
  labels, lbls = collect_legend_info( DE_key_, cases_, vars_, norms_, iters_, layers_, mapping_dict )
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
  if len(norms)==1:
      ax.text(0.05, 0.95, labels[0]["norm_name"], transform=ax.transAxes, fontsize=12,
                                                  verticalalignment='top')
  
  for lbl in lbls:
    for key, value in lbl.items():
      if len(norms)==1 and 'Norm' in key:
        continue
      if "color" in value:
        ax.plot( np.nan, np.nan, label=key, **value, **kwargs)
      else:
        ax.plot( np.nan, np.nan, color="black", label=key, **value, **kwargs)
  ax.set_xscale('log')
  ax.set_axisbelow(True)
  ax.grid(True, which='both', axis='both')
  ax.grid(which='major', linestyle='-', linewidth=0.8, color='black')
  ax.grid(which='minor', linestyle=':', linewidth=0.5, color='gray')
  if ndim is not None:
    x_label = r"$(n_{cells})^{1/" + f"{ndim}" + r"}$"
    ax.set_xlabel(fr"{x_label}")
  else:
    ax.xlabel(r"n_{cells}")
  if ooa:
    ax.set_ylabel('Observed Order of Accuracy')
  else:
    ax.set_ylabel('Error Norm')
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
  DE_key = ['DE_corrected_primitive','DE_corrected_primitive']
  cases  = [0,8] #[i for i in range(8)]
  vars   = [0,1]
  norms  = [0]
  iters  = [[0],[200]]
  ndim   = 2
  markersize = 8
  fillstyle='none'
  # mapping_dict  = { "group" : "color",
  #                   "case"  : "null",
  #                   "var"   : "null",
  #                   "norm"  : "linestyle",
  #                   "layer" : "null",
  #                   "iter"  : "marker",
  #                 }
  mapping_dict  = { "group" : "null",
                    "case"  : "marker",
                    "var"   : "color",
                    "norm"  : "linestyle",
                    "layer" : "null",
                    "iter"  : "null",
                    }
  fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(7.5, 5.5))

  for i,ooa in enumerate([False,True]):
    axs[i], labels = plot_ERR_from_param_refine_dict( DE_dict,
                                              mapping_dict,
                                              DE_key=DE_key,
                                              cases=cases,
                                              vars=vars,
                                              norms=norms,
                                              iters=iters,
                                              ndim=ndim,
                                              markersize=markersize,
                                              fillstyle=fillstyle,
                                              ooa=ooa,
                                              ax=axs[i] )
  axs[0].set_xlim(10,1000)
  axs[1].set_xlim(10,1000)

  start_exp = -11
  end_exp   = -2
  start_order = 0
  end_order   = 6
  ylimits = 10.**np.array([start_exp,end_exp])
  axs[0].set_ylim(ylimits)
  axs[0].yaxis.set_minor_formatter(ticker.NullFormatter())
  axs[0].yaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=np.linspace(0.,1.,11), numticks=1000))
  axs[0].yaxis.set_major_locator(ticker.LogLocator(base=10.0, subs=(1.,), numticks=1000))

  axs[1].set_ylim(start_order,end_order)
  axs[1].yaxis.set_minor_formatter(ticker.NullFormatter())
  axs[1].set_yticks(np.linspace(start_order,end_order,2*(end_order-start_order)+1), minor=True)
  axs[1].legend(ncols=2,loc='lower center',framealpha=1.0)
  plt.show(block=False)
  dumb = 1