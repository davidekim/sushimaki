import os,sys
import numpy as np
import math
from math import sqrt,asin,pi,sin,tan,cos,atan
from scipy.optimize import fsolve
import glob
import argparse
import biolib

debug = 0
import warnings
warnings.filterwarnings('ignore')

# Dependencies
#
# PyRosetta
# https://www.pyrosetta.org
#
# For predicting the transmembrane region of the target protein
# DeepTMHMM
# Jeppe Hallgren, Konstantinos D. Tsirigos, Mads D. Pedersen, José Juan Almagro Armenteros, Paolo Marcatili, 
# Henrik Nielsen, Anders Krogh and Ole Winther (2022). DeepTMHMM predicts alpha and beta transmembrane proteins 
# using deep neural networks. https://doi.org/10.1101/2022.04.08.487609
# https://dtu.biolib.com/DeepTMHMM
# pip3 install pybiolib
#
# For building backbones of parametric beta barrel cylinders
# BBQ (requires Java). Jar files are provided in this repo. 
# Gront, D., Kmiecik, S. and Kolinski, A. (2007), Backbone building from quadrilaterals: A fast and accurate 
# algorithm for protein backbone reconstruction from alpha carbon coordinates. J. Comput. Chem., 28: 1593-1597. 
# https://doi.org/10.1002/jcc.20624
#
# Reference for WRAPs
#   Ljubica Mihaljević et. al. Solubilization of Membrane Proteins using designed protein WRAPS. Submitted to Science.
#
# Reference for parametrically guided beta barrel backbone design:
#   Kim DE. et al. 2024. Parametrically guided design of soluble beta barrels and transmembrane nanopores using deep learning.
#
# References for cylinder generation code:
#   Naveed H. et al. JACS, 2012. Predicting three-dimensional structures of transmembrane domains of beta-barrel membrane proteins.
#   Dou, J. et al. Nature, 2018. De novo design of a fluorescence-activating β-barrel. 
#

# Set CLASSPATH for BBQ 
installdir = os.path.dirname(os.path.abspath(__file__))
os.environ['CLASSPATH'] = f'{installdir}/bioshell.bioinformatics-2.2/bioshell.bioinformatics-2.2.jar:{installdir}/bioshell.bioinformatics-2.2/bioshell.bioinformatics-2.2-mono.jar'

## beta wrap parameters
intra_strand_dist =  3.8
inter_strand_dist =  4.8
beta_tmheight_buffer_max = 8
beta_tmheight_buffer_min = 5

# to determine gap distance range to sample between target and barrel wrap
beta_radius_buffer_max = 10
beta_radius_buffer_min = 8

## helix wrap parameters
rbuffer = 6 # to add to transmembrane radius calculation to determine number of helices
hbuffer = 5 # to add to tmheight for helix length

## General params

# For loop modeling, this is the maximum distance per residue between loop start and stop CA positions.
# If the gap is too large, the wrap will be skipped.
max_dist_per_loop_res = 4.5 

# len for inner and outer coords for superposition into cylinder
blen = 2

import textwrap
parser = argparse.ArgumentParser(
      formatter_class=argparse.RawDescriptionHelpFormatter,
      epilog=textwrap.dedent('''\

         additional information:
             By default:
                TM is the transmembrane segment predicted by DeepTMHMM.
                https://dtu.biolib.com/DeepTMHMM (pip3 install pybiolib)
                --n is calculated as a function of the approximate TM radius.
                --nres is calculated as a function of the approximate TM height.

         '''))

parser.add_argument('--n', type=int, default=0, help='Manually set n (helix count).')
parser.add_argument('--nres', type=int, default=0, help='Manually set nres (helix length).')
parser.add_argument('--radius', type=float, default=0, help='Manually set radius for helix wrap.')
parser.add_argument('--rot_angle', type=int, default=30, help='Manually set rotation sample angle.')
parser.add_argument('--h_rot_angle', type=float, default=0, help='Rotate angle of helices in wrap relative to target.')
parser.add_argument('--rot_n', type=int, default=3, help='Manually set rotation samples.')
parser.add_argument('--looplen', type=int, default=3, help='Loop length.')
parser.add_argument('--partial_diffusion_task_file_prefix', type=str, default="partial_diffusion_task_file", help='RF partial diffusion task file output prefix.')
parser.add_argument('--wrap', type=str, default='', help='Try to place this PDB as a wrap. i.e. this input is a pre-made wrap. --wrap_top_resi and --wrap_bottom_resi may be automatically assigned for helical up and down wraps using DSSP.')
parser.add_argument('--wrap_top_resi', type=str, default='', help='Wrap top comma separated resi for superposition.')
parser.add_argument('--wrap_bottom_resi', type=str, default='', help='Wrap bottom comma separated resi for superposition.')
parser.add_argument('--wrap_barrel_params', type=str, default='', help='Assign wrap_top_resi and wrap_bottom_resi based on comma separated barrel params n,nres,looplen,terminilen.')
parser.add_argument('--flip_wrap', default=False, action='store_true', help='Flip the wrap.')
parser.add_argument('--barrel', default=False, action='store_true', help='Wrap with a beta barrel. n, nres, and radius will be automatically sampled. Helices are used by default.')
parser.add_argument('--barrel_termini_len', type=int, default=3, help='N and C terminal extension length for barrel wraps.')

parser.add_argument('--rf_diffusion_container', type=str, default='python', help='Path to optional Apptainer for running RFDiffusion.')
parser.add_argument('--rf_diffusion', type=str, default=f'{installdir}/ppi_iterative_opt/rf_diffusion/run_inference.py', help='Path to RFDiffusion run_inference.py.')
parser.add_argument('--rf_partial_diffusions', type=int, default=10, help='Number of partial RF diffusion trajectories.')
parser.add_argument('--rf_diffusion_partialT', type=int, default=30, help='RF diffusion partialT value.')

parser.add_argument(
        '--top_residues_to_wrap',
        help='top residue numbers separated by spaces to wrap. These along with the bottom residues will be used to determine axis to align against',
        action='store',
        nargs='+',
        default=[]
        )

parser.add_argument(
        '--bottom_residues_to_wrap',
        help='bottom residue numbers separated by spaces to wrap. These along with the top residues will be used to determine axis to align against',
        action='store',
        nargs='+',
        default=[]
        )

parser.add_argument(
        '--all_residues_to_wrap',
        help='Residue numbers separated by spaces to wrap. Requires --top_residues_to_wrap and --bottom_residues_to_wrap. The default is to use DeepTMHMM to determine these values but you can manually set them, for example, to wrap soluble targets.',
        action='store',
        nargs='+',
        default=[]
        )

parser.add_argument('--verbose', default=False, action='store_true', help='Verbose output.')
parser.add_argument('pdbs', nargs=argparse.REMAINDER, help='Input PDBs.')

args = vars(parser.parse_args())
exit = False

rf_diffusion_container = args['rf_diffusion_container']
rf_diffusion = args['rf_diffusion']
rf_partial_diffusions = args['rf_partial_diffusions']
rf_partial_diffusion_partialT = args['rf_diffusion_partialT']

partial_diffusion_task_file_prefix = args['partial_diffusion_task_file_prefix']
barrel_wrap = args['barrel']
termlen = args['barrel_termini_len']
barrel_wrap_params_str = args['wrap_barrel_params']
looplen = args['looplen']
rotation_sample_angle = args['rot_angle']
h_rot_angle = args['h_rot_angle']
rotation_samples = args['rot_n']
manual_n = args['n']
manual_nres = args['nres']
manual_radius = args['radius']
verbose = args['verbose']
all_residues_to_wrap_ = args['all_residues_to_wrap']
top_residues_to_wrap_ = args['top_residues_to_wrap']
bottom_residues_to_wrap_ = args['bottom_residues_to_wrap']

# pre-made wrap to place onto target?
wrap = args['wrap']
wrap_top_resi = args['wrap_top_resi']
wrap_bottom_resi = args['wrap_bottom_resi']
flip_wrap = args['flip_wrap']
flipped_str = ""
if flip_wrap:
  flipped_str = 'flipped_'
wraptopresi = wrap_top_resi.split(',')
wrapbottomresi = wrap_bottom_resi.split(',')
if len(wrap) > 0 and not os.path.exists(wrap):
  exit = True
wrap_name = wrap.split('/')[-1].split('.pdb')[0]

# manually set residues to wrap?
all_residues_to_wrap = []
top_residues_to_wrap = []
bottom_residues_to_wrap = []
for v in all_residues_to_wrap_:
  vals = v.split('-')
  if len(vals) == 2:
    for i in range(int(vals[0]), int(vals[1])+1):
      all_residues_to_wrap.append(i)
  elif len(vals) == 1:
    all_residues_to_wrap.append(int(vals[0]))
  else:
    print(f'ERROR parsing all_residues_to_wrap!')
    sys.exit(1)
for v in top_residues_to_wrap_:
  vals = v.split('-')
  if len(vals) == 2:
    for i in range(int(vals[0]), int(vals[1])+1):
      top_residues_to_wrap.append(i)
  elif len(vals) == 1:
    top_residues_to_wrap.append(int(vals[0]))
  else:
    print(f'ERROR parsing top_residues_to_wrap!')
    sys.exit(1)
for v in bottom_residues_to_wrap_:
  vals = v.split('-')
  if len(vals) == 2:
    for i in range(int(vals[0]), int(vals[1])+1):
      bottom_residues_to_wrap.append(i)
  elif len(vals) == 1:
    bottom_residues_to_wrap.append(int(vals[0]))
  else:
    print(f'ERROR parsing bottom_residues_to_wrap!')
    sys.exit(1)
    
# input targets to wrap
pdbs = []
if args['pdbs']:
  for pdb in args['pdbs']:
    if pdb.endswith('.pdb'):
      pdbs.append(pdb)
    else:
      with open(pdb) as f:
        for l in f:
          pdb = l.strip().split()[0]
          if pdb.endswith('.pdb'):
            pdbs.append(pdb)
else:
  exit = True
if len(pdbs) == 0: exit = True

if exit:
  parser.print_help(sys.stderr)
  sys.exit(1)


from pyrosetta import *
from pyrosetta.rosetta import *
from pyrosetta.rosetta.std import map_core_id_AtomID_core_id_AtomID
from pyrosetta.rosetta.core.id import AtomID
from pyrosetta.rosetta.protocols.loops.loop_closure.ccd import *
from pyrosetta.rosetta.std import (
    vector_numeric_xyzVector_double_t,
)
from pyrosetta.rosetta.numeric import xyzVector_double_t

if verbose or debug:
  init( " -max_kic_build_attempts 100 -out:level 10000 " )
else:
  init( " -max_kic_build_attempts 100 -mute all " )


# PARAM n (number of helices function of TM radius)
# 110/16 circumference divided by 16 = 6.875 (~7) based on 8efo example w/ 16 helices <- LJ's original count
def number_of_helices(radius):
  if manual_n > 0:
    return manual_n
  return int(round(np.pi*2*(radius+rbuffer)/7))

# PARAM radius (if manual_n and no manual_radius)
def radius_from_n(n):
  if manual_radius > 0:
    return manual_radius
  return (n*7/(np.pi*2))-rbuffer

# PARAM nres (length of helix as a function of TM height)
# 3.6 res/turn 5.4 angstroms rise/turn;
def helix_length(tmheight):
  if manual_nres > 0:
    return manual_nres
  return int(round(((tmheight+hbuffer)/5.4)*3.6))

def generate_helix(n):
  aa = ''
  for i in range(n):
    aa += 'A'
  hp = pose_from_sequence(aa, "fa_standard")
  for i in range(1,n+1):
    hp.set_phi(i, -57)
    hp.set_psi(i, -47)
  return hp

def append_chain_to_pose(p1a,p2a,chain=1,new_chain=True):
  jumpadded = False
  for res in pyrosetta.rosetta.core.pose.get_chain_residues(p2a,chain):
    if not jumpadded:
      p1a.append_residue_by_jump(res, 1, "", "", new_chain)
      jumpadded = True
    else:
      p1a.append_residue_by_bond(res)
  return p1a

def get_transmembrane_residues(pose, input_pdb_name):
  # https://dtu.biolib.com/DeepTMHMM
  # Jeppe Hallgren, Konstantinos D. Tsirigos, Mads D. Pedersen, José Juan Almagro Armenteros, Paolo Marcatili,Henrik Nielsen,
  # Anders Krogh and Ole Winther (2022). DeepTMHMM predicts alpha and beta transmembrane proteins using deep neural networks.
  # https://doi.org/10.1101/2022.04.08.487609
  biolib.utils.STREAM_STDOUT = True # Stream progress from app in real time
  deeptmhmm = biolib.load('DTU/DeepTMHMM')
  tmpred = f'{input_pdb_name}_DeepTMHMM/predicted_topologies.3line'
  if not os.path.exists(tmpred):
    seq = pose.sequence()
    with open(f'{input_pdb_name}_tmp.fasta', 'w') as f:
      f.write(f'>{input_pdb}'+"\n")
      f.write(seq+"\n")
    deeptmhmm_job = deeptmhmm.cli(args=f'--fasta {input_pdb_name}_tmp.fasta') # Blocks until done
    deeptmhmm_job.save_files(f'{input_pdb_name}_DeepTMHMM') # Saves all results to `result` dir
    os.remove(f'{input_pdb_name}_tmp.fasta')
  trans = []
  outer = []
  inner = []
  outer_trans = []
  inner_trans = []
  with open(tmpred) as f:
    lcnt = 1
    for l in f:
      if lcnt == 3:
        llen = len(l.strip())
        for i in range(0,llen):
          if l[i] == 'M' or l[i] == 'B' or l[i] == '1' or l[i] == '2':
            trans.append(i+1)
            tlen = len(trans)
            if i==0 or l[i-1] == 'O': # input Nterm is outer
              for add in range(0,blen):
                outer.append(tlen+add)
                outer_trans.append(i+1+add)
            elif l[i+1] == 'O':
              for sub in range(0,blen):
                outer.append(tlen-sub)
                outer_trans.append(i+1-sub)
            elif l[i-1] == 'I' or l[i-1] == 'P' or l[i-1] == 'S' or l[i-1] == 'X':
              for add in range(0,blen):
                inner.append(tlen+add)
                inner_trans.append(i+1+add)
            elif i == llen-1 or (l[i+1] == 'I' or l[i+1] == 'P' or l[i+1] == 'S' or l[i+1] == 'X'): # input Cterm is inner
              for sub in range(0,blen):
                inner.append(tlen-sub)
                inner_trans.append(i+1-sub)
        break
      lcnt += 1
  if verbose:
    print(f'get_transmembrane_residues from: {tmpred}..')
    print('+'.join(map(str,trans)))
    print('outer trans: '+'+'.join(map(str,outer_trans)))
    print('inner trans: '+'+'.join(map(str,inner_trans)))
    print('outer: '+'+'.join(map(str,outer)))
    print('inner: '+'+'.join(map(str,inner)))
    print()
  return trans,outer_trans,inner_trans,outer,inner

def center_of_mass(xyzVec):
  cofm = xyzVector_double_t( 0.0, 0.0, 0.0 )
  cnt = 0
  for xyz in xyzVec:
    cofm += xyz
    cnt += 1
  cofm /= cnt
  return cofm 

def set_loop_omega(p,pos,isN):
  thisp = Pose()
  thisp.assign(p)
  osdist1 = 0
  if isN:
    odist1 = (thisp.residue(pos-1).atom('O').xyz() - p.residue(pos).atom('H').xyz()).norm()
  else:
    odist1 = (thisp.residue(pos).atom('O').xyz() - p.residue(pos+1).atom('H').xyz()).norm()
  thisp.set_omega(pos, thisp.omega(pos)+180.)
  osdist2 = 0
  if isN:
    odist2 = (thisp.residue(pos-1).atom('O').xyz() - p.residue(pos).atom('H').xyz()).norm()
  else:
    odist2 = (thisp.residue(pos).atom('O').xyz() - p.residue(pos+1).atom('H').xyz()).norm()
  if odist1 > odist2:
    return p
  else:
    return thisp

def add_loops(pose,Nposs,Cposs):
  # add loop
  p = Pose()
  p.assign(pose)
  pstartlen = len(p.sequence())
  loops = protocols.loops.Loops()
  added = 0
  skip = False
  for i,Npos in enumerate(Nposs):
    Npos = Npos+added
    Cpos = Cposs[i] + added
    if verbose: print(f'Adding {looplen} residue loop connecting {Npos} to {Cpos}')
    inserted = 0
    cutpoint = 0
    for j in range(looplen):
      insertpos = Cpos+inserted
      res_type = core.chemical.ChemicalManager.get_instance().residue_type_set( 'fa_standard' ).get_representative_type_name1('A')
      residue = core.conformation.ResidueFactory.create_residue(res_type)
      if j < looplen/2:
        cutpoint = insertpos+1
        p.append_polymer_residue_after_seqpos(residue, insertpos, True)
        inserted += 1
      else:
        p.prepend_polymer_residue_before_seqpos(residue, insertpos+1, True)

    lstart = Cpos
    lstop = Cpos+looplen+1
    loop = protocols.loops.Loop(lstart, lstop, cutpoint, 0.0, True)
    loop.set_extended( True )
    loops.add_loop(loop)
    protocols.loops.set_single_loop_fold_tree(p, loop)
    for i in range(lstop,cutpoint,-1):
      p = set_loop_omega(p,i,False)
    for i in range(lstart,cutpoint+1):
      p = set_loop_omega(p,i,True)
    added = len(p.sequence())-pstartlen
    start_stop_dist = (p.residue(Cpos-1).atom(2).xyz()-p.residue(Cpos+looplen+2).atom(2).xyz()).norm()
    skip = False
    if verbose: print(f'endpoints {Cpos} {Cpos+looplen+1} {start_stop_dist} / looplen {looplen} {start_stop_dist/looplen}')
    if start_stop_dist/looplen > max_dist_per_loop_res:
      print(f'endpoints {start_stop_dist} / looplen {looplen} {start_stop_dist/looplen} > {max_dist_per_loop_res} so cannot close loops. Increase looplen.')
      skip = True
    if verbose: print(f'Added loop: {Cpos} {Cpos+looplen+1} {cutpoint}')
  return p, loops, skip

def is_closed(pose, loops):
  maxd = 7. # allow some play since partial diffusion may fix the breaks
  highd = 0.
  for i in range(1, loops.size()+1):
    d = (pose.residue(loops[i].cut()).atom('CA').xyz()-pose.residue(loops[i].cut()+1).atom('CA').xyz()).norm()
    if verbose: print(f'Loop {i} cut distance: {d}')
    if d > highd: highd = d
  if highd > maxd: return False
  return True

def closeloops(nss, ss_nres, pose):
  Nposs = []
  Cposs = []
  for i in range(1,nss):
    Cposs.append(i*ss_nres)
    Nposs.append(i*ss_nres+1)
  pose, loops, skip = add_loops(pose,Nposs,Cposs)
  if not skip:
    if debug: pose.dump_pdb('pre_close.pdb')
    if verbose: print(f'Attempting to close length {looplen} loops...')
    # Note LoopModeler may not be successful at closing loops for various reasons
    # This should be replaced with a more robust quick and rough loop modeler in the future.
    # The loops do not have to be accurate or physically realistic since RF partial diffusion
    # will refine the WRAP structure.
    lm = protocols.loop_modeler.LoopModeler()
    lm.set_loops(loops)
    lm.disable_centroid_stage()
    lm.disable_fullatom_stage()
    lm.apply(pose)
    if not is_closed(pose, loops):
      if verbose: print(f'Loops could not close so skipping..')
      skip = True
  return pose, skip

def get_top_bottom_coords(ss_p,nss,ss_nres,offset=0,thislooplen=0,thistermlen=0):
  # get ss wrap top and bottom coords
  ctop = []
  cbottom = []
  ctopcoords = vector_numeric_xyzVector_double_t()
  cbottomcoords = vector_numeric_xyzVector_double_t()
  ccoords = vector_numeric_xyzVector_double_t()
  for i in range(1, nss+1):
    clen = (i*ss_nres)+thistermlen+(i-1)*thislooplen
    for j in range(clen-ss_nres+1,clen+1):
      ccoords.append(ss_p.residue(j).atom(2).xyz())
    if not i%2:
      for add in range(1,blen+1):
        ctop.append(clen-ss_nres+add+offset)
      for sub in range(1,blen+1):
        cbottom.append(clen-sub+1+offset)
    else:
      for add in range(1,blen+1):
        cbottom.append(clen-ss_nres+add+offset)
      for sub in range(1,blen+1):
        ctop.append(clen-sub+1+offset)
  for i in ctop:
    ctopcoords.append(ss_p.residue(i).atom(2).xyz())
  for i in cbottom:
    cbottomcoords.append(ss_p.residue(i).atom(2).xyz())
  return ctop,cbottom,ctopcoords,cbottomcoords,ccoords


final_wraps = []
# This outputs the final parametric WRAPS sampling +/- rotations
def save_rotations(p, input_p, nss, ss_nres, output_prefix, termlen, looplen, rotation_samples, move_angle, spinmover_forward=None, spinmover_reverse=None, fix_chain_order=False):
  pose = Pose()
  pose.assign(p)
  if spinmover_forward == None:
    ctop,cbottom,ctopcoords,cbottomcoords,ccoords = get_top_bottom_coords(pose,nss,ss_nres,0,looplen,termlen) 
    # rotate wrap
    spinmover_forward = protocols.rigid.RigidBodyDeterministicSpinMover(pose.num_jump(),center_of_mass(ctopcoords)-center_of_mass(cbottomcoords),center_of_mass(ccoords),move_angle)
    spinmover_reverse = protocols.rigid.RigidBodyDeterministicSpinMover(pose.num_jump(),center_of_mass(ctopcoords)-center_of_mass(cbottomcoords),center_of_mass(ccoords),-1*move_angle)
  offset_angle = 0
  # sample angles forward and reverse of original (kept at center)
  for j in range(int(rotation_samples/2)):
    spinmover_reverse.apply(pose)
    offset_angle -= move_angle
  for i in range(rotation_samples):
    final_wrap_pdb = f'{output_prefix}_rot{offset_angle}.pdb'
    print(f'Creating WRAP {final_wrap_pdb}')
    if fix_chain_order:
      outputp = Pose()
      outputp.assign(pose.split_by_chain(2))
      append_chain_to_pose(outputp,input_p,1,True)
      outputp.dump_pdb(final_wrap_pdb)
    else:
      pose.dump_pdb(final_wrap_pdb)
    final_wraps.append(final_wrap_pdb)
    spinmover_forward.apply(pose)
    offset_angle += move_angle

def chain1_len(ca_atoms):
  chainAlen = 0
  prevchain = ''
  for ca in ca_atoms:
    if len(prevchain) and ca[0] != prevchain:
      break
    chainAlen += 1
    prevchain = ca[0]
  return chainAlen

def read_pdb_atom(l):
  chain = l[20:22].strip()
  atype = l[11:17].strip()
  name3 = l[17:20].strip()
  resnum = int(l[22:26].strip())
  x = float(l[30:38])
  y = float(l[38:46])
  z = float(l[46:54])
  return chain, atype, name3, resnum, x, y, z

def chain2_contigs(ca_atoms):
  chain1len = chain1_len(ca_atoms)
  start =  ca_atoms[chain1len][0]+str(ca_atoms[chain1len][3])
  target_chain_breaks = []
  for i in range(chain1len,len(ca_atoms)-1):
    d = np.linalg.norm( np.array(ca_atoms[i][4:7]) - np.array(ca_atoms[i+1][4:7]) )
    if d > 4.2: # chainbreak
      target_chain_breaks.append(f'{start}-{ca_atoms[i][3]}')
      start = ca_atoms[i+1][0]+str(ca_atoms[i+1][3])
  target_chain_breaks.append(f'{start}-{ca_atoms[-1][3]}')
  return ','.join(target_chain_breaks)

def get_helices_top_bottom_coords_from_DSSP(p_w):
  ctopcoords = vector_numeric_xyzVector_double_t()
  cbottomcoords = vector_numeric_xyzVector_double_t()
  DSSP = pyrosetta.rosetta.core.scoring.dssp.Dssp(p_w)
  ssstr = DSSP.get_dssp_secstruct()
  aastr = p_w.sequence()
  prevaa = ''
  ends = []
  if verbose:
    print(f'Getting top and bottom coords from DSSP')
    print(ssstr)
  for i in range(len(ssstr)):
    aa = ssstr[i]
    if aa == 'H':
      if prevaa != aa:
        ends.append(i+1)
      elif ssstr[i+1] != aa:
        ends.append(i+1)
      elif i == len(ssstr)+1:
        ends.append(i+1)
    prevaa = aa
  top = []
  bottom = []
  added = 0
  for i,resi in enumerate(ends):
    if i == 0 and len(top) == 0:
      top.append(resi)
      ctopcoords.append(p_w.residue(resi).atom(2).xyz())
    elif added < 2:
      bottom.append(resi)
      cbottomcoords.append(p_w.residue(resi).atom(2).xyz())
      added += 1
    elif added < 4:
      top.append(resi)
      ctopcoords.append(p_w.residue(resi).atom(2).xyz())
      added += 1
    if added >= 4:
      added = 0
  topstr = '+'.join(map(str,top))
  bottomstr = '+'.join(map(str,bottom))
  if verbose:
    print(f'top {topstr}')
    print(f'bottom {bottomstr}')
  return ctopcoords, cbottomcoords


## MAIN

p_w = Pose() # wrap pose
ctopcoords = vector_numeric_xyzVector_double_t()
cbottomcoords = vector_numeric_xyzVector_double_t()
if len(wrap) > 0:
  if verbose: print(f'Trying to place input wrap {wrap}')
  p_w = pose_from_file(wrap)
  # get top and bottom coords of wrap
  if len(wraptopresi) >= 3 and len(wrapbottomresi) >= 3:
    # if manually provided
    # get alignment coordinates from input top and bottom resi's
    for i,resi in enumerate(wraptopresi):
      ctopcoords.append(p_w.residue(resi).atom(2).xyz())
    for i,resi in enumerate(wrapbottomresi):
      cbottomcoords.append(p_w.residue(resi).atom(2).xyz())
  else:
    # did user provide barrel params for input wrap?
    barrel_wrap_params = list(map(int,barrel_wrap_params_str.split(',')))
    if len(barrel_wrap_params) > 0:
      if len(barrel_wrap_params) != 4:
        print("--wrap_barrel_params requires 4 comma separated values for strand count, strand length, connecting loop length, and termini length.")
        sys.exit(1)
      ctop,cbottom,ctopcoords,cbottomcoords,ccoords = get_top_bottom_coords(p_w,barrel_wrap_params[0],barrel_wrap_params[1],0,barrel_wrap_params[2],barrel_wrap_params[3])  
    else:
      # For helical symmetric wrap inputs only
      # try to automatically determine the top and bottom based on DSSP
      # get alignment coordinates from secondary structure assuming input is up-down H
      ctopcoords,cbottomcoords = get_helices_top_bottom_coords_from_DSSP(p_w)

for pdb in pdbs:
  input_pdb = pdb
  input_pdb_name = input_pdb.split('.pdb')[0].split('/')[-1]
  input_p = pose_from_file(input_pdb)
  input_p_len = len(input_p.sequence())
  partial_diffusion_task_file_suffix = "" 
  outprefix = "" # output prefix for this pdb target input

  trans = [] # residues to wrap (transmembrane residues for example)
  top = [] # top residues of trans to calculate radius 
  bottom = [] # bottom residues of trans to calculate radius
  top_trans = [] # top residues of trans to calculate axis and points for superposition, translation, etc.
  bottom_trans = [] # bottom residues of trans to calculate axis and points for superposition, translation, etc.
  if len(all_residues_to_wrap) > 0 and len(top_residues_to_wrap) > 0 and len(bottom_residues_to_wrap) > 0:
    trans = all_residues_to_wrap
    for i,r in enumerate(trans):
      if r in top_residues_to_wrap:
        top_trans.append(i+1)
      if r in bottom_residues_to_wrap:
        bottom_trans.append(i+1)
    top = top_residues_to_wrap
    bottom = bottom_residues_to_wrap
  else:
    # get transmembrane part of input based on tmhmm prediction
    trans,top,bottom,top_trans,bottom_trans = get_transmembrane_residues(input_p, input_pdb_name)

  # estimate radius based on top and bottom coords of transmembrane segments
  icoords = vector_numeric_xyzVector_double_t()
  for i in top:
    icoords.append(input_p.residue(i).atom(2).xyz())
  icomt = center_of_mass(icoords)
  idists = []
  for i in icoords:
    idists.append((i - icomt).norm())
  icoords = vector_numeric_xyzVector_double_t()
  for i in bottom:
    icoords.append(input_p.residue(i).atom(2).xyz())
  icomb = center_of_mass(icoords)
  for i in icoords:
    idists.append((i - icomb).norm())
  radius = np.mean(idists)
  if manual_radius > 0:
    radius = manual_radius
  elif manual_n > 0:
    radius = radius_from_n(manual_n) 
  tmheight = (icomt - icomb).norm()
  
  if verbose:
    print(f'Radius: {radius} TM height: {tmheight}')
    print(f'Looplen: {looplen}')
 
  # make transmembrane pose
  prev = 0
  p_trans = Pose()
  for i in range(1, input_p_len+1):
    res = input_p.residue(i)
    if i in trans:
      if i > 1 and i-1 != prev:
        p_trans.append_residue_by_jump(res, 1, "", "", False)
      else:
        p_trans.append_residue_by_bond(res)
    prev = i

  # get top and bottom coords of transmembrane pose
  topcoords = vector_numeric_xyzVector_double_t()
  bottomcoords = vector_numeric_xyzVector_double_t()
  for i in top_trans:
    topcoords.append(p_trans.residue(i).atom(2).xyz())
  for i in bottom_trans:
    bottomcoords.append(p_trans.residue(i).atom(2).xyz())

  ######################################
  ## BETA BARREL WRAP?

  if barrel_wrap and len(wrap) == 0:
    # Create parametric cylinders (Naveed H. et al. JACS, 2012)
    # Sample range of parameters for optimal wrap height and radii to wrap target
    # 27-32 angstroms lipid bilayer height
    barrel_params = []
    for nss in range(4,1000):  # sample a bunch of strands
      for shear in range(nss,(nss*2)+1): # sample a bunch of respective shears
        if shear%2 == 0:
          # cylinder radius based on barrel parameters
          r = math.sqrt((shear*intra_strand_dist)**2+(nss*inter_strand_dist)**2)/(2*nss*math.sin(math.pi/nss))

          # Sample radii optimal for packing with the target (distances to fit side chain interactions well between target and wrap) 
          if r > radius+beta_radius_buffer_min and r < radius+beta_radius_buffer_max:
            for ss_nres in range(6,30):   # sample a bunch of strand lengths
              # coil angle
              theta=asin(shear*intra_strand_dist/(2*math.pi*r))
              def disNextRes(x):
                return sqrt(r**2*(2-2*cos(x[0]))+(r*x[0]/tan(theta))**2)-intra_strand_dist
              def disNextStrand(x):
                return sqrt(r**2*(2-2*cos(x[0]+2*math.pi/nss))+(r*x[0]/tan(theta))**2)-inter_strand_dist
              delta_t1 = fsolve(disNextRes,0)[0]
              delta_t2 = fsolve(disNextStrand,0)[0]
              def dis(x,y):
                s=0
                for i in range(len(x)): s += (x[i]-y[i])**2
                return sqrt(s)
              for ns in range(1,nss+1):
                phi = (ns-1)*2*math.pi/nss
                dt2 = (delta_t2)*(ns-1)
              # get cylinder height
              heighta = []
              n0 = 0 
              for j in range(800):
                dt1 = (delta_t1)*j
                dt = dt1+dt2
                x=r*cos(dt+phi)
                y=r*sin(dt+phi)
                z=r*dt/tan(theta)
                if z<0:continue
                n0 += 1
                if n0>ss_nres: break # +4: break #jump out of the loop
                sign = 1 if j%2 == 0 else -1
                heighta.append(z)
              h = heighta[-1]-heighta[0]
              # save cylinder wrap params
              if h < tmheight+beta_tmheight_buffer_max and h > tmheight+beta_tmheight_buffer_min: 
                barrel_params.append( [ nss, shear, ss_nres, h, r ] )              
    # generate cylinders
    for barrel_param in barrel_params:
      nss = barrel_param[0]
      shear = barrel_param[1]
      ss_nres = barrel_param[2]
      barrel_h = barrel_param[3]
      barrel_r = barrel_param[4]
      outpdbA = f'temp_barrelCylinder_n{nss}_S{shear}_nres{ss_nres}.pdb'
      r=sqrt((nss*inter_strand_dist)**2+(shear*intra_strand_dist)**2)/(2*pi)
      theta=asin(shear*intra_strand_dist/(2*math.pi*r))
      def disNextRes(x):
        return sqrt(r**2*(2-2*cos(x[0]))+(r*x[0]/tan(theta))**2)-intra_strand_dist
      def disNextStrand(x):
        return sqrt(r**2*(2-2*cos(x[0]+2*math.pi/nss))+(r*x[0]/tan(theta))**2)-inter_strand_dist
      delta_t1 = fsolve(disNextRes,0)[0]
      delta_t2 = fsolve(disNextStrand,0)[0]
      def dis(x,y):
        s=0
        for i in range(len(x)): s += (x[i]-y[i])**2
        return sqrt(s)
      coor = []
      for ns in range(1,nss+1):
        phi = (ns-1)*2*pi/nss
        dt2 = (delta_t2)*(ns-1)
        coor_strand = []
        n0 = 0 #number of residues at the strand
        for j in range(800):
          dt1 = (delta_t1)*j
          dt = dt1+dt2
          x=r*cos(dt+phi)
          y=r*sin(dt+phi)
          z=r*dt/tan(theta)
          if z<0:continue
          n0 += 1
          if n0>ss_nres: break 
          sign = 1 if j%2 == 0 else -1
          coor_strand.append([x,y,z,sign])
        coor.append(coor_strand)
      chain_ids = 'ABCDEFGHIJKLMNOPQSTUVWXYZ'
      chaini = 0
      foA = open(outpdbA,'w')
      resi = 1
      atomi = 2
      for i in range(len(coor)):
        coor_strand = coor[i]
        if i%2 == 0: coor_strand.reverse()
        for j in range(len(coor_strand)):
          (x,y,z,sign) = coor_strand[j]
          foA.write("ATOM  %5d  CA  VAL %1s%4d    %8.3f%8.3f%8.3f  1.00  0.00           %1d\n" % (atomi,chain_ids[chaini],resi,x,y,z,sign))
          resi += 1
          atomi += 4
        chaini += 1
      foA.close()
      partial_diffusion_task_file_suffix = f'{input_pdb_name}_WRAP_{flipped_str}barrel'
      outprefix = f'{input_pdb_name}_WRAP_{flipped_str}barrel_n{nss}_S{shear}_nres{ss_nres}_looplen{looplen}_termlen{termlen}_r{barrel_r:.2f}_cheight{barrel_h:.0f}'

      # Create backbones from CA only cylinders using BBQ
      # Gront, D. et. al. (2007), Backbone building from quadrilaterals: A fast and accurate 
      # algorithm for protein backbone reconstruction from alpha carbon coordinates. 
      # J. Comput. Chem., 28: 1593-1597. https://doi.org/10.1002/jcc.20624
      os.system(f"java apps.BBQ -ip={outpdbA} > /dev/null 2>&1")
      # cylinder pose
      cylinder_p_multichain = pose_from_file(outpdbA.split('.pdb')[0]+'-bb.pdb')
      if debug: cylinder_p_multichain.dump_pdb('cylinder_p_multichain.pdb')
      cylinder_p = Pose()
      for i in range(1,(nss*ss_nres)+1):
        core.conformation.remove_upper_terminus_type_from_conformation_residue(cylinder_p_multichain.conformation(), i)
        core.conformation.remove_lower_terminus_type_from_conformation_residue(cylinder_p_multichain.conformation(), i)
        cylinder_p.append_residue_by_bond(cylinder_p_multichain.residue(i))
      cylinder_p_len = len(cylinder_p.sequence())
      if debug: cylinder_p.dump_pdb('cylinder_p_trimmed.pdb')
      
      # cleanup temp pdbs
      if not debug:
        for rmf in [ outpdbA, outpdbA+'.ss2', outpdbA.split('.pdb')[0]+'_rebuilt.pdb', outpdbA.split('.pdb')[0]+'-bb.pdb' ]:
          if os.path.exists(rmf): os.remove(rmf)
 
      # get cylinder top and bottom coords
      ctop,cbottom,ctopcoords,cbottomcoords,ccoords = get_top_bottom_coords(cylinder_p,nss,ss_nres)   
 
      # align cylinder to target based on top, bottom, and center of mass coords
      p1 = Pose()
      p1.assign(cylinder_p)
      p2 = Pose()
      p2.assign(p_trans)
      rsd_set = p1.residue_type_set_for_pose( core.chemical.FULL_ATOM_t )
      tmp_res = core.conformation.ResidueFactory.create_residue( rsd_set.name_map( 'GLY' ) )
      p1.append_residue_by_jump(tmp_res, 1, "", "", False)
      p1.residue(len(p1.sequence())).atom(2).xyz(center_of_mass(ctopcoords))
      p1.append_residue_by_jump(tmp_res, 1, "", "", False)
      p1.residue(len(p1.sequence())).atom(2).xyz(center_of_mass(cbottomcoords))
      p1.append_residue_by_jump(tmp_res, 1, "", "", False)
      p1.residue(len(p1.sequence())).atom(2).xyz(core.pose.get_center_of_mass(p1))
      rsd_set = p2.residue_type_set_for_pose( core.chemical.FULL_ATOM_t )
      tmp_res = core.conformation.ResidueFactory.create_residue( rsd_set.name_map( 'GLY' ) )
      p2.append_residue_by_jump(tmp_res, 1, "", "", False)
      p2.residue(len(p2.sequence())).atom(2).xyz(center_of_mass(topcoords))
      p2.append_residue_by_jump(tmp_res, 1, "", "", False)
      p2.residue(len(p2.sequence())).atom(2).xyz(center_of_mass(bottomcoords))
      p2.append_residue_by_jump(tmp_res, 1, "", "", False)
      p2.residue(len(p2.sequence())).atom(2).xyz(core.pose.get_center_of_mass(p2))
      p2len = len(p2.sequence())
      p1len = len(p1.sequence())

      ca_map = map_core_id_AtomID_core_id_AtomID()
      ca_map[AtomID(p1.residue(p1len).atom_index("CA"), p1len)] = AtomID(p2.residue(p2len).atom_index("CA"), p2len)
      ca_map[AtomID(p1.residue(p1len-1).atom_index("CA"), p1len-1)] = AtomID(p2.residue(p2len-1).atom_index("CA"), p2len-1)
      ca_map[AtomID(p1.residue(p1len-2).atom_index("CA"), p1len-2)] = AtomID(p2.residue(p2len-2).atom_index("CA"), p2len-2)
      rmsd = pyrosetta.rosetta.core.scoring.superimpose_pose(p1,p2, ca_map, 0.00000001, False, False)
      if verbose: print(f'Cylinder to target axis superposition rmsd: {rmsd}')
      if rmsd == 0.0:
        if verbose: print(f'Skipping due to alignment error.')
        continue
    
      p1_flipped = Pose()
      p1_flipped.assign(p1)
      p2_flipped = Pose()
      p2_flipped.assign(p2)
    
      ca_map = map_core_id_AtomID_core_id_AtomID()
      ca_map[AtomID(p1_flipped.residue(p1len).atom_index("CA"), p1len)] = AtomID(p2_flipped.residue(p2len).atom_index("CA"), p2len)
      ca_map[AtomID(p1_flipped.residue(p1len-1).atom_index("CA"), p1len-1)] = AtomID(p2_flipped.residue(p2len-2).atom_index("CA"), p2len-2)
      ca_map[AtomID(p1_flipped.residue(p1len-2).atom_index("CA"), p1len-2)] = AtomID(p2_flipped.residue(p2len-1).atom_index("CA"), p2len-1)
      rmsd = pyrosetta.rosetta.core.scoring.superimpose_pose(p1_flipped,p2_flipped, ca_map, 0.00000001, False, False)
      if verbose: print(f'Cylinder to target flipped axis superposition rmsd: {rmsd}')
      if rmsd == 0.0:
        if verbose: print(f'Skipping due to alignment error.')
        continue

      for i in range(0,3):
        p2.delete_residue_slow(len(p2.sequence()))
        p1.delete_residue_slow(len(p1.sequence()))
        p2_flipped.delete_residue_slow(len(p2_flipped.sequence()))
        p1_flipped.delete_residue_slow(len(p1_flipped.sequence()))

      cylinder_p.assign(p1)
      cylinder_p_flipped = Pose()
      cylinder_p_flipped.assign(p1_flipped)
    
      # append target as a new chain
      core.pose.append_pose_to_pose(cylinder_p, input_p)
      core.pose.append_pose_to_pose(cylinder_p_flipped, input_p)
      
      # At this point the target should be placed in the cylinder and flipped cylinder
      #
      # N- fusion   =>  wrap-C N-target
      # C- fusion   =>  target-C N-wrap
      #

      # Need to determine what wrap orientaion(s) have the N and C termini on the same side
      
      move_angle = 5 # increments to check for the closest N to C termini distances

      # For C- Fusion
      # rotate cylinder until transmembrane C-term is close to cylinder N-term
      mindist_tC_to_wN_p = Pose()
      
      # The cylinder has to be the last chain for the spin mover to rotate it
      cylinder_p_ = Pose()
      cylinder_p_.assign(input_p)
      cylinder_p_ = append_chain_to_pose(cylinder_p_, cylinder_p.split_by_chain(1))

      ctop,cbottom,ctopcoords,cbottomcoords,ccoords = get_top_bottom_coords(cylinder_p_,nss,ss_nres,input_p_len)   
      spinmover = protocols.rigid.RigidBodyDeterministicSpinMover(cylinder_p_.num_jump(),center_of_mass(ctopcoords)-center_of_mass(cbottomcoords),core.pose.get_center_of_mass(cylinder_p_.split_by_chain(2)),move_angle)
      mindist = 9999999.
      min_i = 0
      check_p = Pose()
      check_p.assign(cylinder_p_)
      for i in range(1,int(360/move_angle)+1):
        spinmover.apply(check_p)
        dist = (check_p.residue(input_p_len).atom(2).xyz() - check_p.residue(input_p_len+1).atom(2).xyz()).norm()
        if dist < mindist:
          mindist = dist
          min_i = i
      for i in range(1,min_i+1):
        spinmover.apply(cylinder_p_)
      mindist_tC_to_wN = mindist
      
      # make sure the rotated cylinder is back to being the first chain
      cylinder_p = append_chain_to_pose(cylinder_p_.split_by_chain(2), input_p)

      # get the same transmembrane C-term distance to the cylinder N-term to the flipped complex

      # make the flipped cylinder the last chain for the spin mover
      cylinder_p_flipped_ = Pose()
      cylinder_p_flipped_.assign(input_p)
      cylinder_p_flipped_ = append_chain_to_pose(cylinder_p_flipped_, cylinder_p_flipped.split_by_chain(1))

      ctop,cbottom,ctopcoords,cbottomcoords,ccoords = get_top_bottom_coords(cylinder_p_flipped_,nss,ss_nres,input_p_len)
      spinmoverf = protocols.rigid.RigidBodyDeterministicSpinMover(cylinder_p_flipped_.num_jump(),center_of_mass(cbottomcoords)-center_of_mass(ctopcoords),core.pose.get_center_of_mass(cylinder_p_flipped_.split_by_chain(2)),move_angle)
      mindist = 9999999.
      min_i = 0
      check_p = Pose()
      check_p.assign(cylinder_p_flipped_)
      for i in range(1,int(360/move_angle)+1):
        spinmoverf.apply(check_p)
        dist = (check_p.residue(input_p_len).atom(2).xyz() - check_p.residue(input_p_len+1).atom(2).xyz()).norm()
        if dist < mindist:
          mindist = dist
          min_i = i
      for i in range(1,min_i+1):
        spinmoverf.apply(cylinder_p_flipped_)

      if (mindist_tC_to_wN < mindist and not flip_wrap) or (mindist_tC_to_wN > mindist and flip_wrap):
        mindist_tC_to_wN_p.assign(cylinder_p)
      else:
        # make sure the rotated flipped cylinder is back to being the first chain
        cylinder_p_flipped = append_chain_to_pose(cylinder_p_flipped_.split_by_chain(2), input_p)
        mindist_tC_to_wN_p.assign(cylinder_p_flipped)

      # For N- Fusion
      # rotate cylinder until transmembrane N-term is close to cylinder C-term
      mindist_tN_to_wC_p = Pose()

      mindist = 9999999.
      min_i = 0
      check_p = Pose()
      check_p.assign(cylinder_p_)
      for i in range(1,int(360/move_angle)+1):
        spinmover.apply(check_p)
        dist = (check_p.residue(1).atom(2).xyz() - check_p.residue(len(check_p.sequence())).atom(2).xyz()).norm()
        if dist < mindist:
          mindist = dist
          min_i = i
      for i in range(1,min_i+1):
        spinmover.apply(cylinder_p_)
      mindist_tN_to_wC = mindist

      # make sure the rotated cylinder is back to being the first chain
      cylinder_p = append_chain_to_pose(cylinder_p_.split_by_chain(2), input_p)

      # get the same transmembrane N-term distance to the cylinder C-term to the flipped complex

      mindist = 9999999.
      min_i = 0
      check_p = Pose()
      check_p.assign(cylinder_p_flipped_)
      for i in range(1,int(360/move_angle)+1):
        spinmoverf.apply(check_p)
        dist = (check_p.residue(1).atom(2).xyz() - check_p.residue(len(check_p.sequence())).atom(2).xyz()).norm()
        if dist < mindist:
          mindist = dist
          min_i = i
      for i in range(1,min_i+1):
        spinmoverf.apply(cylinder_p_flipped_)

      if (mindist_tN_to_wC < mindist and not flip_wrap) or (mindist_tN_to_wC > mindist and flip_wrap):
        mindist_tN_to_wC_p.assign(cylinder_p)
      else:
        # make sure the rotated flipped cylinder is back to being the first chain
        cylinder_p_flipped = append_chain_to_pose(cylinder_p_flipped_.split_by_chain(2), input_p)
        mindist_tN_to_wC_p.assign(cylinder_p_flipped)

      p_tC_to_wN = Pose()
      p_tC_to_wN.assign(mindist_tC_to_wN_p)
      p_tC_to_wN, skip = closeloops(nss, ss_nres, p_tC_to_wN.split_by_chain(1))
      if skip: continue
      # add full target back
      p_tC_to_wN = append_chain_to_pose(p_tC_to_wN, input_p)
 
      p_tN_to_wC = Pose()
      p_tN_to_wC.assign(mindist_tN_to_wC_p)
      p_tN_to_wC, skip = closeloops(nss, ss_nres, p_tN_to_wC.split_by_chain(1))
      if skip: continue
      # add full target back
      p_tN_to_wC = append_chain_to_pose(p_tN_to_wC, input_p)

      p_tC_to_wN_orig = Pose()
      p_tC_to_wN_orig.assign(p_tC_to_wN)
      p_tN_to_wC_orig = Pose()
      p_tN_to_wC_orig.assign(p_tN_to_wC)      

      # Extend termini 
      if verbose: print("Extending termini...")
      for i in range(termlen):
        res_type = core.chemical.ChemicalManager.get_instance().residue_type_set( 'fa_standard' ).get_representative_type_name1('A')
        residue = core.conformation.ResidueFactory.create_residue(res_type)

        core.conformation.idealize_position(1, p_tC_to_wN.conformation())
        p_tC_to_wN.prepend_polymer_residue_before_seqpos(residue, 1, True)
        core.conformation.idealize_position(len(core.pose.get_chain_residues(p_tC_to_wN,1)), p_tC_to_wN.conformation())
        p_tC_to_wN.append_polymer_residue_after_seqpos(residue, len(core.pose.get_chain_residues(p_tC_to_wN,1)), True)
    
        core.conformation.idealize_position(1, p_tN_to_wC.conformation())
        p_tN_to_wC.prepend_polymer_residue_before_seqpos(residue, 1, True)
        core.conformation.idealize_position(len(core.pose.get_chain_residues(p_tN_to_wC,1)), p_tN_to_wC.conformation()) 
        p_tN_to_wC.append_polymer_residue_after_seqpos(residue, len(core.pose.get_chain_residues(p_tN_to_wC,1)), True)

      # superimpose extended termini barrel
      ca_map = map_core_id_AtomID_core_id_AtomID()
      for r in range(1,len(core.pose.get_chain_residues(p_tC_to_wN_orig,1))+1):
        ca_map[AtomID(p_tC_to_wN.residue(r+termlen).atom_index("CA"), r+termlen)] = AtomID(p_tC_to_wN_orig.residue(r).atom_index("CA"), r)
      rmsd = pyrosetta.rosetta.core.scoring.superimpose_pose(p_tC_to_wN, p_tC_to_wN_orig, ca_map)
      if verbose: print(f'extended termini C-term barrel placement rmsd: {rmsd}')
      if rmsd > 1: continue

      ca_map = map_core_id_AtomID_core_id_AtomID()
      for r in range(1,len(core.pose.get_chain_residues(p_tN_to_wC_orig,1))+1):
        ca_map[AtomID(p_tN_to_wC.residue(r+termlen).atom_index("CA"), r+termlen)] = AtomID(p_tN_to_wC_orig.residue(r).atom_index("CA"), r)
      rmsd = pyrosetta.rosetta.core.scoring.superimpose_pose(p_tN_to_wC, p_tN_to_wC_orig, ca_map)
      if verbose: print(f'extended termini N-term barrel placement rmsd: {rmsd}')
      if rmsd > 1: continue

      p_tC_to_wN = append_chain_to_pose(p_tC_to_wN.split_by_chain(1), input_p)
      p_tN_to_wC = append_chain_to_pose(p_tN_to_wC.split_by_chain(1), input_p) 
    
      # generate rotation variants as final WRAPs
      save_rotations( p_tC_to_wN, input_p, nss, ss_nres, outprefix+'_C', termlen, looplen, rotation_samples, rotation_sample_angle )
      save_rotations( p_tN_to_wC, input_p, nss, ss_nres, outprefix+'_N', termlen, looplen, rotation_samples, rotation_sample_angle )
    
      if verbose: print(f'Created {outprefix} outputs')

  elif len(wrap) > 0:

  ##################################################
  ## PRE-MADE WRAP? Just place WRAP around target.

    # Wrap with pre-made input wrap
    # align based on top, bottom, and center of mass coords
    p1 = Pose()
    p1.assign(p_w)
    p2 = Pose()
    p2.assign(p_trans)
    rsd_set = p1.residue_type_set_for_pose( core.chemical.FULL_ATOM_t )
    tmp_res = core.conformation.ResidueFactory.create_residue( rsd_set.name_map( 'GLY' ) )
    p1.append_residue_by_jump(tmp_res, 1, "", "", False)
    p1.residue(len(p1.sequence())).atom(2).xyz(center_of_mass(ctopcoords))
    p1.append_residue_by_jump(tmp_res, 1, "", "", False)
    p1.residue(len(p1.sequence())).atom(2).xyz(center_of_mass(cbottomcoords))
    p1.append_residue_by_jump(tmp_res, 1, "", "", False)
    p1.residue(len(p1.sequence())).atom(2).xyz(core.pose.get_center_of_mass(p1))
    p2.append_residue_by_jump(tmp_res, 1, "", "", False)
    p2.residue(len(p2.sequence())).atom(2).xyz(center_of_mass(topcoords))
    p2.append_residue_by_jump(tmp_res, 1, "", "", False)
    p2.residue(len(p2.sequence())).atom(2).xyz(center_of_mass(bottomcoords))
    p2.append_residue_by_jump(tmp_res, 1, "", "", False)
    p2.residue(len(p2.sequence())).atom(2).xyz(core.pose.get_center_of_mass(p2))
    p2len = len(p2.sequence())
    p1len = len(p1.sequence())
    ca_map = map_core_id_AtomID_core_id_AtomID()
    ca_map[AtomID(p1.residue(p1len).atom_index("CA"), p1len)] = AtomID(p2.residue(p2len).atom_index("CA"), p2len)
    ca_map[AtomID(p1.residue(p1len-1).atom_index("CA"), p1len-1)] = AtomID(p2.residue(p2len-1).atom_index("CA"), p2len-1)
    ca_map[AtomID(p1.residue(p1len-2).atom_index("CA"), p1len-2)] = AtomID(p2.residue(p2len-2).atom_index("CA"), p2len-2)
  
    rmsd = pyrosetta.rosetta.core.scoring.superimpose_pose(p1,p2, ca_map, 0.00000001, False, False)
    if verbose: print(f'Wrap to target superposition rmsd: {rmsd}')
    if rmsd == 0.0:
      if verbose: print(f'Skipping due to alignment error.')
      continue

    # remove GLYs
    for i in range(0,3):
      p1.delete_residue_slow(len(p1.sequence()))
  
    targetp = Pose()
    targetp.assign(input_p)
    append_chain_to_pose(targetp,p1,1,True)
    # At this point the wrap should be placed around the target
  
    # TM top and bottom cross
    xyzt = center_of_mass(topcoords)
    xyzb = center_of_mass(bottomcoords)
    pax = pyrosetta.rosetta.numeric.cross(xyzt,xyzb)
    p1com = core.pose.get_center_of_mass(p1)
    ax = xyzt-xyzb
    offsetspinmover = protocols.rigid.RigidBodyDeterministicSpinMover(1,ax,p1com,1)
    offsetspinmoverfor = protocols.rigid.RigidBodyDeterministicSpinMover(1,ax,p1com,rotation_sample_angle)
    offsetspinmoverrev = protocols.rigid.RigidBodyDeterministicSpinMover(1,ax,p1com,-1*rotation_sample_angle)
  
    # place starting helix close to N or C term depending on which term to fuse wrap
    # N- fusion   =>  wrap-C N-target
    # C- fusion   =>  target-C N-wrap
    #
    besthNp = Pose()
    bestdN = 999.
    besthCp = Pose()
    bestdC = 999.
    nres = len(p1.sequence())
    targetpflipped = Pose()
    targetpflipped.assign(targetp)
    for j in range(360):
      offsetspinmover.apply(targetp)
      # N-target - C-wrap
      dN = (input_p.residue(1).atom(2).xyz()-targetp.residue(input_p_len+nres).atom(2).xyz()).norm()
      # C-target - N-wrap
      dC = (input_p.residue(input_p_len).atom(2).xyz()-targetp.residue(input_p_len+1).atom(2).xyz()).norm()
      if dN < bestdN:
        besthNp.assign(targetp)
        bestdN = dN
      if dC < bestdC:
        besthCp.assign(targetp)
        bestdC = dC
    # check flipped wrap
    flipmover = protocols.rigid.RigidBodyDeterministicSpinMover(1,pax,p1com,180)
    flipmover.apply(targetpflipped)
    for j in range(360):
      offsetspinmover.apply(targetpflipped)
      # N-target - C-wrap
      dN = (input_p.residue(1).atom(2).xyz()-targetpflipped.residue(input_p_len+nres).atom(2).xyz()).norm()
      # C-target - N-wrap
      dC = (input_p.residue(input_p_len).atom(2).xyz()-targetpflipped.residue(input_p_len+1).atom(2).xyz()).norm()
      if (dN < bestdN and not flip_wrap) or (dN > bestdN and flip_wrap):
        besthNp.assign(targetpflipped)
        bestdN = dN
      if (dC < bestdC and not flip_wrap) or (dC > bestdC and flip_wrap):
        besthCp.assign(targetpflipped)
        bestdC = dC
    if verbose:
      print(f'Best distance to Cterm {bestdC}')
      print(f'Best distance to Nterm {bestdN}')

    partial_diffusion_task_file_suffix = f'{input_pdb_name}_WRAP_{flipped_str}{wrap_name}'
    
    # generate rotation variants as final WRAPs
    outprefix = f'{input_pdb_name}_WRAP_{flipped_str}{wrap_name}_nres{nres}_wrap_N'
    save_rotations( besthNp, input_p, 0, 0, outprefix, 0, 0, rotation_samples, rotation_sample_angle, offsetspinmoverfor, offsetspinmoverrev, True)
    outprefix = f'{input_pdb_name}_WRAP_{flipped_str}{wrap_name}_nres{nres}_wrap_C'
    save_rotations( besthCp, input_p, 0, 0, outprefix, 0, 0, rotation_samples, rotation_sample_angle, offsetspinmoverfor, offsetspinmoverrev, True)

  else:

  ######################################
  ## HELICAL WRAP? The default.

    # Create parametric helical wraps
    # Both forward and reverse wraps
  
    # determine number of helices based on radius
    nss = number_of_helices(radius)
    odd_helix_n = nss%2
    # determine how long the helices should be
    ss_nres = helix_length(tmheight)
    if verbose:
      print(f'Helices: {nss}')
      print(f'Helix length: {ss_nres}')
    
    # make a helix of ss_nres residues
    p_h = generate_helix(ss_nres)
   
    # get top and bottom coords of helix
    nh = 4
    ctopcoords = vector_numeric_xyzVector_double_t()
    cbottomcoords = vector_numeric_xyzVector_double_t()
    for i in range(1,nh+1):
      ctopcoords.append(p_h.residue(i).atom(2).xyz())
    for i in range(0,nh):
      cbottomcoords.append(p_h.residue(ss_nres-i).atom(2).xyz())
    
    # align to target based on top, bottom, and center of mass coords
    p1 = Pose()
    p1.assign(p_h)
    p2 = Pose()
    p2.assign(p_trans)
    p2com = core.pose.get_center_of_mass(p2) # transmembrane center of mass
    rsd_set = p1.residue_type_set_for_pose( core.chemical.FULL_ATOM_t )
    tmp_res = core.conformation.ResidueFactory.create_residue( rsd_set.name_map( 'GLY' ) )
    p1.append_residue_by_jump(tmp_res, 1, "", "", False)
    p1.residue(len(p1.sequence())).atom(2).xyz(center_of_mass(ctopcoords))
    p1.append_residue_by_jump(tmp_res, 1, "", "", False)
    p1.residue(len(p1.sequence())).atom(2).xyz(center_of_mass(cbottomcoords))
    p1.append_residue_by_jump(tmp_res, 1, "", "", False)
    p1.residue(len(p1.sequence())).atom(2).xyz(core.pose.get_center_of_mass(p1))
    p2.append_residue_by_jump(tmp_res, 1, "", "", False)
    p2.residue(len(p2.sequence())).atom(2).xyz(center_of_mass(topcoords))
    p2.append_residue_by_jump(tmp_res, 1, "", "", False)
    p2.residue(len(p2.sequence())).atom(2).xyz(center_of_mass(bottomcoords))
    p2.append_residue_by_jump(tmp_res, 1, "", "", False)
    p2.residue(len(p2.sequence())).atom(2).xyz(core.pose.get_center_of_mass(p2))
    p2len = len(p2.sequence())
    p1len = len(p1.sequence())
    ca_map = map_core_id_AtomID_core_id_AtomID()
    ca_map[AtomID(p1.residue(p1len).atom_index("CA"), p1len)] = AtomID(p2.residue(p2len).atom_index("CA"), p2len)
    ca_map[AtomID(p1.residue(p1len-1).atom_index("CA"), p1len-1)] = AtomID(p2.residue(p2len-1).atom_index("CA"), p2len-1)
    ca_map[AtomID(p1.residue(p1len-2).atom_index("CA"), p1len-2)] = AtomID(p2.residue(p2len-2).atom_index("CA"), p2len-2)
  
    rmsd = pyrosetta.rosetta.core.scoring.superimpose_pose(p1,p2, ca_map, 0.00000001, False, False)
    if verbose: print(f'helix to target superposition rmsd: {rmsd}')
    if rmsd == 0.0:
      if verbose: print(f'Skipping due to alignment error.')
      continue  

    # remove GLYs
    for i in range(0,3):
      p1.delete_residue_slow(len(p1.sequence()))
    
    # transform helix to target surface
    pyrosetta.rosetta.core.pose.addVirtualResAsRoot(p1)
    transmover = protocols.rigid.RigidBodyTransMover(p1,1)
    xyzt = center_of_mass(topcoords)
    xyzb = center_of_mass(bottomcoords)
    pax = pyrosetta.rosetta.numeric.cross(xyzt,xyzb)
    transmover.trans_axis(pax)
    transmover.step_size(radius+rbuffer)
    transmover.apply(p1)
   
    # create n antiparallel helices around central axis
    ax = center_of_mass(topcoords) - center_of_mass(bottomcoords)
    spinmover = protocols.rigid.RigidBodyDeterministicSpinMover(1,ax,p2com,360/nss)
    spinmoverrev = protocols.rigid.RigidBodyDeterministicSpinMover(1,ax,p2com,-1*360/nss)
    offsetspinmoverfor = protocols.rigid.RigidBodyDeterministicSpinMover(1,ax,p2com,rotation_sample_angle)
    offsetspinmoverrev = protocols.rigid.RigidBodyDeterministicSpinMover(1,ax,p2com,-1*rotation_sample_angle)
    
    # place starting helix close to N or C term depending on which term to fuse wrap
    # N- fusion   =>  wrap-C N-target
    # C- fusion   =>  target-C N-wrap
    #
    besthNp = Pose() 
    bestdN = 999.
    besthCp = Pose() 
    bestdC = 999.
    # determine starting helix placement closest to target termini (for shortest linker distance)
    for j in range(360):
      offsetspinmoverfor.apply(p1)
      dN = (input_p.residue(1).atom(2).xyz()-p1.residue(ss_nres).atom(2).xyz()).norm()
      dC = (input_p.residue(input_p_len).atom(2).xyz()-p1.residue(1).atom(2).xyz()).norm()
      if dN < bestdN:
        besthNp.assign(p1)
        bestdN = dN
      if dC < bestdC:
        besthCp.assign(p1)
        bestdC = dC
      flipmover = protocols.rigid.RigidBodyDeterministicSpinMover(1,pax,core.pose.get_center_of_mass(p1),180)
      flipmover.apply(p1)
      dN = (input_p.residue(1).atom(2).xyz()-p1.residue(ss_nres).atom(2).xyz()).norm()
      dC = (input_p.residue(input_p_len).atom(2).xyz()-p1.residue(1).atom(2).xyz()).norm()
      if (dN < bestdN and not flip_wrap) or (dN > bestdN and flip_wrap):
        besthNp.assign(p1) 
        bestdN = dN
      if (dC < bestdC and not flip_wrap) or (dC > bestdC and flip_wrap): 
        besthCp.assign(p1)
        bestdC = dC
    if verbose:
      print(f'Best distance to Cterm {bestdC}')
      print(f'Best distance to Nterm {bestdN}')
 
    # add helices
    hNfor = Pose()
    hNfor.assign(besthNp)
    hCfor = Pose()
    hCfor.assign(besthCp)
    hNrev = Pose()
    hNrev.assign(besthNp)
    hCrev = Pose()
    hCrev.assign(besthCp)

    wrapCf = Pose()
    wrapCr = Pose()

    wrapNf_helices = []
    wrapNr_helices = []

    for i in range(1,nss+1):
      hNforrot = Pose()
      hNforrot.assign(hNfor)
      hCforrot = Pose()
      hCforrot.assign(hCfor)
      hNrevrot = Pose()
      hNrevrot.assign(hNrev)
      hCrevrot = Pose()
      hCrevrot.assign(hCrev)

      if h_rot_angle > 0:
        rota = h_rot_angle
        flipmoverNf = protocols.rigid.RigidBodyDeterministicSpinMover(1,pax,core.pose.get_center_of_mass(hNforrot),rota)
        flipmoverCf = protocols.rigid.RigidBodyDeterministicSpinMover(1,pax,core.pose.get_center_of_mass(hCforrot),rota)
        flipmoverNf.apply(hNforrot)
        flipmoverCf.apply(hCforrot)

        flipmoverNr = protocols.rigid.RigidBodyDeterministicSpinMover(1,pax,core.pose.get_center_of_mass(hNrevrot),rota)
        flipmoverCr = protocols.rigid.RigidBodyDeterministicSpinMover(1,pax,core.pose.get_center_of_mass(hCrevrot),rota)
        flipmoverNr.apply(hNrevrot)
        flipmoverCr.apply(hCrevrot)

      wrapNf_helices.append(hNforrot)
      append_chain_to_pose(wrapCf,hCforrot,1,False)
      spinmover.apply(hNfor)
      spinmover.apply(hCfor)

      flipmoverNf = protocols.rigid.RigidBodyDeterministicSpinMover(1,pax,core.pose.get_center_of_mass(hNfor),180)
      flipmoverNf.apply(hNfor)
      flipmoverCf = protocols.rigid.RigidBodyDeterministicSpinMover(1,pax,core.pose.get_center_of_mass(hCfor),180)
      flipmoverCf.apply(hCfor)
 
      wrapNr_helices.append(hNrevrot)
      append_chain_to_pose(wrapCr,hCrevrot,1,False)
      spinmoverrev.apply(hNrev)
      spinmoverrev.apply(hCrev)

      flipmoverNr = protocols.rigid.RigidBodyDeterministicSpinMover(1,pax,core.pose.get_center_of_mass(hNrev),180)
      flipmoverNr.apply(hNrev)
      flipmoverCr = protocols.rigid.RigidBodyDeterministicSpinMover(1,pax,core.pose.get_center_of_mass(hCrev),180)
      flipmoverCr.apply(hCrev)

    # hN should be the last chain (closest H-cterm to target-nterm)
    # so we need to reverse the order of helices to append into a single chain
    wrapNf = Pose()
    wrapNr = Pose()
    wrapNf_helices.reverse()
    wrapNr_helices.reverse()
    for h in wrapNf_helices:
      append_chain_to_pose(wrapNf,h,1,False)
    for h in wrapNr_helices:
      append_chain_to_pose(wrapNr,h,1,False)

    partial_diffusion_task_file_suffix = f'{input_pdb_name}_WRAP_{flipped_str}helix'

    outprefix = f'{input_pdb_name}_WRAP_{flipped_str}helix_n{nss}_nres{ss_nres}_looplen{looplen}_r{radius+rbuffer:.2f}_for_wrap_N'
    if verbose: print(f"Trying to close loops for {outprefix}")
    wrapNf, skip = closeloops(nss, ss_nres, wrapNf)
    if not skip:
      input_pNf = Pose()
      input_pNf.assign(input_p)
      append_chain_to_pose(input_pNf, wrapNf,1,True)
      save_rotations( input_pNf, input_p, nss, ss_nres, outprefix, 0, looplen, rotation_samples, rotation_sample_angle, offsetspinmoverfor, offsetspinmoverrev, True)

    outprefix = f'{input_pdb_name}_WRAP_{flipped_str}helix_n{nss}_nres{ss_nres}_looplen{looplen}_r{radius+rbuffer:.2f}_for_wrap_C' 
    if verbose: print(f"Trying to close loops for {outprefix}")
    wrapCf, skip = closeloops(nss, ss_nres, wrapCf)
    if not skip:
      input_pCf = Pose()
      input_pCf.assign(input_p)
      append_chain_to_pose(input_pCf, wrapCf,1,True)
      save_rotations( input_pCf, input_p, nss, ss_nres, outprefix, 0, looplen, rotation_samples, rotation_sample_angle, offsetspinmoverfor, offsetspinmoverrev, True)

    outprefix = f'{input_pdb_name}_WRAP_{flipped_str}helix_n{nss}_nres{ss_nres}_looplen{looplen}_r{radius+rbuffer:.2f}_rev_wrap_N'
    if verbose: print(f"Trying to close loops for {outprefix}")
    wrapNr, skip = closeloops(nss, ss_nres, wrapNr)
    if not skip:
      input_pNr = Pose()
      input_pNr.assign(input_p)
      append_chain_to_pose(input_pNr, wrapNr,1,True)
      save_rotations( input_pNr, input_p, nss, ss_nres, outprefix, 0, looplen, rotation_samples, rotation_sample_angle, offsetspinmoverfor, offsetspinmoverrev, True)

    outprefix = f'{input_pdb_name}_WRAP_{flipped_str}helix_n{nss}_nres{ss_nres}_looplen{looplen}_r{radius+rbuffer:.2f}_rev_wrap_C'
    if verbose: print(f"Trying to close loops for {outprefix}")
    wrapCr, skip = closeloops(nss, ss_nres, wrapCr)
    if not skip:
      input_pCr = Pose()
      input_pCr.assign(input_p)
      append_chain_to_pose(input_pCr, wrapCr,1,True)
      save_rotations( input_pCr, input_p, nss, ss_nres, outprefix, 0, looplen, rotation_samples, rotation_sample_angle, offsetspinmoverfor, offsetspinmoverrev, True)

  # Create task file containing partial diffusion commands
  if len(final_wraps) > 0:
    pdtfname = f'{partial_diffusion_task_file_prefix}_{partial_diffusion_task_file_suffix}.txt'
    print(f'Creating partial diffusion task(s) file: {pdtfname}')
    pdtf = open(pdtfname, 'w')
    ## Generate partial diffusion tasks
    for i in final_wraps:
        ca_atoms = []
        with open(i) as f:
          for l in f:
            if l.startswith('ATOM'):
              chain, atype, name3, resnum, x, y, z = read_pdb_atom(l)
              if atype == 'CA':
                ca_atoms.append([chain, atype, name3, resnum, x, y, z])
        contigstr = f'{chain1_len(ca_atoms)},0\\ {chain2_contigs(ca_atoms)}'
        prefix = i.split('.pdb')[0]+'_partial_diffusion/'+i.split('.pdb')[0]
        cmd = f'{rf_diffusion_container} {rf_diffusion} inference.output_prefix={prefix} '
        cmd += f'inference.input_pdb={i} contigmap.contigs=[\\\'{contigstr}\\\'] inference.num_designs={rf_partial_diffusions} denoiser.noise_scale_ca=0.5 denoiser.noise_scale_frame=0.5 diffuser.partial_T={rf_partial_diffusion_partialT}'
        
        pdtf.write(cmd+'\n')
    pdtf.close()  
    final_wraps = [] 
  
  
  
  
  
  
  
  
  
  
  
