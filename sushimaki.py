import os,sys
import numpy as np
import math
from math import sqrt,asin,pi,sin,tan,cos,atan
from scipy.optimize import fsolve
import glob
import argparse
import biolib

# Optional, point to your RFdiffusion installation
rf_diffusion_container = "/software/containers/SE3nv.sif"
rf_diffusion = "/projects/ml/rf_diffusion/run_inference.py"
partial_diffusions = 100
partial_diffusion_partialT = 20

# Dependencies
#
# PyRosetta
# https://www.pyrosetta.org
# 
# BBQ (requires Java)
# Gront, D., Kmiecik, S. and Kolinski, A. (2007), Backbone building from quadrilaterals: A fast and accurate 
# algorithm for protein backbone reconstruction from alpha carbon coordinates. J. Comput. Chem., 28: 1593-1597. 
# https://doi.org/10.1002/jcc.20624
#
# DeepTMHMM
# https://dtu.biolib.com/DeepTMHMM
# pip3 install pybiolib
#
#
#
# Reference for parametrically guided beta barrel backbone design:
#   Kim DE. et al. 2024. Parametrically guided design of soluble beta barrels and transmembrane nanopores using deep learning.
#
# References for cylinder generation code:
#   Naveed H. et al. JACS, 2012. Predicting three-dimensional structures of transmembrane domains of beta-barrel membrane proteins.
#   Dou, J. et al. Nature, 2018. De novo design of a fluorescence-activating β-barrel. 
#

# For BBQ
installdir = os.path.dirname(os.path.abspath(__file__))
os.environ['CLASSPATH'] = f'{installdir}/bioshell.bioinformatics-2.2/bioshell.bioinformatics-2.2.jar:{installdir}/bioshell.bioinformatics-2.2/bioshell.bioinformatics-2.2-mono.jar'

# beta wrap parameters
intra_strand_dist =  3.8
inter_strand_dist =  4.8
beta_radius_buffer_max = 10
beta_radius_buffer_min = 8
beta_tmheight_buffer = 1
termlen = 3

# helix wrap parameters
rbuffer = 6 # to add to transmembrane radius calculation to determine number of helices
hbuffer = 5 # to add to tmheight for helix length

max_dist_per_loop_res = 2.6 # to prevent loop modeling from hanging, this is the maximum distance per residue to be built between the start and stop CA positions. If the gap is too far apart to connect with loop modeling, the model will be skipped.

blen = 1 # len for inner and outer coords for superposition into cylinder

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
             If you choose --n that is too small or big to connect helices with --looplen
             loop modeling may hang. You may have to increase --looplen and/or --radius.

         '''))

parser.add_argument('--barrel_wrap', type=bool, default=False, help='Wrap with a beta barrel. n, nres, and radius will be automatically sampled. Helices are used by default.')
parser.add_argument('--n', type=int, default=0, help='Manually set n (helix count).')
parser.add_argument('--nres', type=int, default=0, help='Manually set nres (helix length).')
parser.add_argument('--radius', type=float, default=0, help='Manually set radius for helix wrap.')
parser.add_argument('--rot_angle', type=int, default=30, help='Manually set rotation sample angle.')
parser.add_argument('--h_rot_angle', type=float, default=0, help='Rotate angle of helices in wrap relative to target.')
parser.add_argument('--rot_n', type=int, default=3, help='Manually set rotation samples.')
parser.add_argument('--add_loops', type=bool, default=True, help='Add loops.')
parser.add_argument('--looplen', type=int, default=3, help='Loop length.')
parser.add_argument('--partial_diffusion_task_file', type=str, default="partial_diffusion_task_file.txt", help='RF partial diffusion task file.')
parser.add_argument('--debug', type=bool, default=False, help='Debug.')
parser.add_argument('--verbose', type=bool, default=False, help='Verbose output.')

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
        help='Residue numbers separated by spaces to wrap. The default is to use DeepTMHMM to identify transmembrane residues to wrap.',
        action='store',
        nargs='+',
        default=[]
        )

parser.add_argument(
        '--pdbs',
        help='Input PDBs.',
        action='store',
        required=True,
        nargs='+'
        )

args = vars(parser.parse_args())
exit = False

partial_diffusion_task_file = args['partial_diffusion_task_file']
barrel_wrap = args['barrel_wrap']
close_loops = args['add_loops']
looplen = args['looplen']
rotation_sample_angle = args['rot_angle']
h_rot_angle = args['h_rot_angle']
rotation_samples = args['rot_n']
manual_n = args['n']
manual_nres = args['nres']
manual_radius = args['radius']
debug = args['debug']
verbose = args['verbose']
all_residues_to_wrap_ = args['all_residues_to_wrap']
top_residues_to_wrap_ = args['top_residues_to_wrap']
bottom_residues_to_wrap_ = args['bottom_residues_to_wrap']

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
if exit:
  parser.print_help(sys.stderr)
  sys.exit(1)


# PARAM n (number of helices function of TM radius)
def number_of_helices(radius):
  if manual_n > 0:
    return manual_n
  return int(np.pi*2*(radius+rbuffer)/7) # 110/16 circumference divided by 16 = 6.875 based on 8efo example w/ 16 helices <- LJ's original count

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
  return int(((tmheight+hbuffer)/5.4)*3.6)


from pyrosetta import *
from pyrosetta.rosetta import *
from pyrosetta.rosetta.std import map_core_id_AtomID_core_id_AtomID
from pyrosetta.rosetta.core.id import AtomID
from pyrosetta.rosetta.protocols.loops.loop_closure.ccd import *
from pyrosetta.rosetta.std import (
    vector_numeric_xyzVector_double_t,
)
from pyrosetta.rosetta.numeric import xyzVector_double_t

if verbose:
  init( "-beta_nov16 " )
else:
  init( "-beta_nov16 -mute all" )

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

#>2ge4A_model5_rosetta.pdb | BETA
#NTWYTGAKLGFSQYHDTGFINNNGPTHENQLGAGAFGGYQVNPYVGFEMGYDFLGRMPYKGSVENGAYKAQGVQLTAKLGYPITDDLDIYTRLGGMVFRADTKSNVYGKNHDTGVSPVFAGGVEYAITPEIATRLEYQFTNNIGDAHTIGTRPDNGMLSLGVSYRFGQGEAA
#XX1BBBBBB2OOOOOOOOOOOOOOOOOOOOOO2BBBBBB1PPPP1BBBBB2OOOOOOOOOOOOOOOOOOOOOO2BBBBBB1PPPPPP1BBBBBB2OOOOOOOOOOOOOOOOOOOOOO2BBBBBB1PPPPP1BBBBBB2OOOOOOOOOOOOOOOOOOO2BBBBBB1PPPPPPP

  # note this is not a robust DeepTMHMM result parser, at some point I'll make it more robust to handle signal tms etc
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
              for add in range(0,blen+1):
                outer.append(tlen+add)
                outer_trans.append(i+1+add)
            elif l[i+1] == 'O':
              for sub in range(0,blen+1):
                outer.append(tlen-sub)
                outer_trans.append(i+1-sub)
            elif l[i-1] == 'I' or l[i-1] == 'P' or l[i-1] == 'S' or l[i-1] == 'X':
              for add in range(0,blen+1):
                inner.append(tlen+add)
                inner_trans.append(i+1+add)
            elif i == llen-1 or (l[i+1] == 'I' or l[i+1] == 'P' or l[i+1] == 'S' or l[i+1] == 'X'): # input Cterm is inner
              for sub in range(0,blen+1):
                inner.append(tlen-sub)
                inner_trans.append(i+1-sub)
        break
      lcnt += 1

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

def generate_helix(n):
  aa = ''
  for i in range(n):
    aa += 'A'
  hp = pose_from_sequence(aa, "fa_standard") 
  for i in range(1,n+1):
    hp.set_phi(i, -57)
    hp.set_psi(i, -47)
  return hp

EXT_PHI = -150.0
EXT_PSI = +150.0
EXT_OMG = 180.0

def add_loops_to_strands(pose,nstrands,nresstrand,looplen):
  # add loops to cylinder
  print("Adding loops...")
  p = Pose()
  skip = False
  for res in pyrosetta.rosetta.core.pose.get_chain_residues(pose,1):
    core.conformation.remove_upper_terminus_type_from_conformation_residue(pose.conformation(), res.seqpos())
    core.conformation.remove_lower_terminus_type_from_conformation_residue(pose.conformation(), res.seqpos())
    p.append_residue_by_bond(res)
  inserted = 0
  loops = protocols.loops.Loops()
  for i in range(1,nstrands):  # strands
    previous = i*nresstrand+inserted
    for j in range(looplen):
      insertpos = i*nresstrand+inserted
      res_type = core.chemical.ChemicalManager.get_instance().residue_type_set( 'fa_standard' ).get_representative_type_name1('A')
      residue = core.conformation.ResidueFactory.create_residue(res_type)
      p.append_polymer_residue_after_seqpos(residue, insertpos, True)
      inserted += 1
    cutpoint = previous+looplen

    # see if distance is feasable
    start_stop_dist = (p.residue(previous).atom(2).xyz()-p.residue(cutpoint+1).atom(2).xyz()).norm()
    skip = False
    print(f'nstrands: {nstrands} nresstrand: {nresstrand} endpoints {previous} {cutpoint+1} {start_stop_dist} / looplen {looplen} {start_stop_dist/looplen}')
    if start_stop_dist/looplen > max_dist_per_loop_res:
      print(f'endpoints {start_stop_dist} / looplen {looplen} {start_stop_dist/looplen} > {max_dist_per_loop_res} so cannot close loops')
      skip = True
      break

    loop = protocols.loops.Loop(previous, cutpoint+1, cutpoint, 0.0, True)
    loops.add_loop(loop)
    protocols.loops.set_single_loop_fold_tree(p, loop)
    # extend loops before closing
    for l in range(loop.start()+1,loop.stop()):
      core.conformation.idealize_position(l, p.conformation())
      p.set_phi(l, EXT_PHI)
      p.set_psi(l, EXT_PSI)
      p.set_omega(l, EXT_OMG)
  return p, loops, skip


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
    if verbose:
      print(f'Adding {looplen} residue loop connecting {Npos} to {Cpos}')
    inserted = 0
    cutpoint = 0
    for j in range(looplen):
      insertpos = Cpos+inserted
      res_type = core.chemical.ChemicalManager.get_instance().residue_type_set( 'fa_standard' ).get_representative_type_name1('G')
      residue = core.conformation.ResidueFactory.create_residue(res_type)
      if j < looplen/2:
        cutpoint = insertpos+1
        p.append_polymer_residue_after_seqpos(residue, insertpos, True)
        inserted += 1
      else:
        p.prepend_polymer_residue_before_seqpos(residue, insertpos+1, True)

    # see if distance is feasable
    start_stop_dist = (p.residue(Cpos).atom(2).xyz()-p.residue(Cpos+looplen+1).atom(2).xyz()).norm()
    skip = False
    print(f'endpoints {Cpos} {Cpos+looplen+1} {start_stop_dist} / looplen {looplen} {start_stop_dist/looplen}')
    if start_stop_dist/looplen > max_dist_per_loop_res:
      print(f'endpoints {start_stop_dist} / looplen {looplen} {start_stop_dist/looplen} > {max_dist_per_loop_res} so cannot close loops')
      skip = True
      break

    loop = protocols.loops.Loop(Cpos, Cpos+looplen+1, cutpoint, 0.0, True)
    loops.add_loop(loop)

    protocols.loops.set_single_loop_fold_tree(p, loop)

    added = len(p.sequence())-pstartlen
    if verbose:
      print(f'Added loop: {Cpos} {Cpos+looplen+1} {cutpoint}')
  return p, loops, skip

def closeloops(pose):
  Nposs = []
  Cposs = []

  for i in range(1,n):
    Cposs.append(i*nres)
    Nposs.append(i*nres+1)
  pose, loops, skip = add_loops(pose,Nposs,Cposs)
  if not skip:
    lm = protocols.loop_modeler.LoopModeler()
    lm.set_loops(loops)
    lm.enable_build_stage()
    lm.disable_centroid_stage()
    lm.disable_fullatom_stage()
    if verbose:
      print(f'Closing length {looplen} loops...')
    if debug:
      pose.dump_pdb('preclose.pdb')
    lm.apply(pose)
    if debug:
      pose.dump_pdb('postclose.pdb')
  return pose, skip

def save_rotations(p, nstrands, nresstrand, output_prefix, termlen, looplen, move_angle=30):
  global blen
  pose = Pose()
  pose.assign(p)
  ctop = []
  cbottom = []
  ctopcoords = vector_numeric_xyzVector_double_t()
  cbottomcoords = vector_numeric_xyzVector_double_t()
  ccoords = vector_numeric_xyzVector_double_t()
  for i in range(1, nstrands+1):
    clen = (i*nresstrand)+termlen+(i-1)*looplen
    #print(f'strands: {nstrands} nresstrand: {nresstrand} {termlen} {looplen} {i} clen: {clen} {output_prefix}')
    for j in range(clen-nresstrand+1,clen+1):
      #print(f'ccoords: {j}')
      ccoords.append(pose.residue(j).atom(2).xyz())
    if not i%2:
      for add in range(1,blen+1):
        ctop.append(clen-nresstrand+add+1)
      for sub in range(1,blen+1):
        cbottom.append(clen-sub+1)
    else:
      for add in range(1,blen+1):
        cbottom.append(clen-nresstrand+add+1)
      for sub in range(1,blen+1):
        ctop.append(clen-sub+1)
  #print('+'.join(map(str,ctop)))
  #print('+'.join(map(str,cbottom)))

  for i in ctop:
    ctopcoords.append(pose.residue(i).atom(2).xyz())
  for i in cbottom:
    cbottomcoords.append(pose.residue(i).atom(2).xyz())

  # rotate barrel
  spinmover = protocols.rigid.RigidBodyDeterministicSpinMover(1,center_of_mass(ctopcoords)-center_of_mass(cbottomcoords),center_of_mass(ccoords),move_angle)
  for i in range(1,int(360/move_angle)+1):
    spinmover.apply(pose)
    print(f'{output_prefix}_rot{i*move_angle:03d}.pdb')
    pose.dump_pdb(f'{output_prefix}_rot{i*move_angle:03d}.pdb')

def chain1_len(ca_atoms):
  chainAlen = 0
  prevchain = ''
  for ca in ca_atoms:
    if len(prevchain) and ca[0] != prevchain:
      break
    chainAlen += 1
    prevchain = ca[0]
  return chainAlen

def xyzdist(xyz1, xyz2):
  nxyz1 = np.array(xyz1)
  nxyz2 = np.array(xyz2)
  return np.linalg.norm(nxyz1 - nxyz2)

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
    d = xyzdist(ca_atoms[i][4:7],ca_atoms[i+1][4:7])
    if d > 4.2: # chainbreak
      target_chain_breaks.append(f'{start}-{ca_atoms[i][3]}')
      start = ca_atoms[i+1][0]+str(ca_atoms[i+1][3])
  target_chain_breaks.append(f'{start}-{ca_atoms[-1][3]}')
  return ','.join(target_chain_breaks)



## MAIN
pdtf = open(partial_diffusion_task_file, 'w')

for pdb in pdbs:

  input_pdb = pdb
  input_pdb_name = input_pdb.split('.pdb')[0].split('/')[-1]
  input_p = pose_from_file(input_pdb)
  input_p_len = len(input_p.sequence())

  trans = []
  top = []
  bottom = []
  top_trans = []
  bottom_trans = []

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


  if barrel_wrap:
  

    # create parametric cylinders
    
    # First sample range of parameters for optimal wrap height and radii
    # 27-32 angstroms lipid bilayer height
    barrel_params = []
    for nstrands in range(6,1000):  # sample 6 to 999 strands
      for shear in range(nstrands,(nstrands*2)+1): # shear
        if shear%2 == 0:
          # cylinder radius based on barrel parameters
          r = math.sqrt((shear*intra_strand_dist)**2+(nstrands*inter_strand_dist)**2)/(2*nstrands*math.sin(math.pi/nstrands))   # cylinder radius

          if r > radius+beta_radius_buffer_min and r < radius+beta_radius_buffer_max:
            for nres_per_strand in range(6,30):   # nres per strand

              # coil angle
              theta=asin(shear*intra_strand_dist/(2*math.pi*r))

              def disNextRes(x):
                return sqrt(r**2*(2-2*cos(x))+(r*x/tan(theta))**2)-intra_strand_dist
              def disNextStrand(x):
                return sqrt(r**2*(2-2*cos(x+2*math.pi/nstrands))+(r*x/tan(theta))**2)-inter_strand_dist
              delta_t1 = fsolve(disNextRes,0)[0]
              delta_t2 = fsolve(disNextStrand,0)[0]
              def dis(x,y):
                s=0
                for i in range(len(x)): s += (x[i]-y[i])**2
                return sqrt(s)
              for ns in range(1,nstrands+1):
                phi = (ns-1)*2*math.pi/nstrands
                dt2 = (delta_t2)*(ns-1)

              heighta = []
              n0 = 0 #number of residues at the strand
              for j in range(800):
                dt1 = (delta_t1)*j
                dt = dt1+dt2
                x=r*cos(dt+phi)
                y=r*sin(dt+phi)
                z=r*dt/tan(theta)
                if z<0:continue
                n0 += 1
                if n0>nres_per_strand: break # +4: break #jump out of the loop
                sign = 1 if j%2 == 0 else -1
                heighta.append(z)
              h = heighta[-1]-heighta[0]
              if h < tmheight+beta_tmheight_buffer and h > tmheight-beta_tmheight_buffer:  
                barrel_params.append( [ nstrands, shear, nres_per_strand, h, r ] )              

    for barrel_param in barrel_params:
      nstrands = barrel_param[0]
      shear = barrel_param[1]
      nres_per_strand = barrel_param[2]
      barrel_h = barrel_param[3]
      barrel_r = barrel_param[4]
      outpdbA = f'temp_barrelCylinder_n{nstrands}_S{shear}_nres{nres_per_strand}.pdb'
      r=sqrt((nstrands*inter_strand_dist)**2+(shear*intra_strand_dist)**2)/(2*pi)
      theta=asin(shear*intra_strand_dist/(2*pi*r))*1
      def disNextRes(x):
        return sqrt(r**2*(2-2*cos(x))+(r*x/tan(theta))**2)-intra_strand_dist
      def disNextStrand(x):
        return sqrt(r**2*(2-2*cos(x+2*pi/nres_per_strand))+(r*x/tan(theta))**2)-inter_strand_dist
      delta_t1 = fsolve(disNextRes,0)[0]
      delta_t2 = fsolve(disNextStrand,0)[0]
      def dis(x,y):
        s=0
        for i in range(len(x)): s += (x[i]-y[i])**2
        return sqrt(s)

      coor = []
      for ns in range(1,nstrands+1):
        phi = (ns-1)*2*pi/nstrands
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
          if n0>nres_per_strand: break # +4: break #jump out of the loop
          sign = 1 if j%2 == 0 else -1
          coor_strand.append([x,y,z,sign])
        coor.append(coor_strand)

      chain_ids = 'ABCDEFGHIJKLMNOPQSTUVWXYZ'
      foA = open(outpdbA,'w')
      resi = 1
      atomi = 2
      for i in range(len(coor)):
        coor_strand = coor[i]
        if i%2 == 0: coor_strand.reverse()
        for j in range(len(coor_strand)):
          (x,y,z,sign) = coor_strand[j]
          foA.write("ATOM  %5d  CA  VAL %1s%4d    %8.3f%8.3f%8.3f  1.00  0.00           %1d\n" % (atomi,'A',resi,x,y,z,sign))
          resi += 1
          atomi += 4
      foA.close()

      outprefix = f'{input_pdb_name}_WRAP_barrel_n{nstrands}_S{shear}_nres{nres_per_strand}_tradius{radius:.0f}_cradius{barrel_r:.0f}_cheight{barrel_h:.0f}'
      if os.path.exists(outprefix+'.pdb'):
        print(f'{outprefix}.pdb already exists.. skipping')
        continue

      # create backbones from CA only cylinders using BBQ
      cylinder_p = Pose() 
      os.system(f"java apps.BBQ -ip="+outpdbA)
      # cylinder pose
      cylinder_p = pose_from_file(outpdbA.split('.pdb')[0]+'-bb.pdb')

      cylinder_p_len = len(cylinder_p.sequence())
      cylinder_p.center()
   
      # cleanup temp pdbs
      for rmf in [ outpdbA, outpdbA+'.ss2', outpdbA.split('.pdb')[0]+'_rebuilt.pdb', outpdbA.split('.pdb')[0]+'-bb.pdb' ]:
        if os.path.exists(rmf): os.remove(rmf)
 
      # get top and bottom coords
      ctop = []
      cbottom = []
      ctopcoords = vector_numeric_xyzVector_double_t()
      cbottomcoords = vector_numeric_xyzVector_double_t()
      for i in range(1,nstrands+1):
        clen = i*nres_per_strand
        if not i%2:
          for add in range(1,blen+1):
            ctop.append(clen-nres_per_strand+add)
          for sub in range(1,blen+1):
            cbottom.append(clen-sub+1)
        else:
          for add in range(1,blen+1):
            cbottom.append(clen-nres_per_strand+add)
          for sub in range(1,blen+1):
            ctop.append(clen-sub+1)
      for i in ctop:
        ctopcoords.append(cylinder_p.residue(i).atom(2).xyz())
      for i in cbottom:
        cbottomcoords.append(cylinder_p.residue(i).atom(2).xyz())

      # align based on top, bottom, and center of mass coords
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
      ca_map[AtomID(p2.residue(p2len).atom_index("CA"), p2len)] = AtomID(p1.residue(p1len).atom_index("CA"), p1len)
      ca_map[AtomID(p2.residue(p2len-1).atom_index("CA"), p2len-1)] = AtomID(p1.residue(p1len-1).atom_index("CA"), p1len-1)
      ca_map[AtomID(p2.residue(p2len-2).atom_index("CA"), p2len-2)] = AtomID(p1.residue(p1len-2).atom_index("CA"), p1len-2)
    
      rmsd = pyrosetta.rosetta.core.scoring.superimpose_pose(p2,p1, ca_map, 0.00000001, False, False)
      print(f'cylinder to transmembrane superposition rmsd: {rmsd}')
    
      p1_flipped = Pose()
      p1_flipped.assign(p1)
      p2_flipped = Pose()
      p2_flipped.assign(p2)
    
      # if even stranded, make a flipped cylinder for N-term fusion to C-term of cylinder
      ca_map = map_core_id_AtomID_core_id_AtomID()
      ca_map[AtomID(p2_flipped.residue(p2len).atom_index("CA"), p2len)] = AtomID(p1_flipped.residue(p1len).atom_index("CA"), p1len)
      ca_map[AtomID(p2_flipped.residue(p2len-1).atom_index("CA"), p2len-1)] = AtomID(p1_flipped.residue(p1len-2).atom_index("CA"), p1len-2)
      ca_map[AtomID(p2_flipped.residue(p2len-2).atom_index("CA"), p2len-2)] = AtomID(p1_flipped.residue(p1len-1).atom_index("CA"), p1len-1)
    
      rmsd = pyrosetta.rosetta.core.scoring.superimpose_pose(p2_flipped,p1_flipped, ca_map, 0.00000001, False, False)
      print(f'cylinder to transmembrane flipped superposition rmsd: {rmsd}')
      for i in range(0,3):
        p2.delete_residue_slow(len(p2.sequence()))
        p1.delete_residue_slow(len(p1.sequence()))
        p2_flipped.delete_residue_slow(len(p2_flipped.sequence()))
        p1_flipped.delete_residue_slow(len(p1_flipped.sequence()))
    
      cylinder_p.assign(p1)
      cylinder_p_flipped = Pose()
      cylinder_p_flipped.assign(p1_flipped)
    
      # append transmembrane part of target as a new chain
      core.pose.append_pose_to_pose(cylinder_p, p2)
      core.pose.append_pose_to_pose(cylinder_p_flipped, p2_flipped)
    
      # At this point the target transmembrane portion should be placed well in the cylinder and flipped cylinder
    
      # rotate cylinder until transmembrane C-term is close to cylinder N-term
      move_angle = 5
      spinmover = protocols.rigid.RigidBodyDeterministicSpinMover(cylinder_p.num_jump(),center_of_mass(ctopcoords)-center_of_mass(cbottomcoords),core.pose.get_center_of_mass(p1),move_angle)
      mindist = 9999999.
      min_i = 0
      check_p = Pose()
      check_p.assign(cylinder_p)
    
      for i in range(1,int(360/move_angle)+1):
        spinmover.apply(check_p)
        dist = (check_p.residue(1).atom(2).xyz() - check_p.residue(len(check_p.sequence())).atom(2).xyz()).norm()
        if dist < mindist:
          mindist = dist
          min_i = i
      for i in range(1,min_i+1):
        spinmover.apply(cylinder_p)
    
      # rotate flipped cylinder until transmembrane N-term is close to cylinder C-term
      spinmover = protocols.rigid.RigidBodyDeterministicSpinMover(cylinder_p_flipped.num_jump(),center_of_mass(cbottomcoords)-center_of_mass(ctopcoords),core.pose.get_center_of_mass(p1_flipped),move_angle)
      mindist = 9999999.
      min_i = 0
      check_p = Pose()
      check_p.assign(cylinder_p_flipped)
      for i in range(1,int(360/move_angle)+1):
        spinmover.apply(check_p)
        dist = (check_p.residue(cylinder_p_len).atom(2).xyz() - check_p.residue(cylinder_p_len+1).atom(2).xyz()).norm()
        if dist < mindist:
          mindist = dist
          min_i = i
      for i in range(1,min_i+1):
        spinmover.apply(cylinder_p_flipped)
    
      ## place full target onto minimized pose
      p_copy = Pose()
      p_copy.assign(input_p)
      ca_map = map_core_id_AtomID_core_id_AtomID()
      ti = 0
      for r1 in range(cylinder_p_len+1,len(cylinder_p.sequence())+1):
        ca_map[AtomID(p_copy.residue(trans[ti]).atom_index("CA"), trans[ti])] = AtomID(cylinder_p.residue(r1).atom_index("CA"), r1)
        ti += 1
      rmsd = pyrosetta.rosetta.core.scoring.superimpose_pose(p_copy, cylinder_p, ca_map)
      print(f'target placement rmsd: {rmsd}')
      if rmsd > 1: continue    

      p_copy_flipped = Pose()
      p_copy_flipped.assign(input_p)
      ca_map = map_core_id_AtomID_core_id_AtomID()
      ti = 0
      for r1 in range(cylinder_p_len+1,len(cylinder_p_flipped.sequence())+1):
        ca_map[AtomID(p_copy_flipped.residue(trans[ti]).atom_index("CA"), trans[ti])] = AtomID(cylinder_p_flipped.residue(r1).atom_index("CA"), r1)
        ti += 1
      rmsd = pyrosetta.rosetta.core.scoring.superimpose_pose(p_copy_flipped, cylinder_p_flipped, ca_map)
      print(f'target flipped placement rmsd: {rmsd}')
      if rmsd > 1: continue
 
      # add loops to cylinder
      p = Pose()
      p, loops, skip = add_loops_to_strands(cylinder_p,nstrands,nres_per_strand,looplen)

      if skip: continue

      # add target back
      p = append_chain_to_pose(p,p_copy)
    
      # add loops to flipped cylinder
      p_flipped = Pose()
      p_flipped, loops_flipped, skip = add_loops_to_strands(cylinder_p_flipped,nstrands,nres_per_strand,looplen)

      if skip: continue

      # add target back
      p_flipped = append_chain_to_pose(p_flipped,p_copy_flipped)
    
      # close loops (for some reason this gets stuck in an infinite loop somewhere somehow in pyrosetta !! why??
      if close_loops:
        lm = protocols.loop_modeler.LoopModeler()
        lm.set_loops(loops)
        lm.enable_build_stage()
        lm.disable_centroid_stage()
        lm.disable_fullatom_stage()
        print("Closing loops...")
        lm.apply(p)
    
        lm.set_loops(loops_flipped)
        lm.enable_build_stage()
        lm.disable_centroid_stage()
        lm.disable_fullatom_stage()
        print("Closing loops...")
    
        lm.apply(p_flipped)
    
      p_orig = Pose()
      p_orig.assign(p)
    
      p_orig_flipped = Pose()
      p_orig_flipped.assign(p_flipped)
    
      # add extended termini with some phi/psi randomness
      print("Adding termini...")
      for i in range(termlen):
        res_type = core.chemical.ChemicalManager.get_instance().residue_type_set( 'fa_standard' ).get_representative_type_name1('A')
        residue = core.conformation.ResidueFactory.create_residue(res_type)
        core.conformation.idealize_position(1, p.conformation())
        p.prepend_polymer_residue_before_seqpos(residue, 1, True)
        core.conformation.idealize_position(1, p.conformation())
        core.conformation.idealize_position(len(core.pose.get_chain_residues(p,1)), p.conformation())
        p.append_polymer_residue_after_seqpos(residue, len(core.pose.get_chain_residues(p,1)), True)
    
        core.conformation.idealize_position(1, p_flipped.conformation())
        p_flipped.prepend_polymer_residue_before_seqpos(residue, 1, True)
        core.conformation.idealize_position(1, p_flipped.conformation())
        core.conformation.idealize_position(len(core.pose.get_chain_residues(p_flipped,1)), p_flipped.conformation()) 
        p_flipped.append_polymer_residue_after_seqpos(residue, len(core.pose.get_chain_residues(p_flipped,1)), True)
    
      # superimpose extended termini barrel
      ca_map = map_core_id_AtomID_core_id_AtomID()
      for r in range(1,len(core.pose.get_chain_residues(p_orig,1))+1):
        ca_map[AtomID(p_orig.residue(r).atom_index("CA"), r)] = AtomID(p.residue(r+termlen).atom_index("CA"), r+termlen)
      rmsd = pyrosetta.rosetta.core.scoring.superimpose_pose(p_orig, p, ca_map)
      print(f'extended termini barrel placement rmsd: {rmsd}')
      if rmsd > 1: continue    

      ca_map = map_core_id_AtomID_core_id_AtomID()
      for r in range(1,len(core.pose.get_chain_residues(p_orig_flipped,1))+1):
        ca_map[AtomID(p_orig_flipped.residue(r).atom_index("CA"), r)] = AtomID(p_flipped.residue(r+termlen).atom_index("CA"), r+termlen)
      rmsd = pyrosetta.rosetta.core.scoring.superimpose_pose(p_orig_flipped, p_flipped, ca_map)
      print(f'extended termini barrel flipped placement rmsd: {rmsd}')
      if rmsd > 1: continue    
    
      finalp = append_chain_to_pose( p_orig.split_by_chain(2), p, 1 )
      finalp_flipped = append_chain_to_pose( p_orig_flipped.split_by_chain(2), p_flipped, 1 )
    
      # replace target with orig coords (since full atom was lost for loop closure and termini extension)
      ca_map = map_core_id_AtomID_core_id_AtomID()
      for r in range(1,len(core.pose.get_chain_residues(p_copy,1))+1):
        ca_map[AtomID(finalp.residue(r).atom_index("CA"), r)] = AtomID(p_copy.residue(r).atom_index("CA"), r)
      rmsd = pyrosetta.rosetta.core.scoring.superimpose_pose(finalp, p_copy, ca_map)
      print(f'target placement rmsd: {rmsd}')
      if rmsd > 1: continue    
    
      ca_map = map_core_id_AtomID_core_id_AtomID()
      for r in range(1,len(core.pose.get_chain_residues(p_copy_flipped,1))+1):
        ca_map[AtomID(finalp_flipped.residue(r).atom_index("CA"), r)] = AtomID(p_copy_flipped.residue(r).atom_index("CA"), r)
      rmsd = pyrosetta.rosetta.core.scoring.superimpose_pose(finalp_flipped, p_copy_flipped, ca_map)
      print(f'target flipped placement rmsd: {rmsd}')
      if rmsd > 1: continue    
    
      finalp_complex = Pose()
      finalp_complex.assign(finalp)
      finalp_flipped_complex = Pose()
      finalp_flipped_complex.assign(finalp_flipped)
      finalp = append_chain_to_pose( Pose(p_copy), finalp, 2, False )
      finalp_flipped = append_chain_to_pose( finalp_flipped.split_by_chain(2), Pose(p_copy_flipped), 1, False )
      finalp_complex = append_chain_to_pose( finalp_complex.split_by_chain(2), Pose(p_copy), 1, True )
    
    
      finalp_flipped_complex = append_chain_to_pose( finalp_flipped_complex.split_by_chain(2), Pose(p_copy_flipped), 1, True )
    
      #finalp.dump_pdb(outprefix+'.pdb')
      #finalp_flipped.dump_pdb(outprefix+'_flipped.pdb')
      finalp_complex.dump_pdb(outprefix+'_C.pdb')
      finalp_flipped_complex.dump_pdb(outprefix+'_N.pdb')
    
      # generate rotation variants
      save_rotations( finalp_complex, nstrands, nres_per_strand, outprefix+'_C', termlen, looplen, rotation_sample_angle )
      save_rotations( finalp_flipped_complex, nstrands, nres_per_strand, outprefix+'_N', termlen, looplen, rotation_sample_angle )
    
      print(f'Created {outprefix} outputs')

  else:
  
    # determine number of helices based on radius
    n = number_of_helices(radius)
    odd_helix_n = n%2
    print(f'Helices: {n}')
    
    # determine how long the helices should be
    nres = helix_length(tmheight)
    print(f'Helix length: {nres}')
    
    # make helix
    p_h = generate_helix(nres)
   
    # get top and bottom coords of helix
    nh = 4
    ctopcoords = vector_numeric_xyzVector_double_t()
    cbottomcoords = vector_numeric_xyzVector_double_t()
    for i in range(1,nh+1):
      ctopcoords.append(p_h.residue(i).atom(2).xyz())
    for i in range(0,nh):
      cbottomcoords.append(p_h.residue(nres-i).atom(2).xyz())
    
    # align based on top, bottom, and center of mass coords
    # this is ugly
    # \_(:/)_/
    #
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
    print(f'helix to transmembrane superposition rmsd: {rmsd}')
  
    # remove GLYs
    for i in range(0,3):
      p1.delete_residue_slow(len(p1.sequence()))
    
    # transform helix to transmembrane surface
    pyrosetta.rosetta.core.pose.addVirtualResAsRoot(p1)
    transmover = protocols.rigid.RigidBodyTransMover(p1,1)
    xyzt = center_of_mass(topcoords)
    xyzb = center_of_mass(bottomcoords)
    pax = pyrosetta.rosetta.numeric.cross(xyzt,xyzb)
    transmover.trans_axis(pax)
    transmover.step_size(radius+rbuffer)
    transmover.apply(p1)
   
    # create n antiparallel helices around central axis
    # x rotation_samples (rotation_sample_angle)
    ax = center_of_mass(topcoords) - center_of_mass(bottomcoords)
    spinmover = protocols.rigid.RigidBodyDeterministicSpinMover(1,ax,p2com,360/n)
    spinmoverrev = protocols.rigid.RigidBodyDeterministicSpinMover(1,ax,p2com,-1*360/n)
    offsetspinmover = protocols.rigid.RigidBodyDeterministicSpinMover(1,ax,p2com,rotation_sample_angle)
    offsetspinmoverrev = protocols.rigid.RigidBodyDeterministicSpinMover(1,ax,p2com,-1*rotation_sample_angle)
    
    # place starting helix close to N or C term depending on which term to fuse wrap
    # N- fusion   =>  wrap-C N-target
    # C- fusion   =>  target-C N-wrap
    #
    besthNp = Pose() 
    bestdN = 999.
    besthCp = Pose() 
    bestdC = 999.
  
    # determin starting helix placement closest to target termini (for shortest linker distance)
    for j in range(360):
      offsetspinmover.apply(p1)
      dN = (input_p.residue(1).atom(2).xyz()-p1.residue(nres).atom(2).xyz()).norm()
      dC = (input_p.residue(input_p_len).atom(2).xyz()-p1.residue(1).atom(2).xyz()).norm()
      if dN < bestdN:
        besthNp.assign(p1)
        bestdN = dN
      if dC < bestdC:
        besthCp.assign(p1)
        bestdC = dC
      flipmover = protocols.rigid.RigidBodyDeterministicSpinMover(1,pax,core.pose.get_center_of_mass(p1),180)
      flipmover.apply(p1)
      dN = (input_p.residue(1).atom(2).xyz()-p1.residue(nres).atom(2).xyz()).norm()
      dC = (input_p.residue(input_p_len).atom(2).xyz()-p1.residue(1).atom(2).xyz()).norm()
      if dN < bestdN:
        besthNp.assign(p1) 
        bestdN = dN
      if dC < bestdC:
        besthCp.assign(p1)
        bestdC = dC
  
    if verbose:
      print(f'Best distance to Cterm {bestdC}')
      print(f'Best distance to Nterm {bestdN}')
  
    for isN, starth in enumerate( [ besthCp, besthNp ] ):
      termini = 'C'
      if isN:
        termini = 'N'
      prewraps = []
      rotation = 0
  
      thisp1 = Pose()
      thisp1.assign(starth)
  
      if debug:
        thisp1.dump_pdb(termini+'.pdb')
    
      # start offset rotations so the closest distance helix term is at/near the center of the samples so the starting helix is to the left and right of the target termini
      for j in range(int(rotation_samples/2)+1):
        offsetspinmoverrev.apply(thisp1)
      for j in range(rotation_samples):
        offsetspinmover.apply(thisp1)
        for direction in range(2): # 0 forward, 1 reverse
          outp = Pose()
          outp.assign(input_p)
              
          append_chain_to_pose(outp,thisp1,1,True)
          # add the rest of the helices
          p1r = Pose()
          p1r.assign(thisp1)
          for i in range(1,n):
            if direction == 0:
              spinmover.apply(p1r)
            else:
              spinmoverrev.apply(p1r)
            # flip for antiparallel
            flipmover = protocols.rigid.RigidBodyDeterministicSpinMover(1,pax,core.pose.get_center_of_mass(p1r),180)
            flipmover.apply(p1r)
            append_chain_to_pose(outp,p1r,1,True)
   
          #outp.dump_pdb(f'{input_pdb_name}_n{n}_nres{nres}_r{radius+rbuffer:.2f}_rot{rotation}_wrap_{termini}_direction{direction}_prewrap.pdb')
          prewraps.append([rotation,outp,direction])
  
        for p in prewraps:
          outprefix = f'{input_pdb_name}_WRAP_helix_n{n}_nres{nres}_tradius{radius:.0f}_rot{p[0]}_wrap_{termini}_direction{p[2]}'
  
          if os.path.exists(f'{outprefix}.pdb'):
            continue
  
          horder = []
          for i in range(1,n+1):
            horder.append(i)
          if isN:
            horder.reverse()
          
          print(f'{outprefix} horder: {horder}')
  
          pwrap = Pose() # final wrap 
  
          # add wrap helices
          for i in horder:
  
            ptoappend = Pose()
            append_chain_to_pose(ptoappend,p[1],1+i,True)
            pyrosetta.rosetta.core.pose.addVirtualResAsRoot(ptoappend)
            if h_rot_angle != 0:
              flipmover = protocols.rigid.RigidBodyDeterministicSpinMover(1,pax,core.pose.get_center_of_mass(ptoappend),h_rot_angle)
              flipmover.apply(ptoappend)
            append_chain_to_pose(pwrap,ptoappend,1,False)
  
          for res in pyrosetta.rosetta.core.pose.get_chain_residues(pwrap,1):
            core.conformation.remove_upper_terminus_type_from_conformation_residue(pwrap.conformation(), res.seqpos())
            core.conformation.remove_lower_terminus_type_from_conformation_residue(pwrap.conformation(), res.seqpos())
    
          # output wrap without loops
          pretmp = Pose()
          pretmp.assign(pwrap)
          #pretmp.dump_pdb(f'{outprefix}_wraponly.pdb')
          append_chain_to_pose(pretmp,input_p,1,True)
          #pretmp.dump_pdb(f'{outprefix}_noloops.pdb')
      
          # close loops?
          # this may hang... ughh
          if close_loops:
      
            print(f'Closing loops... if this hangs for more than a minute ctrl-z and kill the process w/ (kill -9 {os.getpid()}) and try different radius and/or looplen aggghhh..')
            pwrap, skip = closeloops(pwrap)
            if skip: continue

            wraplen = len(pwrap.sequence())
            closed_contig = f'{wraplen}-{wraplen}\\,0 B{wraplen+1}-{input_p_len+wraplen}'
            print(f'{outprefix} Closed contig: {closed_contig}') 
            append_chain_to_pose(pwrap,input_p,1,True)
            pwrap.dump_pdb(f'{outprefix}.pdb')
      
        rotation += rotation_sample_angle
      
  ## Generate partial diffusion tasks

  for i in glob.glob(f'{input_pdb_name}_WRAP_*.pdb'):
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
    cmd += f'inference.input_pdb={i} contigmap.contigs=[\\\'{contigstr}\\\'] inference.num_designs={partial_diffusions} denoiser.noise_scale_ca=0.5 denoiser.noise_scale_frame=0.5 diffuser.partial_T={partial_diffusion_partialT}'
    
    pdtf.write(cmd+'\n')
pdtf.close()  
  
  
  
  
  
  
  
  
  
  
  
  
