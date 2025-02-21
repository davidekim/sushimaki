import os,sys
import numpy as np
import argparse

import biolib
# https://dtu.biolib.com/DeepTMHMM
# pip3 install pybiolib


rbuffer = 6 # to add to transmembrane radius calculation to determine number of helices
hbuffer = 5 # to add to tmheight
bulkrbuffer = 4 # to add to transmembrane radius calculation to determine number of bulk helices

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

parser.add_argument('--n', type=int, default=0, help='Manually set n (helix count).')
parser.add_argument('--nres', type=int, default=0, help='Manually set nres (helix length).')
parser.add_argument('--radius', type=float, default=0, help='Manually set radius.')
parser.add_argument('--rot_angle', type=int, default=30, help='Manually set rotation sample angle.')
parser.add_argument('--h_rot_angle', type=float, default=0, help='Rotate angle of helices in wrap relative to target.')
parser.add_argument('--rot_n', type=int, default=3, help='Manually set rotation samples.')
parser.add_argument('--add_loops', type=bool, default=True, help='Add loops.')
parser.add_argument('--looplen', type=int, default=3, help='Loop length.')
parser.add_argument('--debug', type=bool, default=False, help='Debug.')
parser.add_argument('--verbose', type=bool, default=False, help='Verbose output.')
parser.add_argument('--add_bulk_helices_n_per_layer', type=int, default=0, help='Add n helices per layer at non connecting terminus. To increase the size of the wrap to facilitate cryo-em.')
parser.add_argument('--add_bulk_helices_n_layers', type=int, default=1, help='Add n layers of bulk helices.')

parser.add_argument(
        '--top_residues_to_wrap',
        help='top residue numbers separated by spaces to wrap. These along with the bottom residues will be used to determine axis to align against',
        action='store',
        nargs='+'
        )

parser.add_argument(
        '--bottom_residues_to_wrap',
        help='bottom residue numbers separated by spaces to wrap. These along with the top residues will be used to determine axis to align against',
        action='store',
        nargs='+'
        )

parser.add_argument(
        '--all_residues_to_wrap',
        help='Residue numbers separated by spaces to wrap. The default is to use DeepTMHMM to identify transmembrane residues to wrap.',
        action='store',
        nargs='+'
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
bulk_helices_n_per_layer = args['add_bulk_helices_n_per_layer']
bulk_helices_n_layers = args['add_bulk_helices_n_layers']
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



blen = 1 # len for inner and outer coords for superposition into cylinder

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

def add_loops(pose,Nposs,Cposs):
  # add loop
  p = Pose()
  p.assign(pose)
  pstartlen = len(p.sequence())
  loops = protocols.loops.Loops()
  added = 0
  for i,Npos in enumerate(Nposs):
    Npos = Npos+added
    Cpos = Cposs[i] + added
    if verbose:
      print(f'Adding {looplen} residue loop connecting {Npos} to {Cpos}')
    #core.conformation.remove_upper_terminus_type_from_conformation_residue(p.conformation(), Npos)
    #core.conformation.remove_lower_terminus_type_from_conformation_residue(p.conformation(), Cpos)
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
    loop = protocols.loops.Loop(Cpos, Cpos+looplen+1, cutpoint, 0.0, True)
    loops.add_loop(loop)

    protocols.loops.set_single_loop_fold_tree(p, loop)

    added = len(p.sequence())-pstartlen
    if verbose:
      print(f'Added loop: {Cpos} {Cpos+looplen+1} {cutpoint}')
  return p, loops

def closeloops(pose):
  Nposs = []
  Cposs = []
  thisn = n
  if bulk_helices_n_per_layer > 0:
    thisn += bulk_helices_n_per_layer*bulk_helices_n_layers 

  for i in range(1,thisn):
    Cposs.append(i*nres)
    Nposs.append(i*nres+1)
  pose, loops = add_loops(pose,Nposs,Cposs)
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
  return pose



## MAIN


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
  
  # Note: 27-32 angstroms typical for lipid bilayer height
  print(f'Radius: {radius} TM height: {tmheight}')
  print(f'Looplen: {looplen}')
  
  # determine number of helices based on radius
  n = number_of_helices(radius)
  odd_helix_n = n%2
  print(f'Helices: {n}')
  
  # determine how long the helices should be
  nres = helix_length(tmheight)
  print(f'Helix length: {nres}')
  
  # make helix
  p_h = generate_helix(nres)
 
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
 
        outp.dump_pdb(f'{input_pdb_name}_n{n}_nres{nres}_r{radius+rbuffer:.2f}_rot{rotation}_wrap_{termini}_direction{direction}_prewrap.pdb')
        prewraps.append([rotation,outp,direction])

      for p in prewraps:
        outprefix = f'{input_pdb_name}_n{n}_nres{nres}_r{radius+rbuffer:.2f}_rot{p[0]}_wrap_{termini}_direction{p[2]}'
        if bulk_helices_n_per_layer > 0:
          outprefix += f'_bulkh_{bulk_helices_n_layers}_{bulk_helices_n_per_layer}'

        if os.path.exists(f'{outprefix}_closed.pdb'):
          continue

        # pose with bulk helices
        bwrap = Pose()
        # add bulk helices to N-term? (target at C-term)
        if bulk_helices_n_per_layer > 0:


          # create initial bulk helix
          bh1 = Pose()
          append_chain_to_pose(bh1,p[1],n+1,True)

          # create each bulk layer
          for i in range(1, bulk_helices_n_layers+1):
            # max number of helices in this layer
            hn = number_of_helices(radius+(i*2*bulkrbuffer))

            # translate out the initial helix (put into this layer)
            pyrosetta.rosetta.core.pose.addVirtualResAsRoot(bh1)
            transmover = protocols.rigid.RigidBodyTransMover(bh1,1)
            bh1com = core.pose.get_center_of_mass(bh1)
            hpaxis = bh1com-p2com
            transmover.trans_axis(hpaxis)
            transmover.step_size(bulkrbuffer*2)
            transmover.apply(bh1)

            # create this layer of helices
            for k in range(1, bulk_helices_n_per_layer+1):
              bh1com = core.pose.get_center_of_mass(bh1)
              hpaxis = bh1com-p2com
              # flip helix for antiparallel
              flipmover = protocols.rigid.RigidBodyDeterministicSpinMover(1,hpaxis,core.pose.get_center_of_mass(bh1),180)
              #flipmover = protocols.rigid.RigidBodyDeterministicSpinMover(1,pax,core.pose.get_center_of_mass(bh1),180)
              flipmover.apply(bh1)
              # append helix to bulk pose

              ptoappend = Pose()
              ptoappend.assign(bh1)

              if h_rot_angle != 0:
                flipmover = protocols.rigid.RigidBodyDeterministicSpinMover(1,pax,core.pose.get_center_of_mass(ptoappend),h_rot_angle)
                flipmover.apply(ptoappend)

              append_chain_to_pose(bwrap,ptoappend,1,True)

              # shift helix over within layer (either forward or reverse depending on layer - direction alternating for each layer)
              lspinmover = protocols.rigid.RigidBodyDeterministicSpinMover(1,ax,p2com,360/hn)
              lspinmoverrev = protocols.rigid.RigidBodyDeterministicSpinMover(1,ax,p2com,-1*360/hn)
              if k < bulk_helices_n_per_layer:
                if p[2] == 0: # forward direction
                  if i%2 == 0:
                    lspinmover.apply(bh1)
                  else:
                    lspinmoverrev.apply(bh1)
                else:
                  if i%2 == 0:
                    lspinmoverrev.apply(bh1)
                  else:
                    lspinmover.apply(bh1)

        horder = []
        for i in range(1,n+1):
          horder.append(i)
        if isN:
          horder.reverse()
        
        print(f'{outprefix} horder: {horder}')

        pwrap = Pose() # final wrap (includes bulk helices)

        # add N-term bulk helices?
        bhorder = []
        if bulk_helices_n_per_layer > 0:
          for i in range(1,(bulk_helices_n_layers*bulk_helices_n_per_layer)+1):
            bhorder.append(i)
        if isN:
          bhorder.reverse()
          for i in bhorder:
            append_chain_to_pose(pwrap,bwrap,i,False)

        # add wrap helices
        for i in horder:

          ptoappend = Pose()
          append_chain_to_pose(ptoappend,p[1],1+i,True)
          pyrosetta.rosetta.core.pose.addVirtualResAsRoot(ptoappend)
          if h_rot_angle != 0:
            flipmover = protocols.rigid.RigidBodyDeterministicSpinMover(1,pax,core.pose.get_center_of_mass(ptoappend),h_rot_angle)
            flipmover.apply(ptoappend)
          append_chain_to_pose(pwrap,ptoappend,1,False)
        # add C-term bulk helices?
        if bulk_helices_n_per_layer > 0 and not isN:
          for i in bhorder:
            append_chain_to_pose(pwrap,bwrap,i,False)

        for res in pyrosetta.rosetta.core.pose.get_chain_residues(pwrap,1):
          core.conformation.remove_upper_terminus_type_from_conformation_residue(pwrap.conformation(), res.seqpos())
          core.conformation.remove_lower_terminus_type_from_conformation_residue(pwrap.conformation(), res.seqpos())
  
        # output wrap without loops
        pretmp = Pose()
        pretmp.assign(pwrap)
        pretmp.dump_pdb(f'{outprefix}_wraponly.pdb')
        append_chain_to_pose(pretmp,input_p,1,True)
        pretmp.dump_pdb(f'{outprefix}.pdb')
    
        # close loops?
        # this may hang... ughh
        if close_loops:
    
          print(f'Closing loops... if this hangs for more than a minute ctrl-z and kill the process w/ (kill -9 {os.getpid()}) and try different radius and/or looplen aggghhh..')
          pwrap = closeloops(pwrap)
          wraplen = len(pwrap.sequence())
          closed_contig = f'{wraplen}-{wraplen}\\,0 B{wraplen+1}-{input_p_len+wraplen}'
          print(f'{outprefix} Closed contig: {closed_contig}') 
          append_chain_to_pose(pwrap,input_p,1,True)
          pwrap.dump_pdb(f'{outprefix}_closed.pdb')
    
      rotation += rotation_sample_angle
    
  
  














