import os,sys
import numpy as np
import argparse

import biolib
# https://dtu.biolib.com/DeepTMHMM
# pip3 install pybiolib

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

parser.add_argument('--n', type=int, default=0, help='Manually set n (strand count).')
parser.add_argument('--nres', type=int, default=0, help='Manually set nres (strand length).')
parser.add_argument('--shear', type=int, default=0, help='Manually set shear (barrel shear).')
parser.add_argument('--radius', type=float, default=0, help='Manually set radius.')
parser.add_argument('--rot_angle', type=int, default=30, help='Manually set rotation sample angle.')
parser.add_argument('--rot_n', type=int, default=3, help='Manually set rotation samples.')
parser.add_argument('--add_loops', type=bool, default=True, help='Add loops.')
parser.add_argument('--looplen', type=int, default=3, help='Loop length.')
parser.add_argument('--debug', type=bool, default=False, help='Debug.')
parser.add_argument('--verbose', type=bool, default=False, help='Verbose output.')

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
rotation_samples = args['rot_n']
manual_n = args['n']
manual_nres = args['nres']
manual_shear = args['shear']
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

def append_chain_to_pose(p1,p2,chain=1,new_chain=True):
  jumpadded = False
  for res in pyrosetta.rosetta.core.pose.get_chain_residues(p2,chain):
    if not jumpadded:
      p1.append_residue_by_jump(res, 1, "", "", new_chain)
      jumpadded = True
    else:
      p1.append_residue_by_bond(res)
  return p1

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

EXT_PHI = -150.0
EXT_PSI = +150.0
EXT_OMG = 180.0

def add_loops(pose,nstrands,nresstrand,looplen):
  # add loops to cylinder
  print("Adding loops...")
  p = Pose()
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
    loop = protocols.loops.Loop(previous, cutpoint+1, cutpoint, 0.0, True)
    loops.add_loop(loop)
    protocols.loops.set_single_loop_fold_tree(p, loop)
    # extend loops before closing
    for l in range(loop.start()+1,loop.stop()):
      core.conformation.idealize_position(l, p.conformation())
      p.set_phi(l, EXT_PHI)
      p.set_psi(l, EXT_PSI)
      p.set_omega(l, EXT_OMG)
  return p, loops


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


  # create parametric cylinders using parameters to sample the radius and height










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





for c in cylinders:
  
  # strand len
  nres = int(c.split('_nres')[1].split('.0_b')[0])
  # strands
  n = int(c.split('_n')[1].split('_S')[0])

  print(f'Working on {c} with strand length {nres} and {n} strands...')

  finalpdb = c.split('.pdb')[0]+'_'+gpcr_pdb_name+'.pdb'
  if os.path.exists(finalpdb):
    print(f'{finalpdb} already exists.. skipping')
    continue

  # cylinder pose
  cylinder_p = pose_from_file(c)
  cylinder_p_len = len(cylinder_p.sequence())
  cylinder_p.center()

  # get top and bottom coords
  ctop = []
  cbottom = []
  ctopcoords = vector_numeric_xyzVector_double_t()
  cbottomcoords = vector_numeric_xyzVector_double_t()
  for i in range(1,n+1):
    clen = i*nres
    if not i%2:
      for add in range(1,blen+1):
        ctop.append(clen-nres+add)
      for sub in range(1,blen+1):
        cbottom.append(clen-sub+1)
    else: 
      for add in range(1,blen+1):
        cbottom.append(clen-nres+add)
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
  p2.assign(gpcr_p_trans)
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

  # if even stranded, make a flipped cylinder for N-term gpcr fusion to C-term of cylinder
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

  # append transmembrane part of gpcr as a new chain
  core.pose.append_pose_to_pose(cylinder_p, p2)
  core.pose.append_pose_to_pose(cylinder_p_flipped, p2_flipped)

  # At this point the gpcr transmembrane portion should be placed well in the cylinder and flipped cylinder

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

  ## place full gpcr onto minimized pose
  gpcr_p_copy = Pose()
  gpcr_p_copy.assign(gpcr_p)
  ca_map = map_core_id_AtomID_core_id_AtomID()
  ti = 0
  for r1 in range(cylinder_p_len+1,len(cylinder_p.sequence())+1):
    ca_map[AtomID(gpcr_p_copy.residue(trans[ti]).atom_index("CA"), trans[ti])] = AtomID(cylinder_p.residue(r1).atom_index("CA"), r1)
    ti += 1
  rmsd = pyrosetta.rosetta.core.scoring.superimpose_pose(gpcr_p_copy, cylinder_p, ca_map)
  print(f'gpcr placement rmsd: {rmsd}')
  gpcr_p_copy_flipped = Pose()
  gpcr_p_copy_flipped.assign(gpcr_p)
  ca_map = map_core_id_AtomID_core_id_AtomID()
  ti = 0
  for r1 in range(cylinder_p_len+1,len(cylinder_p_flipped.sequence())+1):
    ca_map[AtomID(gpcr_p_copy_flipped.residue(trans[ti]).atom_index("CA"), trans[ti])] = AtomID(cylinder_p_flipped.residue(r1).atom_index("CA"), r1)
    ti += 1
  rmsd = pyrosetta.rosetta.core.scoring.superimpose_pose(gpcr_p_copy_flipped, cylinder_p_flipped, ca_map)
  print(f'gpcr flipped placement rmsd: {rmsd}')
 
  # add loops to cylinder
  p = Pose()
  p, loops = add_loops(cylinder_p,n,nres,looplen)
  # add gpcr back
  p = append_chain_to_pose(p,gpcr_p_copy)

  # add loops to flipped cylinder
  p_flipped = Pose()
  p_flipped, loops_flipped = add_loops(cylinder_p_flipped,n,nres,looplen)
  # add gpcr back
  p_flipped = append_chain_to_pose(p_flipped,gpcr_p_copy_flipped)

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
    perturb_bb(p, 1)
    core.conformation.idealize_position(len(core.pose.get_chain_residues(p,1)), p.conformation())
    perturb_bb(p, len(core.pose.get_chain_residues(p,1)))
    p.append_polymer_residue_after_seqpos(residue, len(core.pose.get_chain_residues(p,1)), True)
    perturb_bb(p, len(core.pose.get_chain_residues(p,1)))

    core.conformation.idealize_position(1, p_flipped.conformation())
    p_flipped.prepend_polymer_residue_before_seqpos(residue, 1, True)
    core.conformation.idealize_position(1, p_flipped.conformation())
    perturb_bb(p_flipped, 1)
    core.conformation.idealize_position(len(core.pose.get_chain_residues(p_flipped,1)), p_flipped.conformation())
    perturb_bb(p_flipped, len(core.pose.get_chain_residues(p_flipped,1)))
    p_flipped.append_polymer_residue_after_seqpos(residue, len(core.pose.get_chain_residues(p_flipped,1)), True)
    perturb_bb(p_flipped, len(core.pose.get_chain_residues(p_flipped,1)))

  # superimpose extended termini barrel
  ca_map = map_core_id_AtomID_core_id_AtomID()
  for r in range(1,len(core.pose.get_chain_residues(p_orig,1))+1):
    ca_map[AtomID(p_orig.residue(r).atom_index("CA"), r)] = AtomID(p.residue(r+termlen).atom_index("CA"), r+termlen)
  rmsd = pyrosetta.rosetta.core.scoring.superimpose_pose(p_orig, p, ca_map)
  print(f'extended termini barrel placement rmsd: {rmsd}')

  ca_map = map_core_id_AtomID_core_id_AtomID()
  for r in range(1,len(core.pose.get_chain_residues(p_orig_flipped,1))+1):
    ca_map[AtomID(p_orig_flipped.residue(r).atom_index("CA"), r)] = AtomID(p_flipped.residue(r+termlen).atom_index("CA"), r+termlen)
  rmsd = pyrosetta.rosetta.core.scoring.superimpose_pose(p_orig_flipped, p_flipped, ca_map)
  print(f'extended termini barrel flipped placement rmsd: {rmsd}')
 
  finalp = append_chain_to_pose( p_orig.split_by_chain(2), p, 1 )
  finalp_flipped = append_chain_to_pose( p_orig_flipped.split_by_chain(2), p_flipped, 1 )

  # replace gpcr with orig coords (since full atom was lost for loop closure and termini extension)
  ca_map = map_core_id_AtomID_core_id_AtomID()
  for r in range(1,len(core.pose.get_chain_residues(gpcr_p_copy,1))+1):
    ca_map[AtomID(finalp.residue(r).atom_index("CA"), r)] = AtomID(gpcr_p_copy.residue(r).atom_index("CA"), r)
  rmsd = pyrosetta.rosetta.core.scoring.superimpose_pose(finalp, gpcr_p_copy, ca_map)
  print(f'gpcr placement rmsd: {rmsd}')

  ca_map = map_core_id_AtomID_core_id_AtomID()
  for r in range(1,len(core.pose.get_chain_residues(gpcr_p_copy_flipped,1))+1):
    ca_map[AtomID(finalp_flipped.residue(r).atom_index("CA"), r)] = AtomID(gpcr_p_copy_flipped.residue(r).atom_index("CA"), r)
  rmsd = pyrosetta.rosetta.core.scoring.superimpose_pose(finalp_flipped, gpcr_p_copy_flipped, ca_map)
  print(f'gpcr flipped placement rmsd: {rmsd}')

  finalp_complex = Pose()
  finalp_complex.assign(finalp)
  finalp_flipped_complex = Pose()
  finalp_flipped_complex.assign(finalp_flipped)
  finalp = append_chain_to_pose( Pose(gpcr_p_copy), finalp, 2, False )
  finalp_flipped = append_chain_to_pose( finalp_flipped.split_by_chain(2), Pose(gpcr_p_copy_flipped), 1, False )
  finalp_complex = append_chain_to_pose( finalp_complex.split_by_chain(2), Pose(gpcr_p_copy), 1, True )
  finalp_flipped_complex = append_chain_to_pose( finalp_flipped_complex.split_by_chain(2), Pose(gpcr_p_copy_flipped), 1, True )

  finalp.dump_pdb(finalpdb)
  finalp_flipped.dump_pdb(c.split('.pdb')[0]+'_'+gpcr_pdb_name+'_flipped.pdb')
  finalp_complex.dump_pdb(c.split('.pdb')[0]+'_'+gpcr_pdb_name+'_complex.pdb')
  finalp_flipped_complex.dump_pdb(c.split('.pdb')[0]+'_'+gpcr_pdb_name+'_flipped_complex.pdb')

  # generate rotation variants
  save_rotations( finalp_complex, n, nres, c.split('.pdb')[0]+'_'+gpcr_pdb_name+'_complex', termlen, looplen, rotation_angle )
  save_rotations( finalp_flipped_complex, n, nres, c.split('.pdb')[0]+'_'+gpcr_pdb_name+'_flipped_complex', termlen, looplen, rotation_angle ) 


  print(f'Created {finalpdb}')
  print("Done!")
  exit()





