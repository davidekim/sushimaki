# sushimaki

![header.png](./header.png)

## Description
This repo includes utility scripts to generate helical or beta barrel WRAPs parametrically around a target protein.

## Reference
Ljubica Mihaljević et. al. Solubilization of Membrane Proteins using designed protein WRAPS. Submitted to Science.

## Installation
You can clone this repo into a preferred destination directory by going to that directory and then running:

`git clone https://github.com/davidekim/sushimaki.git`

### For a complete pipeline to generate backbone refined, sequence designed and AF2 validated WRAPs
Install the ppi_iterative_opt submodule for RF partial diffusion, ProteinMPNN + Rosetta FastRelax, and Alphafold2 protein-protein interaction design optimization.

~~~
cd sushimaki
git submodule init
git submodule update --remote
~~~

Complete the ppi_iterative_opt installation by following instructions at https://github.com/davidekim/ppi_iterative_opt and the websites of the dependencies.


### Dependencies

#### To generate input WRAPs for backbone refinement and sequence design

PyRosetta https://www.pyrosetta.org

BBQ https://biocomp.chem.uw.edu.pl/tools/bbq

DeepTMHMM https://dtu.biolib.com/DeepTMHMM

#### For backbone refinement, sequence design, and AF2 validation

RFDiffusion https://github.com/RosettaCommons/RFdiffusion

Protein MPNN https://github.com/dauparas/ProteinMPNN

Alphafold2 https://github.com/google-deepmind/alphafold


## Usage
sushimaki.py is the main script that generates inputs for RF partial diffusion structure refinement and Protein MPNN sequence design to generate WRAPs.

For helical input WRAPs
~~~
python ./sushimaki.py 2ge4A.pdb
~~~

For beta barrel input WRAPs
~~~
python ./sushimaki.py --barrel 2ge4A.pdb
~~~

For RF partial diffusion backbone refinement, ProteinMPNN sequence design, and Alphafold2 validation
~~~
python ./ppi_iterative_opt/ppi_iterative_opt.py *_WRAP_*pdb
~~~


## Support
Contact David Kim (dekim@uw.edu) for any questions.

## Authors and acknowledgment
This work was conceptualized and developed by David Kim (dekim@uw.edu) and Ljubica Mihaljevic (ljubim@uw.edu)

