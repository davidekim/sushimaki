# sushimaki

![header.png](./header.png)

## Description
This repo includes a script that generates helical or beta barrel WRAPs parametrically around a target transmembrane protein.

## Reference
Solubilization of Membrane Proteins using designed protein WRAPS
Ljubica Mihaljević, David E. Kim, Helen E. Eisenach, Pooja D. Bandawane, Andrew J. Borst, Alexis Courbet, Everton Bettin, Qiushi Liu, Connor Weidle, Sagardip Majumder, Xinting Li, Mila Lamb, Analisa Nicole Azcárraga Murray, Rashmi Ravichandran, Elizabeth C. Williams, Shuyuan Hu, Lynda Stuart, Linda Grillová, Nicholas R. Thomson, Pengxiang Chang, Melissa J. Caimano, Kelly L. Hawley, Neil P. King, David Baker
bioRxiv 2025.02.04.636539; doi: https://doi.org/10.1101/2025.02.04.636539

## Installation
You can clone this repo into a preferred destination directory by going to that directory and then running:

`git clone https://github.com/davidekim/sushimaki.git`

## Usage
sushimaki.py is the main script that generates WRAPed inputs for RF partial diffusion structure refinement.

`python ../sushimaki.py --pdbs 2ge4A.pdb`

### Dependencies
PyRosetta [https://www.pyrosetta.org](https://www.pyrosetta.org)

BBQ [https://biocomp.chem.uw.edu.pl/tools/bbq](https://biocomp.chem.uw.edu.pl/tools/bbq)

DeepTMHMM [https://dtu.biolib.com/DeepTMHMM](https://dtu.biolib.com/DeepTMHMM)
pip3 install pybiolib


## Support
Contact David Kim (dekim@uw.edu) for any questions.


