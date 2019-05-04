# hic-eventgen

Stripped down version of the hic-eventgen code, used only to generate an EOS

To install: 
git clone https://github.com/j-f-paquet/eos_maker.git --recursive
cd eos_maker
git checkout smash --recurse-submodules

# Usage

python3 -m venv --system-site-packages --without-pip .
source bin/activate
bash local/install  #To install the `frzout` package

cd models

## For production

python3 eos.py --res-width-off --species=all --Tmax 1.0 --write-bin eos_smash.bin --music_output_format

## Just to see the output

python3 eos.py --res-width-off --species=all --Tmax 1.0
