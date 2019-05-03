# hic-eventgen

Stripped down version of the hic-eventgen code, used only to generate an EOS

# Usage

python3 -m venv --system-site-packages --without-pip .
source bin/activate
bash local/install  #To install the `frzout` package

cd models

## For production

python3 eos.py --res-width-off --species=all --Tmax 1.0 --write-bin eos_urqmd.bin

## Just to see the output

python3 eos.py --res-width-off --species=all --Tmax 1.0
