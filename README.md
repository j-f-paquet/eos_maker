# hic-eventgen

Stripped down version of the hic-eventgen code, used only to generate an EOS

# Installation
```
git clone --recursive --recurse-submodules --branch urqmd https://github.com/j-f-paquet/eos_maker.git eos_maker_urqmd

cd eos_maker_urqmd

python3 -m venv --system-site-packages --without-pip .
source bin/activate

bash local/install  # To install the `frzout` package
```

# Usage

## For production

```
python3 eos.py --res-width-off --species=urqmd --Tmax 1.0 --write-bin eos_urqmd.bin --music_output_format
```

## Just to see the output

```
python3 eos.py --res-width-off --species=urqmd --Tmax 1.0
```
