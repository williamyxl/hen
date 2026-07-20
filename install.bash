conda create -n hen312 python=3.12 -y

conda activate hen312

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

pip install ase pymatgen

