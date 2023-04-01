# Processing ADs


## Creating the environment
Load conda environment as:
```
conda env create -f environment.yml
```
Install torch in conda environment:
```
#pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
```
