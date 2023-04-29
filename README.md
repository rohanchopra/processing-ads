# Processing ADs

## Directory Structure

├── documentation/              <- All project related documentation and reports
├── notebooks/                  <- Jupyter notebooks
│  ├── data_preprocessing/      <- Notebooks for cleaning the dataset
│  ├── image_preprocessing/     <- Notebooks for processing images and visualization
├── src/                        <- Source code for the project
│  ├── multilabel/              <- Scripts for the multilabel dataset
│  ├── __init__.py              <- Makes src a Python module
├── .gitignore                  <- List of files and folders git should ignore
├── LICENSE                     <- Project's License
├── README.md                   <- The top-level README for developers using this project
└── environment.yml             <- Conda environment file

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


