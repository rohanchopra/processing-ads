
# Processing ADs
Promotion of products is now a common practice and is heavily controlled by broadcasting of advertisements. It is painstakingly difficult to personalize the content search by understanding the sentiments behind these image advertisements. It is proven that an image can be perceived in different manners and hence different emotions can be conveyed via them. This study tries to compare three backbone deep learning architectures on an advertisements dataset to potentioally classify the underlying sentiments interleaved in the images. 

## Directory Structure

```
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
```

## Creating the environment
Load conda environment as:
```
conda env create -f environment.yml
```
Install torch in conda environment:
```
#pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
```


