# Getting Started
Docker environment setup is not provided separately.
Here, we provide instructions for setting up a anconda environment.


## Anaconda

### 0. Preliminary
We assume that both the conda environment and Python, as well as the PyTorch-related libraries, are already installed.
* We recommend PyTorch 1.13 and above including PyTorch 2.0.
* We recommend Python 3.8 and above.

```bash
# torch install example
# please refer to https://pytorch.org/get-started/previous-versions/
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 1. PyTorch Geometric Library
#### 1.1. Install PyTorch Geometric
In this tutorial, you need to install the PyTorch Geometric library for batch training implementation, model construction, and data downloading:
```bash
# We recommend version 2.4, as version 2.5 has a bug in data downloading.
pip3 install torch_geometric==2.4.0
```

#### 1.2. Install PyTorch Geometric Dependencies
After installing PyTorch Geometric library, you need to check the torch and cuda versions.
```bash
# Check cuda version
nvcc -V
# e.g. 11.8

# Check torch version
python3 -c "import torch; print(torch.__version__)"
# e.g. 2.2.0
```
Then, you have to install dependencies of the PyTorch Geometric library according the above versions:
```bash
# e.g. cuda: 11.8, torch: 2.2.0 case
pip3 install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.2.0+cu118.html

# e.g. cuda: 12.1, torch: 2.3.0 case
pip3 install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.3.0+cu121.html
```
For more details, please refer to the [PyTorch Geometric Installation guides](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html).


### 2. Package Installation

We install packages using the following command:
```bash
pip3 install -r requirements.txt
```