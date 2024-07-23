# Getting Started
여기서는 도커 환경 구성에 대한 방법을 제공하지 않습니다.
여기서는 Anaconda 환경을 구성하는 방법을 제공합니다.


## Anaconda
### 0. Preliminary
Python, conda 환경, PyTorch 관련 라이브러리가 모두 설치가 되어있음을 가정합니다.
* PyTorch 2.0을 포함한 PyTorch 1.13 이상 버전을 권장.
* Python 3.8 이상 버전을 권장.

```bash
# torch install example
# please refer to https://pytorch.org/get-started/previous-versions/
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 1. PyTorch Geometric Library
#### 1.1. Install PyTorch Geometric
본 튜토리얼에서는 batch 학습 구현, 모델 제작, 데이터 다운로드를 위해 PyTorch Geometric 라이브러리가 사용됩니다.
```bash
# We recommend version 2.4, as version 2.5 has a bug in data downloading.
pip3 install torch_geometric==2.4.0
```

#### 1.2. Install PyTorch Geometric Dependencies
PyTorch Geometric 라이브러리 설치 후, cuda와 torch 버전을 확인해야합니다.
```bash
# Check cuda version
nvcc -V
# e.g. 11.8

# Check torch version
python3 -c "import torch; print(torch.__version__)"
# e.g. 2.2.0
```
그후, 위의 버전에 따라서 dependency들을 설치합니다.
```bash
# e.g. cuda: 11.8, torch: 2.2.0 case
pip3 install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.2.0+cu118.html

# e.g. cuda: 12.1, torch: 2.3.0 case
pip3 install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.3.0+cu121.html
```
더 자세한 설명은 [PyTorch Geometric Installation guides](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)를 참고하시기 바랍니다.


### 2. Package Installation
다음과 같은 명령어로 pacakge를 설치합니다.
```bash
pip3 install -r requirements.txt
```