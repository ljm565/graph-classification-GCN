# Data Preparation
여기서는 기본적으로 [TUDataset](https://chrsmrrs.github.io/datasets/) 중 COLLAB 데이터셋을 활용하여 GCN 모델 학습 튜토리얼을 진행합니다.
Custom 데이터를 이용하기 위해서는 아래 설명을 참고하시기 바랍니다.

### 1. COLLAB
COLLAB 데이터를 학습하고싶다면 아래처럼 `config/config.yaml`의 `collab_dataset_train`을 `True` 설정하면 됩니다.
```yaml
collab_dataset_train: True             # if True, collab dataset will be loaded automatically.
collab_dataset:
    path: data/
CUSTOM:
    train_data_path: null
    validation_data_path: null
    test_data_path: null
```
<br>

### 2. Custom Data
만약 custom 데이터를 학습하고 싶다면 아래처럼 `config/config.yaml`의 `collab_dataset_train`을 `False`로 설정하면 됩니다.
Custom data 사용을 위해 train/validation/test 데이터셋 경로를 입력해주어야 합니다.
```yaml
collab_dataset_train: False            # if True, collab dataset will be loaded automatically.
collab_dataset:
    path: data/
CUSTOM:
    train_data_path: null
    validation_data_path: null
    te