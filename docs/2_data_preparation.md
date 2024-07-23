# Data Preparation
Here, we will proceed with a GCN model training tutorial using COLLAB dataset among the [TUDataset](https://chrsmrrs.github.io/datasets/).
Please refer to the following instructions to utilize custom datasets.


### 1. COLLAB
If you want to train on the Cora dataset, simply set the `collab_dataset_train` value in the `config/config.yaml` file to `True` as follows.
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
If you want to train on the custom dataset, simply set the `collab_dataset_train` value in the `config/config.yaml` file to `False` as follows.
You have to set your custom training/validation/test datasets.
```yaml
collab_dataset_train: False            # if True, collab dataset will be loaded automatically.
collab_dataset:
    path: data/
CUSTOM:
    train_data_path: null
    validation_data_path: null
    test_data_path: null
```
<br>