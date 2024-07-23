# Trained Model Feature Visualization
Here, we provide guides for visualizing the trained GCN model features.

### 1. Visualization
#### 1.1 Arguments
There are several arguments for running `src/run/vis_tsne.py`:
* [`-r`, `--resume_model_dir`]: Directory to the model to visualize features. Provide the path up to `${project}/${name}`, and it will automatically select the model from `${project}/${name}/weights/` to visualize.
* [`-l`, `--load_model_type`]: Choose one of [`loss`, `last`].
    * `metric` (default): Resume the model with the best validation set's accuracy.
    * `loss`: Resume the model with the minimum validation loss.
    * `last`: Resume the model saved at the last epoch.
* [`-d`, `--dataset_type`]: (default: `test`) Choose one of [`train`, `validation`, `test`].


#### 1.2 Command
`src/run/vis_tsne.py` file is used to visualize output features of the trained GCN model.
```bash
python3 src/run/vis_tsne.py --resume_model_dir ${project}/${name}
```