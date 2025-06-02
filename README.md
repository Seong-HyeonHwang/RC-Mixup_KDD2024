# RC-Mixup: A Data Augmentation Strategy against Noisy Data for Regression Tasks
This repository contains the official implementation of our paper accepted at KDD 2024.
# Abstract
This work presents RC-Mixup, a novel data augmentation technique specifically designed to improve regression model performance in the presence of noisy data. Our method combines robust regression principles with mixup augmentation to create synthetic training samples that enhance model generalization.

# Quick Start
Please refer to the train.py for implementing main algorithm, and demo.ipynb for running the codes.

```
from model import *
from utils import *
from train import *
import numpy as np

# Hyperparameters

label_dim = 4
noise_ratio = 0.3
device = torch.device('cuda')
seed = 410
warmup_epochs = 1000
update_epochs = 1000
lr_model = 1e-2
batch_size = 128
alpha = 2
L = 500
N = 500
bw_list = [5, 10, 15, 20]
start_bandwidth = 20

# Preprocessing
raw_data = np.load("Synthetic_dataset.npy")

dataset = {}
dataset[seed] = {}
train_data, valid_data, test_data, Y_max, Y_min, noise_idx = preprocess_dataset(seed, raw_data, noise_ratio)
x_tr, y_tr = train_data[:, :-label_dim], train_data[:, -label_dim:]
x_val, y_val = valid_data[:, :-label_dim], valid_data[:, -label_dim:]
x_test, y_test = test_data[:, :-label_dim], test_data[:, -label_dim:]

dataset[seed]['x_tr'], dataset[seed]['y_tr'] = x_tr, y_tr
dataset[seed]['x_val'], dataset[seed]['y_val'] = x_val, y_val
dataset[seed]['x_test'], dataset[seed]['y_test'] = x_test, y_test
dataset[seed]['Y_max'], dataset[seed]['Y_min'] = Y_max, Y_min
dataset[seed]['noise_idx'] = noise_idx

# run RC-Mixup algorithm
updated_model = rc_mixup(dataset, noise_ratio, seed, lr_model, alpha, warmup_epochs, update_epochs,
             L, N, bw_list, start_bandwidth, batch_size, device = device)
```

# Citation
If you find this work useful, please cite our paper:
```bibtex
@inproceedings{hwang2024rc,
  title={RC-Mixup: A Data Augmentation Strategy against Noisy Data for Regression Tasks},
  author={Hwang, Seong-Hyeon and Kim, Minsu and Whang, Steven Euijong},
  booktitle={Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
  pages={1155--1165},
  year={2024}
}
```
