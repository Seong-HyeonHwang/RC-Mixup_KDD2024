# RC-Mixup: A Data Augmentation Strategy against Noisy Data for Regression Tasks
This repository contains the official implementation of our paper accepted at KDD 2024.
[![Paper](https://img.shields.io/badge/Paper-KDD%202024-blue)](https://dl.acm.org/doi/10.1145/3637528.3671234)

# Abstract
This work presents RC-Mixup, a novel data augmentation technique specifically designed to improve regression model performance in the presence of noisy data. Our method combines robust regression principles with mixup augmentation to create synthetic training samples that enhance model generalization.

# Quick Start

```bash
# Run with default parameters
python main.py

# Customize parameters
python main.py --noise_ratio 0.5 --alpha 1.5 --warmup_epochs 500 --batch_size 64
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
