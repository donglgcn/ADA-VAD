## ADA-VAD
Offcial implementation of "Ada-VAD: Domain Adaptable Video Anomaly Detection, SDM-2024".

[\[Paper\]](https://epubs.siam.org/doi/pdf/10.1137/1.9781611978032.73)

<!-- ![pipeline](./assets/fig2-CR.png) -->
## 1. Dependencies
```
conda env create -f VAD.yaml
conda activate VAD
```
## 2. Usage
### 2.1 Data preparation
Please follow the [instructions](./pre_process/readme.md) to prepare the training and testing dataset.

### 2.2 Train
We train the **Pretraining Stage** mode at first, then train **Adaptaion Stage** model. All the config files are located at `./cfgs`. 

To train the **Pretraining Stage** model, run:
```python
$ python unetonlyInfomaxAnypredictStage1.py -f .cfgs/unetonlyAnypredictStage1/***
```
To train the **Adaptaion Stage** model with 1-shot, run:
```python
$ python unetonlyInfomaxAnypredictStage2.py -f .cfgs/unetonlyAnypredictStage2/***
```
And to train the **Adaptaion Stage** model with 5-shot, run:
```python
$ python unetonlyInfomaxAnypredictStage2.py -f .cfgs/unetonlyAnypredictStage2_5shot/***
```
For different datasets, please modify the configuration files accordingly.

### 2.3 Evaluation
Each training process will be stored by tensorboard. You can still evaluation the anomaly detection performance of the trained model, run:
```python
$ python evaluation_Anypredictstage1.py [--model_save_path] [--cfg_file] 

$ python evaluation_Anypredictstage2.py [--model_save_path] [--cfg_file] 
```

## Acknowledgment
We thank LiUzHiAn for the PyTorch implementation of the [hf2vad](https://github.com/LiUzHiAn/hf2vad).

## Citation
If you find this repo useful, please consider citing:
```
@inbook{doi:10.1137/1.9781611978032.73,
author = {Dongliang Guo and Yun Fu and Sheng Li},
title = {Ada-VAD: Domain Adaptable Video Anomaly Detection},
booktitle = {Proceedings of the 2024 SIAM International Conference on Data Mining (SDM)},
chapter = {},
pages = {634-642},
doi = {10.1137/1.9781611978032.73},
URL = {https://epubs.siam.org/doi/abs/10.1137/1.9781611978032.73},
eprint = {https://epubs.siam.org/doi/pdf/10.1137/1.9781611978032.73}
}
```
