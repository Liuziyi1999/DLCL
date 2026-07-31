# DLCL
When Vision-Language Models Meet Fetal Cardiac Ultrasound: Dual-Level Contrastive Learning for Out-of-Distribution Detection

In this work, we introduce Dual-Level Contrastive Learning (DLCL) for out-of-distribution detection in fetal cardiac ultrasound. DLCL leverages the semantic knowledge of vision-language models and performs contrastive learning at two complementary levels to improve the discrimination between in-distribution and out-of-distribution samples.

### Framework
![Framework](https://github.com/Liuziyi1999/DLCL/blob/main/figs/framework3.png)

## Overview
This repository is a PyTorch implementation of the paper.  

## How to Install
Our code is built based on the source code of [CoOp](https://github.com/KaiyangZhou/CoOp). So you need to install some dependent environments.
```# install clip
pip install ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git

# clone dapl
git clone https://github.com/LeapLabTHU/DAPrompt.git

# install dassl
git clone https://github.com/KaiyangZhou/Dassl.pytorch.git
cd dassl
pip install -r requirements.txt
pip install .
cd ..
```
You may follow the installation guide from [CLIP](https://github.com/KaiyangZhou/CoOp) and [dassl](https://github.com/KaiyangZhou/Dassl.pytorch).

## How to Run
We provide the running scripts in ```scripts/```. Make sure you change the path in ```DATA``` and run the commands under ```DLCL/scripts/```.

### Training

The training scripts are provided in the `scripts/` directory. Before training, please update the dataset path specified by `DATA` in the corresponding script or configuration file.

Run the following commands from the `DLCL/scripts/` directory:

```bash
cd DLCL/scripts
bash train_ivlp.sh CAMELYON17 epo20+10_v8_t8_deep12_lossx_lossu_01 0.8 1.0 1
```

The general command format is:

```bash
bash train_ivlp.sh <DATASET> <CONFIG_NAME> <ARG_1> <ARG_2> <ARG_3>
```

For the provided example:

* `CAMELYON17` specifies the training dataset.
* `epo20+10_v8_t8_deep12_lossx_lossu_01` specifies the experiment configuration.
* `0.8`, `1.0`, and `1` are the hyperparameters passed to the training script.

Please refer to `train_ivlp.sh` for the exact definitions and supported values of these arguments.

Training outputs, including model checkpoints and log files, will be saved to the output directory specified in the script or configuration file.


### Load a pre-trained Model
We have upload a pretrained weight. You can load it and evaluate in the target domain. The command is
```
bash eval.sh ISIC2019
```

## Repository Status

This repository provides the training and evaluation scripts for DLCL, together with a pre-trained model checkpoint for reproducing the reported results.

Additional documentation and model checkpoints will be released upon publication of the official IJCAI 2026 proceedings.


### Acknowledgement
Thanks for the following projects:
- [CLIP](https://github.com/openai/CLIP)
- [Dassl](https://github.com/KaiyangZhou/Dassl.pytorch)
- [CoOp](https://github.com/KaiyangZhou/CoOp)
- [DAPL](https://github.com/LeapLabTHU/DAPrompt)
- [Maple](https://github.com/muzairkhattak/multimodal-prompt-learning)



