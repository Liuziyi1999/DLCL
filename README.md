# DLCL
When Vision-Language Models Meet Fetal Cardiac Ultrasound: Dual-Level Contrastive Learning for Out-of-Distribution Detection

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


### Load a pre-trained Model
We have upload a pretrained weight. You can load it and evaluate in the target domain. The command is
```
bash eval.sh ISIC2019
```

### Other information
Currently, only the test version is accessible. The training version will be released to the public once the article is published.

### Acknowledgement
Thanks for the following projects:
- [CLIP](https://github.com/openai/CLIP)
- [Dassl](https://github.com/KaiyangZhou/Dassl.pytorch)
- [CoOp](https://github.com/KaiyangZhou/CoOp)
- [DAPL](https://github.com/LeapLabTHU/DAPrompt)
- [Maple](https://github.com/muzairkhattak/multimodal-prompt-learning)



