# Training and Inference with Integers in Deep Neural Networks

PyTorch implementation for the ICLR 2018 oral paper, training on CIFAR10. This is replicate from the Tensorflow [repo](https://github.com/boluoweifenda/WAGE) by the paper's authors. We hope the PyTorch implementation could also help with low-precision training research.

## Prerequisites
- NVIDIA GPU + CUDA + CuDNN
- PyTorch
- TensorboardX 
- Tabulate
- tqdm

Please follow the official instruction to install PyTorch and NVIDIA related prerequisites. Other things should be handled by
```bash
pip install -r requirements.txt
```

## Train
Start training using the following scripts:
```bash
./wage.sh
```

## Results 

Averaging four seeds gives: 93.04% accuracy at 300 epochs.


## Citation
If you find this paper or this repository helpful, please cite the original paper:
```bash
@inproceedings{
wu2018training,
title={Training and Inference with Integers in Deep Neural Networks},
author={Shuang Wu and Guoqi Li and Feng Chen and Luping Shi},
booktitle={International Conference on Learning Representations},
year={2018},
url={https://openreview.net/forum?id=HJGXzmspb},
} 
```


