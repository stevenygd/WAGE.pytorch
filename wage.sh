#! /bin/bash

python train.py \
    --dataset CIFAR10 \
    --data_path ./data \
    --dir ./checkpoint/wage-replicate/sgd \
    --model WAGEVGG7 \
    --epochs=300 \
    --lr_init 8 \
    --log-name wage-replicate/wage/ \
    --wl-weight 2 \
    --wl-grad 8 \
    --wl-activate 8 \
    --wl-error 8 \
    --wl-rand 16 \
    --quant-type stochastic \
    --weight-type wage \
    --grad-type wage \
    --layer-type wage \
    --quant-back \
    --seed 100 \
    --batch_size 128 \
    --wd 0;
