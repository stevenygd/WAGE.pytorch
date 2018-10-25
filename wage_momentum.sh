#! /bin/bash

python train.py \
    --dataset CIFAR10 \
    --data_path ./data \
    --dir ./checkpoint/wage-replicate/sgd \
    --model WAGEVGG7 \
    --epochs=300 \
    --log-name momentum-rejoin \
    --wl-weight 2 \
    --wl-grad 8 \
    --wl-activate 8 \
    --wl-error 8 \
    --wl-rand 16 \
    --quant-type stochastic \
    --weight-type wage \
    --grad-type wage \
    --layer-type wage \
    --seed 100 \
    --batch_size 128 \
    --lr_init 8 \
    --lr_changes 200 250 \
    --lr_schedules 1. 0.125 \
    --wd 0;
