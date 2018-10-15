#! /bin/bash

python train.py \
    --dataset CIFAR10 \
    --data_path ./data \
    --dir ./checkpoint/wage-replicate/sgd \
    --model WAGEVGG7 \
    --epochs=300 \
    --log-name wage-momentum-quantaf-lowvel-swa \
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
    --lr_init 4 \
    --lr_changes 200 250 \
    --lr_schedules .5 0.05 \
    --momentum 0.9 \
    --swa \
    --swa_start 240 \
    --swa_lr 0.5 \
    --wd 0
