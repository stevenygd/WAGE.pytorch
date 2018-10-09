#! /bin/bash

python train.py \
       --dataset CIFAR10 \
       --data_path ./data \
       --dir ./checkpoint/wage-replicate/sgd \
       --model WAGEVGG7 \
       --epochs=300 \
       --lr_init 8 \
       --log-name wage-replicate/wage/ \
       --swa \
       --swa_start 260 \
       --wl-weight 2 \
       --wl-grad 8 \
       --wl-activate 8 \
       --wl-error 8 \
       --wl-rand 16 \
       --weight-type wage \
       --grad-type wage \
       --layer-type wage \
       --seed 100 \
       --batch_size 128 \
       --wd 0 \
       --save_freq 25;
