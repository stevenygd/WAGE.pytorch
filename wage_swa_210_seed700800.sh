#! /bin/bash

for seed in "700" "800"; do
    python3 train.py \
            --dataset CIFAR10 \
            --data_path ./data \
            --dir ./checkpoint/wage-replicate/sgd \
            --model WAGEVGG7 \
            --epochs=300 \
            --lr_init 8 \
            --log-name wage-replicate/wage/ \
            --swa \
            --swa_start 210 \
            --swa_lr 1 \
            --wl-weight 2 \
            --wl-grad 8 \
            --wl-activate 8 \
            --wl-error 8 \
            --wl-rand 16 \
            --weight-type wage \
            --grad-type wage \
            --layer-type wage \
            --batch_size 128 \
            --wd 0 \
            --save_freq 25 \
            --log-distribution \
            --seed ${seed};
done
