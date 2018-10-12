#! /bin/bash

dir="./checkpoint/wage/lr8lr2_210_testrun";
seed=100;
python3 train.py \
        --dataset CIFAR10 \
        --data_path ./data \
        --dir $dir \
        --model WAGEVGG7 \
        --epochs=300 \
        --log-name wage-replicate/wage/lr8lr2_210_testrun \
        --swa \
        --swa_start 200 \
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
        --seed ${seed} \
        --lr_init 8 \
        --lr_changes 210 \
        --lr_schedules 2 \
        --swa_lr 2;
