#! /bin/bash
for seed in "100" "200" "300" "400"; do
    python3 train.py \
            --dataset CIFAR10 \
            --data_path ./data \
            --dir ./checkpoint/wage-swa-diffc/swa \
            --model WAGEVGG7 \
            --epochs 300 \
            --lr_init 8 \
            --log-name wage-swa-diffc/swa/ \
            --swa \
            --swa_lr 1 \
            --swa_start 290 \
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
            --lr_changes 200 250 \
            --lr_schedules 8 1;
done
