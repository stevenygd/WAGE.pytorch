#! /bin/bash
for seed in "100" "200" "300" "400" "500" "600"; do
    python3 train.py \
            --dataset CIFAR10 \
            --data_path ./data \
            --dir ./checkpoint/best/full_$2_300_lr$1 \
            --model WAGEVGG7 \
            --epochs 300 \
            --lr_init 8 \
            --log-name best/eval_full_$2_300_lr$1 \
            --swa \
            --swa_lr $1 \
            --swa_start $2 \
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
            --resume ./checkpoint/wage-new-replicate/sgd-seed-${seed}/checkpoint-200.pt \
            --lr_changes 200 \
            --lr_schedules $1;
done
