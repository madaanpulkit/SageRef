#!/bin/bash

# Run 1: Reduced LR, Reduced batch size, High Epochs, High Dim
nohup python main.py \
    --gpu 0 \
    --mode train \
    --epochs 1000 \
    --batch_size 16 \
    --latent_dim 128 \
    --learning_rate 0.00005 \
    --data_dir data/CEILNET \
    --out_dir out/ceil_.00005lr_1000ep_128dim_16bs \
&> out/exp1_ceil_.0005lr_1000ep_128dim_16bs.out &

# Run 1: Reduced LR, Reduced batch size, High Epochs, Higher Dim
nohup python main.py \
    --gpu 1 \
    --mode train \
    --epochs 1000 \
    --batch_size 16 \
    --latent_dim 512 \
    --learning_rate 0.00005 \
    --data_dir data/CEILNET \
    --out_dir out/ceil_.00005lr_1000ep_512dim_16bs \
&> out/exp1_ceil_.0005lr_1000ep_512dim_16bs.out &
