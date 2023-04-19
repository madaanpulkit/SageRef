#!/bin/bash

# Run 1: Reduced LR
nohup python main.py \
    --gpu 0 \
    --mode train \
    --epochs 100 \
    --learning_rate 0.0005 \
    --out_dir out/.0005lr \
&> out/exp0_.0005lr.out &

# Run 2: More Epochs
nohup python main.py \
    --gpu 1 \
    --mode train \
    --epochs 1000 \
    --out_dir out/1000epoch \
&> out/exp0_1000epoch.out &

# Run 3: Higher Latent Dim
nohup python main.py \
    --gpu 2 \
    --latent_dim 128 \
    --mode train \
    --epochs 100 \
    --out_dir out/128latent \
&> out/exp0_128latent.out &

# Run 3: Lower Latent Dim
nohup python main.py \
    --gpu 3 \
    --latent_dim 32 \
    --mode train \
    --epochs 100 \
    --out_dir out/32latent \
&> out/exp0_32latent.out &