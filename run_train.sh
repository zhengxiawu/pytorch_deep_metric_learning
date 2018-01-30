#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 python train.py -data cub -net resnet_50  -lr 1e-4 -dim 512 -num_instances 4 -BatchSize 64  -loss softmax -epochs 401 -log_dir cub_m_01_1e4_n_4_b_64  -save_step 10

