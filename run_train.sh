#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 python train.py -data cub -net mxnet_resnet_50  -lr 0.0001 -dim 0 -num_instances 16 -BatchSize 64  -loss softmax -epochs 2000 -log_dir mxnet_resnet_50_cub_s_1_dim_0

