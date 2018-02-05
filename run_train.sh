#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 python train.py -data cub -net mxnet_resnet_50  -lr 1e-4 -dim 512 -num_instances 16 -BatchSize 64  -loss softmax -epochs 1000 -log_dir mxnet_resnet_50_cub_s_1

