#!/bin/bash

cd $(dirname $0)/..

export CUDA_VISIBLE_DEVICES=0 # Change this to 'CUDA_VISIBLE_DEVICES=0,1' for example if you are using two GPUs
data=2021_v3_en
model=t5_small
SAVEDIR=osu_model_checkpoints/$data.$model
CACHEDIR=osu_model_cache/$model

mkdir -p $SAVEDIR
mkdir -p $CACHEDIR

python finetune-transformers/train_single.py \ # Change this to 'accelerate launch finetune-transformers/train.py' if running distributed training (multiple GPUs)
  --pretrained-model-path "t5-small" \
  --train-source-data-path $(readlink -f "data/data-prep/$data/train.mr") \
  --train-target-data-path $(readlink -f "data/data-prep/$data/train.lx") \
  --valid-source-data-path $(readlink -f "data/data-prep/$data/valid.mr") \
  --valid-target-data-path $(readlink -f "data/data-prep/$data/valid.lx") \
  --save-dir $(readlink -f $SAVEDIR) \
  --cache-dir $(readlink -f $CACHEDIR) \
  --max-epoch 500 --patience 10 \
  --batch-size 8 --update-frequency 1 \
  --learning-rate 2e-5 \
  --valid-batch-size 8 \
  --valid-beam-size 5 --valid-max-length 200

  # Took this out of the arguments because this file was not created in preprocessing and a 'None' value is acceptable here according to train.py
  #--indivisible-tokens-path $(readlink -f "data-prep/$data/indivisible_tokens.txt") \
