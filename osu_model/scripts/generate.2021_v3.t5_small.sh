#!/bin/bash

cd $(dirname $0)/..

export CUDA_VISIBLE_DEVICES=0
data=2021_v3_en
model=t5_small
SAVEDIR=osu_model_checkpoints/$data.$model
testpfx=valid
hyp=$SAVEDIR/hyp.$testpfx.txt
src=data-prep/$data/$testpfx.mr
ref=data-prep/$data/$testpfx.lx

#python finetune-transformers/generate.py \
  #--model-class "t5" \
  #--output-path $(readlink -f $hyp) \
  #--test-source-data-path $(readlink -f $src) \
  #--save-dir $(readlink -f $SAVEDIR) \
  #--batch-size 8 \
  #--beam-size 5 \
  #--max-length 200 \
  #--clean-up-tokenization-spaces

python e2e-metrics/measure_scores.py -p $ref $hyp 2> /dev/null
