#!/bin/bash

nohup python main.py \
  --tokenizer AutoTokenizer \
  --model LlamaModel \
  --pretrained-class meta-llama/Llama-3.2-1B-Instruct \
  --dataset ./process_wikipedia/output \
  --batch-size 4 \
  --epochs 3 \
  --max-seq-length 256 \
  --core-lr 5e-6 \
  --head-lr 1e-3 \
  --accumulation-steps 8 \
  --fp16 > llama_nsp_training.log 2>&1 &
