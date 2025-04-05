#!/bin/bash

nohup python main.py \
  --tokenizer AutoTokenizer \
  --model LlamaModel \
  --pretrained-class meta-llama/Llama-3.2-1B-Instruct \
  --dataset ./process_wikipedia/output \
  --fp16 > llama_nsp_training.log 2>&1 &
