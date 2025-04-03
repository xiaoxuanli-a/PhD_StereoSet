#!/bin/bash

python eval_generative_models.py \
  --pretrained-class meta-llama/Llama-3.2-1B-Instruct \
  --intrasentence-model LLaMALM \
  --intrasentence-load-path /workspace/LATEST/policy.pt \
  --intersentence-model ModelNSP \
  --tokenizer AutoTokenizer \
  --batch-size 1 \
  --input-file ../data/dev.json \
  --output-dir ./predictions \
  --unconditional_start_token "<s>" \

python eval_generative_models.py \
  --pretrained-class meta-llama/Llama-3.2-1B-Instruct \
  --intrasentence-model LLaMALM_gender_debiased \
  --intrasentence-load-path /workspace/LATEST/policy.pt \
  --intersentence-model ModelNSP \
  --tokenizer AutoTokenizer \
  --batch-size 1 \
  --input-file ../data/dev.json \
  --output-dir ./predictions \
  --unconditional_start_token "<s>" \

python eval_generative_models.py \
  --pretrained-class meta-llama/Llama-3.2-1B-Instruct \
  --intrasentence-model LLaMALM_race_debiased \
  --intrasentence-load-path /workspace/LATEST/policy.pt \
  --intersentence-model ModelNSP \
  --tokenizer AutoTokenizer \
  --batch-size 1 \
  --input-file ../data/dev.json \
  --output-dir ./predictions \
  --unconditional_start_token "<s>" \

python eval_generative_models.py \
  --pretrained-class meta-llama/Llama-3.2-1B-Instruct \
  --intrasentence-model LLaMALM_religion_debiased \
  --intrasentence-load-path /workspace/LATEST/policy.pt \
  --intersentence-model ModelNSP \
  --tokenizer AutoTokenizer \
  --batch-size 1 \
  --input-file ../data/dev.json \
  --output-dir ./predictions \
  --unconditional_start_token "<s>" \

python eval_generative_models.py \
  --pretrained-class meta-llama/Llama-3.2-1B-Instruct \
  --intrasentence-model LLaMALM_profession_debiased \
  --intrasentence-load-path /workspace/LATEST/policy.pt \
  --intersentence-model ModelNSP \
  --tokenizer AutoTokenizer \
  --batch-size 1 \
  --input-file ../data/dev.json \
  --output-dir ./predictions \
  --unconditional_start_token "<s>" \

# python eval_generative_models.py \
#   --pretrained-class meta-llama/Llama-3.2-3B-Instruct \
#   --intrasentence-model LLaMALM_debiased \
#   --intrasentence-load-path ../../PhD_causalDPO/.cache/root/hh_dpo_llama3b_random_stereo_debias_2025-03-23_06-00-07_066205/LATEST/policy.pt \
#   --intersentence-model LLaMALM_debiased \
#   --tokenizer AutoTokenizer \
#   --batch-size 1 \
#   --input-file ../data/dev.json \
#   --output-dir ./predictions \
#   --unconditional_start_token "<s>" \
