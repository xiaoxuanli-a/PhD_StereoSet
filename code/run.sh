python eval_generative_models.py \
  --pretrained-class meta-llama/Llama-3.2-3B-Instruct \
  --intrasentence-model LLaMALM \
  --intrasentence-load-path ../../PhD_causalDPO/.cache/root/hh_dpo_llama3b_random_2025-03-23_00-13-18_121917/LATEST/policy.pt \
  --tokenizer LlamaTokenizer \
  --batch-size 1 \
  --input-file ../data/dev.json \
  --output-dir ./predictions \
  --skip-intersentence
