python eval_generative_models.py \
  --pretrained-class ../../PhD_causalDPO/.cache/root/hh_dpo_llama3b_random_2025-03-23_00-13-18_121917/LATEST/policy.pt \
  --intrasentence-model LLaMALM \
  --intrasentence-load-path ../../PhD_causalDPO/.cache/root/hh_dpo_llama3b_random_2025-03-23_00-13-18_121917/LATEST/policy.pt \
  --tokenizer LlamaTokenizer \
  --batch_size 1 \
  --input-file ../data/dev.json \
  --output-dir ./predictions \
  --skip-intersentence
