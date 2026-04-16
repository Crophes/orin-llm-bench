#!/usr/bin/env bash
MODELS_DIR="/data-llm/llama_container/models/"
for model in "$MODELS_DIR"/*.gguf; do
  model_name=$(basename "$model")
  model_path="/models/${model_name}"
  jetson-containers run \-v /data-llm/llama_container/models:/models \
  b47912ab278c \
  python3 /data/ram_usage.py--model "$model_path"
done
