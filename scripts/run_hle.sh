#!/usr/bin/env bash

python3 open_agent/run_hle.py \
  config=configs/hle.yaml \
  model_id=o4-mini-2025-04-16 \
  model_id_search=o4-mini-2025-04-16 \
  model_id_retrieval=o4-mini-2025-04-16 \
  run_name=o4-mini-2025-04-16-hle \
