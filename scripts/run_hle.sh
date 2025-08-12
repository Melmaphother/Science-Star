#!/usr/bin/env bash
# Simple launcher with line-broken CLI overrides
# Example:
#   sh scripts/run_hle.sh \
#     split=test \
#     run_name=my_run \
#     concurrency=4 \
#     debug=true

set -euo pipefail

python3 open_agent/run_hle.py \
  config=configs/hle.yaml \
  "$@"
