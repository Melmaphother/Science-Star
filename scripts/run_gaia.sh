#!/usr/bin/env bash

# Single agent (ToolCallingAgent only)
# PYTHONPATH=.:science_star python3 science_star/run_single_agent.py config=configs/gaia.yaml

# Multi agent (CodeAgent + search agent)
PYTHONPATH=.:science_star python3 science_star/run_multi_agent.py \
  config=configs/gaia.yaml \
  models.name=gpt-4o-mini \
  runtime.run_name=gpt-4o-mini-gaia
