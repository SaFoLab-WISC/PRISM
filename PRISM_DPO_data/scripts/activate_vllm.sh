#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

vllm serve /home/user/workspace/models/PRISM-CoT \
    --dtype bfloat16 \
    --served-model-name actor \
    --tensor-parallel-size 1 \
    --allowed-local-media-path /home/user/workspace/MLLM_safeguard/PRISM_DPO_data/prompt_data \
    --port 8000
