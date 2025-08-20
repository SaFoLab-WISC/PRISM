#!/bin/bash

vllm serve /home/user/workspace/Qwen \
    --dtype bfloat16 \
    --tensor-parallel-size 1 \
    --served-model-name reward_model \
    --allowed-local-media-path /home/user/workspace/MLLM_safeguard/PRISM_DPO_data/prompt_data \
    --port 8001
