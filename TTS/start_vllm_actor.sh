#!/bin/bash

vllm serve /home/user/workspace/models/PRISM-CoT \
    --dtype bfloat16 \
    --tensor-parallel-size 1 \
    --served-model-name actor \
    --allowed-local-media-path /home/user/workspace/MLLM_safeguard/PRISM_DPO_data/prompt_data \
    --port 8000
