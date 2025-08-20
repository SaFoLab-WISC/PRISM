# PRISM: Principled Reasoning for Integrated Safety in Multimodality

This repository provides the implementation of **PRISM**, an alignment framework that integrates principled reasoning with safety through structured, multi-step reasoning.


## 🚀 Quick Start

### Prerequisites

```bash
conda create -n PRISM python=3.10
conda activate PRISM
pip install 'ms-swift[all]' -U
pip install vllm
```

## 1) Model Training

### 📦 Datasets

We open-source the training datasets on Hugging Face:

- PRISM-CoT: https://huggingface.co/datasets/andyc03/PRISM-CoT
- PRISM-DPO: https://huggingface.co/datasets/andyc03/PRISM-DPO

First, prepare the data. We have released the PRISM-CoT and PRISM-DPO datasets. Convert your dataset to a Swift-compatible format by providing the absolute path to your data folder:

```bash
python utils/formatting.py --folder /your_path_here/PRISM_COT
```

Then add the special tokens for your model using `utils/add_tokens.py`:

```bash
python utils/add_tokens.py --model_path /your_mode_path_here
```

Now you can train your PRISM model. Update the JSON and model path in `training_scripts/qwen2_vl.sh`, for example:

```bash
cd training_scripts

# For Qwen2-VL with full-parameters SFT
bash qwen2_vl.sh
```

### 📦 Model Weights

We provide the model weights used in our experiments on Hugging Face:

- Qwen2-VL-PRISM-SFT: https://huggingface.co/andyc03/Qwen2-VL-PRISM-SFT
- Qwen2-VL-PRISM-DPO: https://huggingface.co/andyc03/Qwen2-VL-PRISM-DPO

## 2) MCTS Data Generation

If you want to generate preference data using Monte Carlo Tree Search (MCTS), we provide scripts to help you do so:

```bash
cd PRISM_DPO_data
```

First, change the model path of your downloaded PRISM-CoT model in `scripts/activate_vllm.sh`, then launch it:

```bash
bash scripts/activate_vllm.sh
```

Next, configure your model path and data in `config/qwen_tree_generate.yaml`, then run MCTS data generation:

```bash
# Then run MCTS data generation
bash scripts/generate_MCT.sh
```

**Configuration parameters:**
- `actor_model_dir`: Path to your model
- `train_prompt_path`: Input prompts for data generation
- `iterations`: Number of MCTS iterations (default: 200)
- `c`: UCB exploration parameter (default: 1.5)
- `max_depth`: Maximum reasoning depth (default: 5)

## 3) Test-Time Scaling

Please refer to `TTS/TTS.md` for running details.

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## 📚 Citation

If you use PRISM in your research, please consider citing our paper:

```bibtex
@article{prism2025,
  title={Robust VLM Alignment with PRISM: Principled Reasoning for Integrated Safety in Multimodality},
  author={[Authors]},
  year={2025}
}
```

## 🙏 Acknowledgments

Built on top of excellent open-source projects including [ms-swift](https://github.com/modelscope/ms-swift), [vLLM](https://github.com/vllm-project/vllm), and [STAIR](https://github.com/thu-ml/STAIR).

---

For questions, issues, or discussions, please open an issue in this repository or contact the author at andyc_03@sjtu.edu.cn.
