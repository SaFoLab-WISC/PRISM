# Test-Time Scaling (TTS) with Beam Search and Self-Rewarding

A sophisticated test-time scaling implementation for the PRISM reasoning framework that uses beam search at each reasoning step combined with self-rewarding evaluation to improve response quality.

## 🚀 Quick Start
### Basic Usage

You should change the config/tts_config.yaml for the model path and data path.


### Starting the VLLM Server for Reward Model

You should first change the model path in start_vllm_reward.sh and start_vllm_actor.sh.

```bash
# Start the server
./start_vllm_reward.sh
./start_vllm_actor.sh
```

**Using Configuration File:**
```bash
python tts_beam_search.py --config custom_cfg/tts_config.yaml
```


## ⚙️ Configuration

### Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `beam_width` | Candidates kept at each step | 5 |
| `num_candidates` | Initial candidates per step | 8 |
| `final_candidates` | Complete responses for selection | 10 |
| `use_step_rewards` | Enable step-wise evaluation | true |
| `use_final_rewards` | Enable final evaluation | true |

### Step-Specific Token Limits

```yaml
step_max_tokens:
  PROBLEM: 512     # Problem analysis
  CAPTION: 512     # Image description
  REASONING: 1024  # Logical reasoning
  OUTPUT: 512      # Final answer
```

## 🔧 Usage

### Beam Search vs Best-of-N

**Pure Beam Search:**
```python
config = TTSConfig(
    beam_width=5,
    num_candidates=8,
    use_step_rewards=True,
    use_final_rewards=True
)
```

**Pure Best-of-N:**
```python
config = TTSConfig(
    beam_width=1,
    num_candidates=1,
    final_candidates=10,
    use_step_rewards=False,
    use_final_rewards=True
)
```
### Multi-threading Support

```yaml
worker_num: 4           # Number of worker threads
worker_prompt_num: 25   # Prompts per worker
```