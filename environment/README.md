# Environment

Python **3.10** and NVIDIA GPUs. Paper runs use FSDP + vLLM.

## Install

```bash
conda create -n deshortcut python=3.10 -y
conda activate deshortcut

# Important: keep installs inside the conda env (not ~/.local)
export PYTHONNOUSERSITE=1
export PATH="$CONDA_PREFIX/bin:$PATH"

python -m pip install -r environment/requirements.txt
python -m pip install -e verl/
python -m pip install google-genai
```

`environment/requirements-verl-freeze.clean.txt` is an optional freeze of a full training stack. Prefer `requirements.txt` for a fresh install. The header of `requirements.txt` documents known pitfalls in detail.

## flash-attn

Install **after** torch. Source builds need `nvcc` and a matching `CUDA_HOME`. If those are missing, use a prebuilt wheel that matches **torch 2.4 + cu12 + cp310 + cxx11abiFALSE**:

```bash
# Source build (when CUDA toolkit matches torch's CUDA 12.1):
python -m pip install flash-attn==2.7.3 --no-build-isolation

# Or prebuilt wheel from Dao-AILab releases, e.g.:
# flash_attn-2.7.3+cu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
python -m pip install /path/to/flash_attn-....whl
```

GRPO/PPO launchers set `VLLM_ATTENTION_BACKEND=FLASH_ATTN`, so flash-attn is required for the RL recipe.

## Reference stack

| Package | Version |
|---------|---------|
| Python | 3.10 |
| torch | 2.4.0+cu121 |
| vLLM | 0.6.3 |
| flash-attn | 2.7.3 |
| transformers | 4.57.3 |

Stay on this torch pin unless you also upgrade vLLM. veRL code paths that import `torch.distributed.tensor.DTensor` or call `.sum()` on NestedTensors assume torch ≥ 2.5 in places; on 2.4 use `torch.distributed._tensor` and `nested.values().sum()` (see `requirements.txt` notes).

## API keys / mirrors

```bash
cp .env.example .env
# OPENAI_API_KEY / OPENAI_BASE_URL  — RL reward judge and AGCA synthesis
# HF_ENDPOINT                       — optional Hub mirror
# SWANLAB_API_KEY                   — optional logging
```

Do not commit `.env`. SFT + CCR can run without an OpenAI key; GRPO/PPO with the paper reward need a working judge API.

## Sanity check

```bash
python -c "import torch, vllm, flash_attn, verl; print(torch.__version__, torch.cuda.is_available())"
which python torchrun   # should both live under $CONDA_PREFIX/bin
```
