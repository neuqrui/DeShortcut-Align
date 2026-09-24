# Environment

Python 3.10 and NVIDIA GPUs. The paper runs use FSDP and vLLM.

```bash
conda create -n deshortcut python=3.10 -y
conda activate deshortcut
pip install -r environment/requirements.txt
pip install -e verl/
pip install google-genai
```

`environment/requirements-verl-freeze.clean.txt` is an optional freeze of the training stack. Prefer `requirements.txt` for a fresh install.

`flash-attn` often needs a prebuilt wheel that matches your torch and CUDA build:

```bash
pip install flash-attn==2.7.3 --no-build-isolation
```

Reference stack: Python 3.10, torch 2.4.0+cu121, vLLM 0.6.3, flash-attn 2.7.3.

Copy `.env.example` to `.env` and set API keys only if you use the reward judge or AGCA synthesis. Do not commit `.env`.
