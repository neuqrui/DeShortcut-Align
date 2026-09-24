<h1 align="center">
  <img src="assets/deshortcut-align-icon.png" alt="" width="120" align="absmiddle">
  <img src="assets/title.png" alt="DeShortcut-Align" height="92" align="absmiddle">
</h1>

<p align="center"><b>Decoupling Spurious Shortcuts for Robust Safety Alignment in Large Reasoning Models</b></p>

<p align="center">
  <a href="#citation">📄 Paper</a>
  &nbsp;&middot;&nbsp;
  <a href="#training">💻 Code</a>
</p>

Safety alignment of large reasoning models often looks strong because refusals get tied to chat templates and sensitive keywords. Strip those cues and the defense drops, benign queries are refused, and general reasoning pays an alignment tax. DeShortcut-Align trains the policy to decide from the query itself.

## Problem

Aligned models lean on two spurious shortcuts. A **formatting shortcut** binds refusal to structural wrappers that show up throughout safety corpora (pre-training and post-training chat templates). A **lexical shortcut** treats a sensitive word as enough reason to refuse, including on benign requests.

The formatting shortcut is easy to measure. The user instruction stays the same; only the template is removed. Defense success still falls on WildJailbreak, StrongReject, and WildChat, for the base model and for standard alignment methods (TARS, SFT, STAR-1, PPO, GRPO).

![Template vulnerability. Defense success with the post-training template, the pre-training template, and no template.](assets/problem_bar.png)

*Defense success rate with the DeepSeek-R1 post-training template, with the pre-training template, and with the template removed. The gap is the formatting shortcut.*

## Method

DeShortcut-Align decouples those shortcuts in three stages.

1. **Refusal Sensitivity Attribution.** Mask each prompt token and measure how much the final refusal distribution moves. The Token Sensitivity Score (TSS) is the absolute teacher-forced log-likelihood shift on the refusal tokens, averaged over three wrappers: bare instruction, pre-training template, and full template.
2. **Attribution-Guided Contrastive Augmentation (AGCA).** Keep the top-K high-TSS tokens, rewrite the query so the intent is benign, and mix those pairs into SFT and RL. Keyword matching is then penalized by the helpfulness reward.
3. **Counterfactual Consistency Regularization (CCR).** Attention blinding hides formatting tokens `S_fmt`. SFT adds `alpha * L_CE(y | X_cf)`. RL adds `lambda_dyn * G * (-log pi(y | X_cf))` on rollouts that pass the safety gate `G = 1(R_judge > gamma)`.

![Overview of DeShortcut-Align: sensitivity attribution, contrastive augmentation, and counterfactual consistency.](assets/deshortcut.drawio.svg)

Training runs on a modified veRL trainer. Defaults in `examples/deshortcut_align/` follow the paper recipe.

| Paper | Code |
|-------|------|
| Refusal Sensitivity Attribution | `scripts/attribution/build_attribution_set.py` |
| Token Sensitivity Score (TSS) | `scripts/attribution/compute_tss.py` |
| AGCA synthesis | `scripts/agca/synthesize.py` |
| CCR in SFT, weight `alpha` | `model.ccr.alpha` |
| Formatting tokens `S_fmt` | `model.ccr.fmt_token_ids` or `algorithm.ccr.fmt_tokens` |
| CCR in RL, hard cross-entropy on `X_cf` | `algorithm.ccr.loss_type=ce_loss`, `ce_mode=hard` |
| Safety gate `G = 1(R_judge > gamma)` | `algorithm.ccr.apply_mode=positive_only`, `algorithm.ccr.gamma` |
| Adaptive `lambda_dyn`, `rho_max`, `tau` | `algorithm.ccr.lambda`, `rho_max`, `tau` |

File-level notes are in [docs/IMPLEMENTATION.md](docs/IMPLEMENTATION.md).

## Training dynamics

The same four signals, on standard GRPO and on DeShortcut-Align.

<p align="center">
  <img src="assets/reward_hacking_curves_pure_grpo.png" width="49%" alt="Standard GRPO: reward, over-refusal, template gap, and policy-gradient loss.">
  <img src="assets/reward_hacking_curves_qmm_grpo.png" width="49%" alt="DeShortcut-Align: reward stays unsaturated and the policy gradient remains.">
</p>

<p align="center"><i>Left: standard GRPO. Right: DeShortcut-Align.</i></p>

Standard GRPO falls into three phases. From step 0 to 40 the reward climbs, over-refusal rises with it, and the template gap opens to about 45: the update locks onto the superficial map from chat template to a safe response. From step 40 to 100 that gap narrows and the reward sits near saturation, but over-refusal keeps climbing toward 0.74, so the shortcut has moved from format to sensitive words. Past step 100 the reward is pinned at 1.0. Within-group reward variance collapses, and by step 100 the policy-gradient magnitude is about 95% below its early peak, which is gradient starvation: later steps have almost nothing left to move the policy off the shortcut. DeShortcut-Align takes a different path. Suppressing lexical shortcuts first pushes the template gap to about 65. Once counterfactual consistency is on, attention blinding brings that gap 42.9% under the standard-GRPO baseline, and the contrastive benign queries cut over-refusal by 58.3%. The reward stays around 0.85–0.95 instead of saturating at 1.0, and about 87% of the gradient RMS remains, so the optimizer still has a signal to refine the policy.

## Main results

Template robustness, over-refusal, and reasoning on DeepSeek-R1-Distill 7B/14B and Qwen3-4B. Robustness is **with template / without template** (gap underneath). Best and second-best exclude the base model. Shaded rows are DeShortcut-Align.

![Main results](assets/main_results.jpg)

Relative to standard GRPO, DeShortcut-Align cuts the template gap by about 44% (7B), 68% (14B), and 60% (Qwen3-4B), and cuts average false rejection by about 58%, 52%, and 45%. Reasoning (`Avg2`) is retained rather than taxed: about +2.0, +2.6, and +0.7 points versus standard GRPO. On the 7B jailbreak suite (PAIR, GCG, TAP) the RL variant reaches 97.7 average defense success.

## Getting started

Python 3.10+ and NVIDIA GPUs (FSDP + vLLM).

```bash
conda create -n deshortcut python=3.10 -y
conda activate deshortcut
pip install -r environment/requirements.txt
pip install -e verl/
pip install google-genai

cp .env.example .env
# Set OPENAI_API_KEY if you use the RL reward judge or AGCA synthesis.
```

More install notes are in [environment/README.md](environment/README.md).

## Data

| Directory | Role |
|-----------|------|
| `scripts/attribution/` | Build `D_attr` and compute TSS |
| `scripts/agca/` | Synthesize contrastive benign queries and responses |
| `datasets/raw/` | Harmful pools and shipped AGCA JSON |
| `examples/deshortcut_align/` | SFT, GRPO, and PPO launchers |
| `reward/` | Safety and helpfulness reward used by RL |
| `verl/` | Training backend |

Stage 1–2, from a harmful instruction file:

```bash
python scripts/run_tss_agca.py \
    --input datasets/raw/risk_data_7B.json \
    --generator_model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
    --tss_model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
    --top_k 15
```

Helpful responses for the benign queries:

```bash
python scripts/agca/generate_responses.py \
    --input outputs/tss_agca/<run>/agca_contrastive_instructions.json \
    --output datasets/raw/agca_benign_sft.json
```

`datasets/raw/agca_benign_rl.json` and `agca_benign_sft.json` are ready to train on. See [datasets/README.md](datasets/README.md).

## Training

The launchers default to `MODEL_PRESET=ds-7b` (DeepSeek-R1-Distill-Qwen-7B). DeepSeek presets use the R1-Distill chat markers as formatting tokens. `ds-14b` and `qwen3-4b` are optional overrides; the Qwen3-4B preset turns thinking on and blinds `<|im_start|>`, `<|im_end|>`, `user`, and `assistant`. RL formatting tokens are auto-detected when `algorithm.ccr.fmt_tokens` is left unset.

```bash
# SFT + CCR (ds-7b)
bash examples/deshortcut_align/run_sft.sh

# GRPO + CCR (ds-7b)
bash examples/deshortcut_align/run_grpo.sh

# PPO + CCR (ds-7b)
bash examples/deshortcut_align/run_ppo.sh
```

Common overrides:

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
export N_GPUS=4
export MODEL_PATH=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
export LAMBDA_CCR=0.1
export CCR_GAMMA=0.8
CCR_ENABLE=false MODEL_PRESET=ds-7b bash examples/deshortcut_align/run_grpo.sh
```

`CCR_ENABLE=false` runs the same data and optimizer without the counterfactual term.

| Preset | Model | GPUs | Notes |
|--------|-------|------|-------|
| `ds-7b` | DeepSeek-R1-Distill-Qwen-7B | 4 | GRPO `apply_mode=all` on the harmful allowlist, `rho_max=10` |
| `ds-14b` | DeepSeek-R1-Distill-Qwen-14B | 8 | GRPO `apply_mode=positive_only`, `rho_max=5`, parameter offload |
| `qwen3-4b` | Qwen3-4B | 4 | `enable_thinking=true`, Qwen formatting-token ids for SFT |

SFT uses learning rate `1e-5` for 5 epochs and `alpha=1`. GRPO and PPO use learning rate `5e-6`. PPO gates CCR with `gamma=0.8`.

## Evaluation

Safety, template robustness, and over-refusal use [LLM-Safety-Eval](https://github.com/neuqrui/LLM-Safety-Eval): WildJailbreak, StrongReject, WildChat, XSTest, OKTest, FalseReject, and the PAIR / GCG / TAP attacks. General capabilities (MATH-500, MMLU, LiveCodeBench, HumanEval) use [OpenCompass](https://github.com/open-compass/opencompass).

## Citation

```bibtex
@misc{deshortcut-align2026,
  title  = {DeShortcut-Align: Decoupling Spurious Shortcuts for Robust Safety Alignment in Large Reasoning Models},
  author = {},
  year   = {2026}
}
```

Please also cite veRL.

## License

This repository is released under the Apache License 2.0. It includes a modified copy of veRL. See `verl/LICENSE`.
