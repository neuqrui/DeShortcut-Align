# Implementation map

DeShortcut-Align is a modified [veRL](https://github.com/volcengine/verl) trainer plus the Stage 1–2 scripts under `scripts/`.

| Paper | Code |
|-------|------|
| Refusal Sensitivity Attribution, modes A/B/C | `scripts/attribution/build_attribution_set.py` |
| Token Sensitivity Score (TSS) | `scripts/attribution/compute_tss.py` (`eta` is TSS_m; aggregated field is `tss`) |
| Attribution-Guided Contrastive Augmentation (AGCA) | `scripts/agca/synthesize.py`, `scripts/agca/generate_responses.py` |
| CCR in SFT, weight alpha | `model.ccr` in `verl/verl/trainer/sft_trainer.py`, `losses.py`, `transformer_impl.py` |
| Attention blinding of S_fmt | `fmt_token_ids` / `algorithm.ccr.fmt_tokens` |
| CCR in RL, L_cf = G * (-log pi(y \| X_cf)) | `algorithm.ccr.loss_type=ce_loss`, `ce_mode=hard` in `dp_actor.py` |
| Safety gate G = 1(R_judge > gamma) | `algorithm.ccr.apply_mode=positive_only`, `algorithm.ccr.gamma` |
| Adaptive lambda_dyn, rho_max, tau | `algorithm.ccr.lambda`, `rho_max`, `tau` |

Set `model.ccr.enable=false` or `algorithm.ccr.enable=false` to recover the unmodified SFT or GRPO/PPO update.
