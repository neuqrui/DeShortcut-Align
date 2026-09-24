#!/usr/bin/env bash
# GRPO with Counterfactual Consistency Regularization (CCR).
#
# On a verified rollout y ~ pi(. | X), CCR blinds formatting tokens and adds
#   L_cf = G * (-log pi(y | X_cf))
# with adaptive weight lambda_dyn. G = 1(R_judge > gamma) when
# apply_mode=positive_only. apply_mode=all still restricts CCR to the
# harmful data-source allowlist, so AGCA benign prompts are not pulled
# toward refusal.
#
# Usage:
#   MODEL_PRESET=ds-7b    bash examples/deshortcut_align/run_grpo.sh
#   MODEL_PRESET=ds-14b   bash examples/deshortcut_align/run_grpo.sh
#   MODEL_PRESET=qwen3-4b bash examples/deshortcut_align/run_grpo.sh
#   CCR_ENABLE=false MODEL_PRESET=ds-7b bash examples/deshortcut_align/run_grpo.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
export PROJECT_ROOT
# shellcheck disable=SC1091
source "$PROJECT_ROOT/scripts/common/env.sh"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/presets.sh"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"
export RAY_ADDRESS=""
export RAY_DEFAULT_PORT="${RAY_DEFAULT_PORT:-6402}"
export RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8269}"

EXPERIMENT_NAME="${EXPERIMENT_NAME:-deshortcut-grpo-${MODEL_PRESET}}"
CCR_ENABLE="${CCR_ENABLE:-true}"
LAMBDA_CCR="${LAMBDA_CCR:-0.1}"
CCR_GAMMA="${CCR_GAMMA:-0.8}"
CCR_TAU="${CCR_TAU:-0.001}"
MIXED_DIR="${MIXED_DIR:-$DATASETS_ROOT/processed/rl_${MODEL_PRESET}}"

cd "$VERL_ROOT"

python3 -m examples.data_preprocess.safety_data \
    --harmful_json "$HARMFUL_JSON" \
    --benign_json "$BENIGN_RL_JSON" \
    --harmful "$N_HARMFUL" \
    --benign "$N_BENIGN" \
    --output_dir "$MIXED_DIR" \
    --train_data_size "$TRAIN_DATA_SIZE" \
    --val_data_size "$VAL_DATA_SIZE" \
    --seed "${SEED:-123}"

THINKING_ARGS=()
if [[ "$ENABLE_THINKING" == "true" ]]; then
    THINKING_ARGS+=(data.apply_chat_template_kwargs.enable_thinking=true)
fi

mkdir -p "$VERL_ROOT/experiment_logs"
export EXPERIMENT_LOG_DIR="$VERL_ROOT/experiment_logs/grpo_${EXPERIMENT_NAME}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$EXPERIMENT_LOG_DIR"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="['$MIXED_DIR/train.parquet']" \
    data.val_files="['$MIXED_DIR/test.parquet']" \
    data.train_batch_size="$GRPO_TRAIN_BSZ" \
    data.val_batch_size="$GRPO_VAL_BSZ" \
    data.max_prompt_length=1024 \
    data.max_response_length=4096 \
    data.return_full_prompt=True \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    "${THINKING_ARGS[@]}" \
    reward.custom_reward_function.path="$REWARD_ROOT/async_safe_reward.py" \
    reward.custom_reward_function.name="compute_score" \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.actor.optim.lr=5e-6 \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.actor.ppo_mini_batch_size="$GRPO_MINI_BSZ" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="$GRPO_MICRO_BSZ" \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    +actor_rollout_ref.model.override_config.attn_implementation="eager" \
    actor_rollout_ref.actor.fsdp_config.param_offload="$GRPO_OFFLOAD" \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload="$GRPO_OFFLOAD" \
    actor_rollout_ref.actor.checkpoint.save_contents='[model]' \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="$GRPO_LOGPROB_MICRO" \
    actor_rollout_ref.rollout.tensor_model_parallel_size="$GRPO_TP" \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization="$GRPO_GPU_MEM" \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.temperature="$GRPO_TEMP" \
    actor_rollout_ref.rollout.top_p=0.95 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=4096 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="$GRPO_REF_LOGPROB_MICRO" \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    algorithm.ccr.enable="$CCR_ENABLE" \
    algorithm.ccr.start_epoch=1 \
    algorithm.ccr.lambda="$LAMBDA_CCR" \
    algorithm.ccr.rho_max="$GRPO_RHO_MAX" \
    algorithm.ccr.tau="$CCR_TAU" \
    algorithm.ccr.loss_type=ce_loss \
    algorithm.ccr.kl_penalty=low_var_kl \
    algorithm.ccr.ce_mode=hard \
    algorithm.ccr.apply_mode="$GRPO_APPLY_MODE" \
    algorithm.ccr.gamma="$CCR_GAMMA" \
    +algorithm.ccr.data_source_allowlist='["custom_safety_dataset"]' \
    trainer.critic_warmup=0 \
    trainer.use_legacy_worker_impl="enable" \
    trainer.logger='["console","swanlab"]' \
    trainer.log_val_generations="$VAL_DATA_SIZE" \
    trainer.project_name=deshortcut-align \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    trainer.n_gpus_per_node="$N_GPUS" \
    trainer.nnodes=1 \
    trainer.save_freq=75 \
    trainer.test_freq=20 \
    trainer.total_epochs=1 \
    trainer.resume_mode=disable \
    trainer.val_before_train=True \
    "$@"
