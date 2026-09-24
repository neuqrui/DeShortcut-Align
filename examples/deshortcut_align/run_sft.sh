#!/usr/bin/env bash
# Supervised fine-tuning with Counterfactual Consistency Regularization (CCR).
#
# L_total = L_CE(y | X) + alpha * L_CE(y | X_cf)
# X_cf is the same sequence with attention blinded on formatting tokens S_fmt.
# CCR is applied on harmful demonstrations only, so intention-inverted benign
# samples are not trained toward a refusal target.
#
# Usage:
#   MODEL_PRESET=ds-7b   bash examples/deshortcut_align/run_sft.sh
#   MODEL_PRESET=ds-14b  bash examples/deshortcut_align/run_sft.sh
#   MODEL_PRESET=qwen3-4b bash examples/deshortcut_align/run_sft.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
export PROJECT_ROOT
# shellcheck disable=SC1091
source "$PROJECT_ROOT/scripts/common/env.sh"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/presets.sh"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export RAY_ADDRESS=""
unset HYDRA_FULL_ERROR || true
unset TORCH_DISTRIBUTED_DEBUG || true

EXPERIMENT_NAME="${EXPERIMENT_NAME:-deshortcut-sft-${MODEL_PRESET}}"
CCR_ENABLE="${CCR_ENABLE:-true}"
CCR_ALPHA="${CCR_ALPHA:-1}"
SFT_OUTPUT_DIR="${SFT_OUTPUT_DIR:-$DATASETS_ROOT/processed/sft_${MODEL_PRESET}}"
CUSTOM_TEMPLATE_PATH="${CUSTOM_TEMPLATE_PATH:-$TEMPLATES_ROOT/keep_think_chat_template.jinja}"

cd "$VERL_ROOT"

python examples/data_preprocess/star1_to_verl_sft_parquet.py \
    --harmful_json "$HARMFUL_SFT_JSON" \
    --benign_json "$BENIGN_SFT_JSON" \
    --output_dir "$SFT_OUTPUT_DIR" \
    --seed 123 \
    --test_size 0.01 \
    --think_style think \
    --harmful -1 \
    --benign -1

# DeepSeek-R1-Distill templates split the think span. Export a template that
# keeps the reasoning tokens inside the supervised target. Qwen3 does this
# when enable_thinking=true, so it uses the model chat template as-is.
TEMPLATE_ARGS=()
THINKING_ARGS=()
if [[ "$USE_DS_TEMPLATE" == "true" ]]; then
    python examples/data_preprocess/export_keep_think_template.py \
        --model_path "$MODEL_PATH" \
        --output_path "$CUSTOM_TEMPLATE_PATH" \
        --trust_remote_code
    TEMPLATE_ARGS+=(model.custom_chat_template="$CUSTOM_TEMPLATE_PATH")
else
    THINKING_ARGS+=(
        data.enable_thinking_default=true
        data.apply_chat_template_kwargs.enable_thinking=true
    )
fi

mkdir -p "$VERL_ROOT/experiment_logs"
export EXPERIMENT_LOG_DIR="$VERL_ROOT/experiment_logs/sft_${EXPERIMENT_NAME}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$EXPERIMENT_LOG_DIR"

torchrun --standalone --nnodes=1 --nproc_per_node="$N_GPUS" \
    -m verl.trainer.sft_trainer \
    data.train_files="$SFT_OUTPUT_DIR/train.parquet" \
    data.val_files="$SFT_OUTPUT_DIR/test.parquet" \
    data.micro_batch_size_per_gpu=1 \
    data.messages_key=messages \
    data.ignore_input_ids_mismatch=True \
    data.max_token_len_per_gpu=16384 \
    data.pad_mode=no_padding \
    data.max_length=8192 \
    data.train_batch_size=8 \
    data.truncation=error \
    optim.lr=1e-5 \
    engine=fsdp \
    model.path="$MODEL_PATH" \
    "${TEMPLATE_ARGS[@]}" \
    "${THINKING_ARGS[@]}" \
    model.enable_gradient_checkpointing=True \
    model.use_remove_padding=True \
    model.ccr.enable="$CCR_ENABLE" \
    model.ccr.start_epoch="$SFT_CCR_START" \
    model.ccr.alpha="$CCR_ALPHA" \
    model.ccr.loss_type=ce \
    model.ccr.kl_penalty=kl \
    model.ccr.fmt_token_ids="$SFT_FMT_IDS" \
    model.ccr.only_harmful=true \
    trainer.project_name=deshortcut-align \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    trainer.n_gpus_per_node="$N_GPUS" \
    trainer.nnodes=1 \
    trainer.logger='["console","swanlab"]' \
    trainer.total_epochs=5 \
    trainer.save_freq=-1 \
    trainer.test_freq=10 \
    checkpoint.save_contents='[model]' \
    trainer.resume_mode=disable \
    "$@"
