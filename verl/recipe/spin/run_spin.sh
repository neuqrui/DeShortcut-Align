set -e
set -x
VISIBLE_DEVICES="4,5,6,7"
export HYDRA_FULL_ERROR=1
export HOME="${HOME:-$PWD}"

safety_train_path=$HOME/data/custom_safety/train.parquet
safety_test_path=$HOME/data/custom_safety/test.parquet

train_files="['$safety_train_path']"
test_files="['$safety_test_path']"
model="${MODEL_PATH:-deepseek-ai/DeepSeek-R1-Distill-Qwen-7B}"
EXPERIMENT_NAME="safe-dpo"

CURRENT_TIME=$(date +"%Y%m%d_%H%M%S")
export EXPERIMENT_LOG_DIR="$HOME/experiment_logs/dpo_${EXPERIMENT_NAME}_${CURRENT_TIME}"
mkdir -p $EXPERIMENT_LOG_DIR
reward_path=$HOME/reward/async_safe_reward.py

CUDA_VISIBLE_DEVICES=${VISIBLE_DEVICES} python3 -m recipe.spin.main_spin \
  data.train_files="$train_files" \
  data.val_files="$test_files" \
  data.train_batch_size=8 \
  data.max_prompt_length=1024 \
  data.max_response_length=4096 \
  actor_rollout_ref.model.path=$model \
  actor_rollout_ref.actor.optim.lr=5e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size=8 \
  actor_rollout_ref.actor.ppo_micro_batch_size=8 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size=8 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
  actor_rollout_ref.rollout.n=4 \
  reward.custom_reward_function.path=$reward_path \
  reward.custom_reward_function.name="compute_score" \
  reward_model.reward_manager=naive \
  actor_rollout_ref.ref.log_prob_micro_batch_size=8 \
  algorithm.kl_ctrl.kl_coef=0.001 \
  trainer.logger=console \
  trainer.val_before_train=True \
  trainer.n_gpus_per_node=4 \
  trainer.nnodes=1 \
  trainer.save_freq=40 \
  trainer.test_freq=10 \
  trainer.project_name='saver' \
  trainer.experiment_name=$experiment_name \
  +trainer.log_freq=1 \
  trainer.ref_update_freq=1 \
  trainer.total_epochs=2 2>&1 | tee verl_demo.log