# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import torch
from tensordict import TensorDict

from verl.trainer.ppo.core_algos import agg_loss, compute_value_loss, get_policy_loss_fn, kl_penalty, kl_penalty_forward
from verl.utils import tensordict_utils as tu
from verl.utils.dataset.dataset_utils import DatasetPadMode
from verl.utils.metric import AggregationType, Metric
from verl.utils.tensordict_utils import unwrap_non_tensor_data
from verl.utils.torch_functional import masked_mean, masked_sum
from verl.workers.config import ActorConfig, CriticConfig
from verl.workers.utils.padding import no_padding_2_padding


def _harmful_flags_from_batch(data: TensorDict, key: str) -> list[bool] | None:
    if key not in data.keys():
        return None
    input_nested = data["input_ids"]
    try:
        bsz = int(input_nested.shape[0])
    except Exception:
        bsz = len(input_nested.unbind())
    stack = data[key]
    flags: list[bool] = []
    for i in range(bsz):
        cell = stack[i] if hasattr(stack, "__getitem__") else stack
        raw = unwrap_non_tensor_data(cell)
        if raw is None:
            flags.append(True)
            continue
        s = str(raw).strip().lower()
        flags.append(s == "harmful")
    return flags


def _flat_harmful_token_weight(
    input_ids_nested: torch.Tensor,
    per_sample_harmful: list[bool],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    lengths = input_ids_nested.offsets().diff().to(dtype=torch.long)
    n = int(lengths.shape[0])
    if n != len(per_sample_harmful):
        raise ValueError(
            f"data_source batch ({len(per_sample_harmful)}) != nested batch ({n}) for CCR mask"
        )
    parts: list[torch.Tensor] = []
    for i in range(n):
        L = int(lengths[i].item())
        w = 1.0 if per_sample_harmful[i] else 0.0
        parts.append(torch.full((L,), w, device=device, dtype=dtype))
    return torch.cat(parts, dim=0)
def sft_loss(config: ActorConfig, model_output, data: TensorDict, dp_group=None, **extra):
    model_output_ccr = extra.pop("model_output_ccr", None)
    ccr_alpha = float(extra.pop("ccr_alpha", 0.0))
    ccr_loss_type = str(extra.pop("ccr_loss_type", "ce")).lower()
    ccr_kl_penalty = str(extra.pop("ccr_kl_penalty", "kl"))
    ccr_only_harmful = bool(extra.pop("ccr_only_harmful", False))
    ccr_data_source_key = str(extra.pop("ccr_data_source_key", "data_source"))

    pad_mode = tu.get_non_tensor_data(data=data, key="pad_mode", default=DatasetPadMode.NO_PADDING)
    dp_size = data["dp_size"]
    batch_num_tokens = data["batch_num_tokens"]

    log_prob = model_output["log_probs"]

    if pad_mode == DatasetPadMode.NO_PADDING:
        loss_mask = data["loss_mask"]

        log_prob_flatten = log_prob.values()
        loss_mask_flatten = loss_mask.values()

        loss_mask_flatten = torch.roll(loss_mask_flatten, shifts=-1, dims=0)

        loss = -masked_sum(log_prob_flatten, loss_mask_flatten) / batch_num_tokens * dp_size
        # Standard supervised CE on the factual input. Cast to float32 before .item().
        ce_loss_val = float(loss.detach().float().item())

        metrics: dict = {"ce_loss": ce_loss_val}
        if model_output_ccr is not None and ccr_alpha > 0.0:
            log_cf_flat = model_output_ccr["log_probs"].values()
            cf_mask = loss_mask_flatten
            if ccr_only_harmful:
                flags = _harmful_flags_from_batch(data, ccr_data_source_key)
                if flags is not None:
                    flat_w = _flat_harmful_token_weight(
                        data["input_ids"],
                        flags,
                        device=log_prob_flatten.device,
                        dtype=loss_mask_flatten.dtype,
                    )
                    flat_w = torch.roll(flat_w, shifts=-1, dims=0)
                    cf_mask = loss_mask_flatten * flat_w
            if ccr_loss_type == "ce":
                cf_loss = -masked_sum(log_cf_flat, cf_mask) / batch_num_tokens * dp_size
            elif ccr_loss_type == "kl":
                kl_mat = kl_penalty_forward(log_cf_flat, log_prob_flatten.detach(), ccr_kl_penalty)
                cf_loss = masked_sum(kl_mat, cf_mask) / batch_num_tokens * dp_size
            else:
                raise ValueError(f"Unknown model.ccr.loss_type: {ccr_loss_type} (expected 'ce' or 'kl')")
            loss = loss + ccr_alpha * cf_loss
            metrics["cf_loss"] = float(cf_loss.detach().item())
        return loss, metrics

    if model_output_ccr is not None and ccr_alpha > 0.0:
        raise NotImplementedError("CCR is only implemented for DatasetPadMode.NO_PADDING")
    response_mask = data["response_mask"].to(bool)
    loss = -masked_sum(log_prob, response_mask) / batch_num_tokens * dp_size
    ce_loss_val = float(loss.detach().float().item())
    return loss, {"ce_loss": ce_loss_val}


def ppo_loss(config: ActorConfig, model_output, data: TensorDict, dp_group=None, **kwargs):
    """Computes ppo loss from model output (log_prob, entropy, values, etc. ) and old_log_probs from data."""
    log_prob = no_padding_2_padding(model_output["log_probs"], data)
    entropy = model_output.get("entropy", None)
    if entropy is not None:
        entropy = no_padding_2_padding(entropy, data)

    # global batch info for loss aggregation
    config.global_batch_info["dp_size"] = data["dp_size"]
    config.global_batch_info["batch_num_tokens"] = data["batch_num_tokens"]
    config.global_batch_info["global_batch_size"] = data["global_batch_size"]
    config.global_batch_info["loss_scale_factor"] = config.loss_scale_factor

    # assumes that if any of the global batch info is set, the policy_loss_fn will
    # normalize using dp_size/global_bsz/global_token; in this case, metric aggregation should be SUM
    # to reflect the mean loss over the global batch
    if (
        data["dp_size"] > 1
        or data["batch_num_tokens"] is not None
        or data["global_batch_size"] is not None
        or config.loss_scale_factor is not None
    ):
        metric_aggregation = AggregationType.SUM
    else:
        metric_aggregation = AggregationType.MEAN

    metrics = {}

    response_mask = data["response_mask"].to(bool)
    # compute policy loss
    old_log_prob = data["old_log_probs"]
    advantages = data["advantages"]
    rollout_is_weights = data.get("rollout_is_weights", None)

    loss_agg_mode = config.loss_agg_mode

    loss_mode = config.policy_loss.get("loss_mode", "vanilla")

    policy_loss_fn = get_policy_loss_fn(loss_mode)
    pg_loss, pg_metrics = policy_loss_fn(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        response_mask=response_mask,
        loss_agg_mode=loss_agg_mode,
        config=config,
        rollout_is_weights=rollout_is_weights,
    )

    # AggregationType.MEAN for pg metrics: assumes policy_loss_fn normalizes by local_bsz/local_tokens
    # Ex: in compute_policy_loss_vanilla, pg_metrics are pg_clipfrac, ppo_kl, pg_clipfrac_lower
    pg_metrics = Metric.from_dict(pg_metrics, aggregation=AggregationType.MEAN)

    metrics.update(pg_metrics)
    metrics["actor/pg_loss"] = Metric(value=pg_loss, aggregation=metric_aggregation)
    policy_loss = pg_loss

    # add entropy loss
    if entropy is not None:
        entropy_loss = agg_loss(
            loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode, **config.global_batch_info
        )
        entropy_coeff = config.entropy_coeff
        policy_loss -= entropy_coeff * entropy_loss
        metrics["actor/entropy_loss"] = Metric(value=entropy_loss, aggregation=metric_aggregation)

    # add kl loss
    if config.use_kl_loss:
        ref_log_prob = data["ref_log_prob"]
        # compute kl loss
        kld = kl_penalty(logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=config.kl_loss_type)
        kl_loss = agg_loss(
            loss_mat=kld, loss_mask=response_mask, loss_agg_mode=config.loss_agg_mode, **config.global_batch_info
        )

        policy_loss += kl_loss * config.kl_loss_coef
        metrics["kl_loss"] = Metric(value=kl_loss, aggregation=metric_aggregation)
        metrics["kl_coef"] = config.kl_loss_coef

    return policy_loss, metrics


def value_loss(config: CriticConfig, model_output, data: TensorDict, dp_group=None, **kwargs):
    """value loss

    Args:
        config: CriticConfig
        model_output: model output from the model
        data: the input to the model
        dp_group: data paralle group
        kwargs: ignored; FSDP forward_step may pass CCR keys (model_output_ccr, etc.)

    Returns:
        value loss
    """
    vpreds = no_padding_2_padding(model_output["values"], data)  # (bsz, response_length)

    values = data["values"]
    returns = data["returns"]
    response_mask = data["response_mask"].to(bool)

    vf_loss, vf_clipfrac = compute_value_loss(
        vpreds=vpreds,
        values=values,
        returns=returns,
        response_mask=response_mask,
        cliprange_value=config.cliprange_value,
        loss_agg_mode=config.loss_agg_mode,
    )

    metrics = {}

    metrics.update(
        {
            "critic/vf_loss": vf_loss.detach().item(),
            "critic/vf_clipfrac": vf_clipfrac.detach().item(),
            "critic/vpred_mean": masked_mean(vpreds, response_mask).detach().item(),
        }
    )

    return vf_loss, metrics
