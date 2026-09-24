"""Stage 1: Token Sensitivity Score (TSS).

For each prompt token, mask it and measure the absolute teacher-forced log-likelihood
shift on the refusal answer tokens. The per-mode score stored as ``eta`` is TSS_m.
Scores are averaged across template modes A/B/C and written as ``tss`` / ``tss_token``.
"""

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from verl.trainer.ppo.core_algos import agg_loss, kl_penalty
except ImportError:
    repo_verl = Path(__file__).resolve().parents[2] / "verl"
    if str(repo_verl) not in sys.path:
        sys.path.insert(0, str(repo_verl))
    from verl.trainer.ppo.core_algos import agg_loss, kl_penalty


def load_causal_data(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, dict) and "data" in obj:
        return obj["data"]
    if isinstance(obj, list):
        return obj
    raise ValueError(f"Unsupported dataset format: {path}")


def answer_char_range_in_full_text(prompt: str, response: str) -> Optional[Tuple[int, int]]:
    if not response or "</think>" not in response:
        return None
    parts = response.split("</think>", 1)
    answer_body = parts[1].strip()
    if not answer_body:
        return None
    full = prompt + response
    start = full.find(answer_body)
    if start == -1:
        start = len(prompt) + len(parts[0]) + len("</think>")
        end = len(full)
    else:
        end = start + len(answer_body)
    return (start, end)


def instruction_char_range_in_prompt(prompt: str, instruction: str) -> Optional[Tuple[int, int]]:
    if not instruction:
        return None
    idx = prompt.find(instruction)
    if idx < 0:
        ins = instruction.strip()
        idx = prompt.find(ins)
        if idx < 0:
            return None
        instruction = ins
    return (idx, idx + len(instruction))


def token_indices_overlapping_range(offsets: List[Tuple[int, int]], range_start: int, range_end: int) -> List[int]:
    out = []
    for i, (s, e) in enumerate(offsets):
        if s is None or e is None or s == e:
            continue
        if s < range_end and e > range_start:
            out.append(i)
    return out


def get_mask_token_id(tokenizer) -> int:
    for token_id in (tokenizer.mask_token_id, tokenizer.unk_token_id, tokenizer.pad_token_id, tokenizer.eos_token_id):
        if token_id is not None:
            return int(token_id)
    raise ValueError("Tokenizer does not provide usable mask/unk/pad/eos token id.")


def gather_answer_log_probs(logits: torch.Tensor, input_ids: torch.Tensor, answer_token_indices: List[int]) -> torch.Tensor:
    if not answer_token_indices:
        return torch.zeros(1, 0, device=logits.device, dtype=logits.dtype)
    device = logits.device
    t = torch.tensor(answer_token_indices, device=device, dtype=torch.long)
    pred_pos = t - 1
    tgt = input_ids[0, t]
    # Memory-safe log p(y): logits_y - logsumexp(logits)
    # Avoid materializing full log_softmax tensor of shape [B, T, V].
    row_logits = logits[0, pred_pos, :]
    row_lse = torch.logsumexp(row_logits.float(), dim=-1)
    row_tgt_logits = row_logits.gather(dim=-1, index=tgt.unsqueeze(-1)).squeeze(-1).float()
    sel = (row_tgt_logits - row_lse).unsqueeze(0)
    return sel


def gather_answer_log_probs_batch(logits: torch.Tensor, input_ids: torch.Tensor, answer_token_indices: List[int]) -> torch.Tensor:
    if not answer_token_indices:
        return torch.zeros(logits.shape[0], 0, device=logits.device, dtype=logits.dtype)
    device = logits.device
    t = torch.tensor(answer_token_indices, device=device, dtype=torch.long)
    pred_pos = t - 1
    tgt = input_ids[0, t]
    batch_size = logits.shape[0]
    # Memory-safe batched log p(y): logits_y - logsumexp(logits)
    # Keeps peak memory much lower than full log_softmax over vocabulary.
    sel_logits = logits[:, pred_pos, :]  # [B, A, V]
    lse = torch.logsumexp(sel_logits.float(), dim=-1)  # [B, A]
    tgt_expand = tgt.unsqueeze(0).expand(batch_size, -1).unsqueeze(-1)  # [B, A, 1]
    tgt_logits = sel_logits.gather(dim=-1, index=tgt_expand).squeeze(-1).float()  # [B, A]
    return tgt_logits - lse


def compute_eta_for_masked_vs_ref(masked_lp: torch.Tensor, ref_lp: torch.Tensor, kl_penalty_type: str, loss_agg_mode: str) -> torch.Tensor:
    kl_mat = kl_penalty(logprob=masked_lp, ref_logprob=ref_lp, kl_penalty=kl_penalty_type)
    mask = torch.ones_like(kl_mat, dtype=kl_mat.dtype)
    return agg_loss(loss_mat=kl_mat, loss_mask=mask, loss_agg_mode=loss_agg_mode)


def _eta_unit_key(entry: dict) -> Optional[Tuple[Tuple[int, ...], str]]:
    """Stable key for a single-token or n-gram TSS unit."""
    ttxt = entry.get("token_text")
    if ttxt is None:
        return None
    text = str(ttxt)
    if not text:
        return None
    tids = entry.get("token_ids")
    if isinstance(tids, list) and tids:
        try:
            id_tuple = tuple(int(x) for x in tids)
        except (TypeError, ValueError):
            return None
        return (id_tuple, text)
    tid = entry.get("token_id")
    if tid is None:
        return None
    try:
        return ((int(tid),), text)
    except (TypeError, ValueError):
        return None


def run_one_mode(
    model,
    tokenizer,
    mask_token_id: int,
    prompt: str,
    response: str,
    instruction: str,
    mask_scope: str,
    kl_penalty_type: str,
    loss_agg_mode: str,
    mask_batch_size: int,
    mask_ngram_size: int = 1,
) -> Tuple[List[dict], int, int]:
    ngram = max(1, int(mask_ngram_size))
    full_text = prompt + response
    enc = tokenizer(
        full_text,
        return_tensors="pt",
        return_offsets_mapping=True,
        add_special_tokens=False,
        truncation=False,
    )
    input_ids = enc["input_ids"].to(model.device)
    offsets = enc["offset_mapping"][0].tolist()

    ans_range = answer_char_range_in_full_text(prompt, response)
    if ans_range is None:
        return [], 0, 0

    if mask_scope == "query":
        query_range = instruction_char_range_in_prompt(prompt, instruction)
        if query_range is None:
            return [], 0, 0
        q_tokens = token_indices_overlapping_range(offsets, query_range[0], query_range[1])
    else:
        q_tokens = token_indices_overlapping_range(offsets, 0, len(prompt))

    raw_answer_tokens = token_indices_overlapping_range(offsets, ans_range[0], ans_range[1])
    answer_tokens = [i for i in raw_answer_tokens if i >= 1]
    if not answer_tokens:
        return [], len(q_tokens), 0

    # Sliding windows of adjacent query tokens: [(i), ...] or [(i,i+1), ...] etc.
    mask_units: List[List[int]] = []
    if len(q_tokens) >= ngram:
        for i in range(len(q_tokens) - ngram + 1):
            mask_units.append(q_tokens[i : i + ngram])
    if not mask_units:
        return [], len(q_tokens), len(answer_tokens)

    with torch.no_grad():
        base_logits = model(input_ids=input_ids).logits
    ref_lp = gather_answer_log_probs(base_logits, input_ids, answer_tokens)
    del base_logits

    eta_list = []
    seq_list = input_ids[0].cpu().tolist()
    seq_len = int(input_ids.shape[1])
    eff_bs = max(1, int(mask_batch_size))
    # Eager attention materializes [B,H,L,L]. Keep the mask batch small on long traces.
    if seq_len >= 6144:
        eff_bs = 1
    elif seq_len >= 3072:
        eff_bs = min(eff_bs, 2)
    for start in range(0, len(mask_units), eff_bs):
        batch_units = mask_units[start : start + eff_bs]
        pert = input_ids.repeat(len(batch_units), 1)
        for row_idx, unit in enumerate(batch_units):
            for token_idx in unit:
                pert[row_idx, token_idx] = mask_token_id
        with torch.no_grad():
            m_logits = model(input_ids=pert).logits
        masked_lps = gather_answer_log_probs_batch(m_logits, input_ids, answer_tokens)
        del m_logits
        if seq_len >= 4096 and torch.cuda.is_available():
            torch.cuda.empty_cache()

        for row_idx, unit in enumerate(batch_units):
            eta = compute_eta_for_masked_vs_ref(
                masked_lps[row_idx : row_idx + 1],
                ref_lp,
                kl_penalty_type,
                loss_agg_mode,
            )
            tok_ids = [int(seq_list[t]) for t in unit]
            tok_txt = tokenizer.decode(tok_ids, skip_special_tokens=False)
            entry = {
                "prompt_token_indices": list(unit),
                "token_ids": tok_ids,
                "token_text": tok_txt,
                "mask_ngram_size": ngram,
                "eta": float(eta.detach().cpu().item()),
            }
            # Keep single-token fields for backward compatibility (ngram=1).
            if ngram == 1:
                entry["prompt_token_index"] = unit[0]
                entry["token_id"] = tok_ids[0]
            eta_list.append(entry)
    return eta_list, len(q_tokens), len(answer_tokens)


def build_instruction_tss_from_modes(modes_out: Dict[str, dict]) -> Dict[str, List]:
    required_modes = ["A", "B", "C"]
    if not all(m in modes_out for m in required_modes):
        return {"tss": [], "tss_token": []}

    per_mode_eta: Dict[str, Dict[Tuple[Tuple[int, ...], str], List[float]]] = {
        m: defaultdict(list) for m in required_modes
    }
    for mode in required_modes:
        eta_list = (modes_out.get(mode) or {}).get("eta_list", [])
        for e in eta_list:
            key = _eta_unit_key(e)
            eta = e.get("eta")
            if key is None or eta is None:
                continue
            per_mode_eta[mode][key].append(float(eta))

    common_keys = set(per_mode_eta["A"].keys()) & set(per_mode_eta["B"].keys()) & set(per_mode_eta["C"].keys())
    if not common_keys:
        return {"tss": [], "tss_token": []}

    rows = []
    for key in common_keys:
        mode_means = [float(np.mean(np.abs(np.array(per_mode_eta[m][key], dtype=np.float64)))) for m in required_modes]
        id_tuple, token_text = key
        row = {
            "token_text": token_text,
            "tss_mean_abc": float(np.mean(np.array(mode_means, dtype=np.float64))),
        }
        if len(id_tuple) == 1:
            row["token_id"] = id_tuple[0]
        else:
            row["token_ids"] = list(id_tuple)
        rows.append(row)
    rows.sort(key=lambda x: abs(x["tss_mean_abc"]), reverse=True)
    return {"tss": [r["tss_mean_abc"] for r in rows], "tss_token": [r["token_text"] for r in rows]}


def load_causal_lm(model_path: str):
    """Prefer flash_attention_2 so long traces do not materialize a full attention matrix."""
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    kwargs = {
        "torch_dtype": dtype,
        "device_map": "auto" if torch.cuda.is_available() else None,
    }
    last_err: Optional[Exception] = None
    for impl in ("flash_attention_2", "sdpa", "eager"):
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_path, attn_implementation=impl, **kwargs
            )
            print(f"[phase2] loaded {model_path} attn_implementation={impl}", flush=True)
            return model
        except Exception as e:
            last_err = e
            print(f"[phase2] attn_implementation={impl} failed: {e}", flush=True)
    raise RuntimeError(f"failed to load {model_path}") from last_err


def save_mode_bar_plots(per_sample: List[dict], modes: List[str], out_dir: Path, min_count: int = 10) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    for mode in modes:
        agg = defaultdict(list)
        for item in per_sample:
            mode_obj = (item.get("modes") or {}).get(mode, {})
            for e in mode_obj.get("eta_list", []):
                tok = e.get("token_text")
                eta = e.get("eta")
                if tok is None or eta is None:
                    continue
                agg[str(tok)].append(float(eta))
        rows = [
            {
                "token_text": tok,
                "eta_mean": float(np.mean(np.abs(np.array(vals, dtype=np.float64)))),
                "count": int(len(vals)),
            }
            for tok, vals in agg.items()
            if vals and len(vals) >= min_count
        ]
        rows.sort(key=lambda x: abs(x["eta_mean"]), reverse=True)
        rows = rows[:30]
        if not rows:
            continue
        labels = [r["token_text"].replace("\n", "\\n")[:24] for r in rows]
        vals = [abs(r["eta_mean"]) for r in rows]
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(range(len(vals)), vals)
        ax.set_xticks(list(range(len(vals))))
        ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=8)
        ax.set_ylabel("Mean |TSS|")
        ax.set_title(f"Top-30 mean token |TSS| (count>={min_count}) - mode {mode}")
        fig.tight_layout()
        fig.savefig(out_dir / f"phase2_mode_{mode}_top30_eta_bar.png", dpi=150)
        plt.close(fig)


def run_tss(
    input_path: str,
    model_path: str,
    output_dir: str,
    mode: str = "all",
    mask_scope: str = "prompt",
    only_safe_anchor: bool = False,
    max_samples: int = -1,
    kl_penalty_type: str = "kl",
    loss_agg_mode: str = "token-mean",
    plot_mode: str = "sorted",
    mask_batch_size: int = 32,
    mask_ngram_size: int = 1,
) -> Dict[str, str]:
    ngram = max(1, int(mask_ngram_size))
    scope_suffix = f"_{mask_scope}"
    if ngram > 1:
        scope_suffix = f"{scope_suffix}_ngram{ngram}"
    out_dir = Path(output_dir)
    if not str(out_dir).endswith(scope_suffix):
        out_dir = Path(f"{out_dir}{scope_suffix}")
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_causal_data(Path(input_path))
    if only_safe_anchor:
        data = [x for x in data if x.get("all_safe_anchor", False)]
    if max_samples > 0:
        data = data[: max_samples]

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    model = load_causal_lm(model_path)
    model.eval()
    mask_token_id = get_mask_token_id(tokenizer)
    modes = ["A", "B", "C"] if mode == "all" else [mode]

    per_sample: List[dict] = []
    phase3_ready_data: List[dict] = []

    for item in tqdm(data, desc="phase2_eta"):
        instruction = (item.get("instruction") or "").strip()
        sid = item.get("id")
        modes_out: Dict[str, dict] = {}
        for m in modes:
            sample = (item.get("samples") or {}).get(m, {})
            prompt = sample.get("prompt") or ""
            response = sample.get("response") or ""
            if not prompt or not response:
                modes_out[m] = {"eta_list": [], "n_prompt_tokens": 0, "n_answer_tokens": 0}
                continue
            eta_list, nq, na = run_one_mode(
                model=model,
                tokenizer=tokenizer,
                mask_token_id=mask_token_id,
                prompt=prompt,
                response=response,
                instruction=instruction,
                mask_scope=mask_scope,
                kl_penalty_type=kl_penalty_type,
                loss_agg_mode=loss_agg_mode,
                mask_batch_size=mask_batch_size,
                mask_ngram_size=ngram,
            )
            modes_out[m] = {"eta_list": eta_list, "n_prompt_tokens": nq, "n_answer_tokens": na}

        per_sample.append({"id": sid, "instruction": instruction, "modes": modes_out})
        instruction_tss_obj = build_instruction_tss_from_modes(modes_out)
        phase3_ready_data.append(
            {
                "id": sid,
                "instruction": instruction,
                "mask_ngram_size": ngram,
                "tss": instruction_tss_obj["tss"],
                "tss_token": instruction_tss_obj["tss_token"],
            }
        )

    per_sample_path = out_dir / "per_sample_eta.json"
    tss_path = out_dir / "tss.json"
    run_meta_path = out_dir / "run_meta.json"

    with per_sample_path.open("w", encoding="utf-8") as f:
        json.dump({"per_sample": per_sample}, f, ensure_ascii=False, indent=2)
    with tss_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "summary": {
                    "num_items": len(phase3_ready_data),
                    "items_with_tss": int(sum(1 for x in phase3_ready_data if len(x.get("tss", [])) > 0)),
                },
                "data": phase3_ready_data,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    with run_meta_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "input": input_path,
                "model": model_path,
                "modes": modes,
                "num_items": len(per_sample),
                "kl_penalty": kl_penalty_type,
                "loss_agg_mode": loss_agg_mode,
                "mask_scope": mask_scope,
                "mask_ngram_size": ngram,
                "plot_mode": plot_mode,
                "mask_batch_size": mask_batch_size,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    save_mode_bar_plots(per_sample=per_sample, modes=modes, out_dir=out_dir, min_count=10)

    return {
        "phase2_output_dir": str(out_dir),
        "tss_path": str(tss_path),
        "per_sample_eta_path": str(per_sample_path),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage 1: Token Sensitivity Score (TSS) under template modes A/B/C.")
    p.add_argument("--input", type=str, required=True)
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--mode", type=str, default="all", choices=["A", "B", "C", "all"])
    p.add_argument("--mask_scope", type=str, default="prompt", choices=["query", "prompt"])
    p.add_argument("--only_safe_anchor", action="store_true")
    p.add_argument("--max_samples", type=int, default=-1)
    p.add_argument("--kl_penalty", type=str, default="kl")
    p.add_argument("--loss_agg_mode", type=str, default="token-mean")
    p.add_argument("--plot_mode", type=str, default="sorted")
    p.add_argument("--mask_batch_size", type=int, default=32)
    p.add_argument(
        "--mask_ngram_size",
        type=int,
        default=1,
        help="Mask this many adjacent query tokens together (1=single token, 2=bigram, 3=trigram).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = run_tss(
        input_path=args.input,
        model_path=args.model,
        output_dir=args.output_dir,
        mode=args.mode,
        mask_scope=args.mask_scope,
        only_safe_anchor=args.only_safe_anchor,
        max_samples=args.max_samples,
        kl_penalty_type=args.kl_penalty,
        loss_agg_mode=args.loss_agg_mode,
        plot_mode=args.plot_mode,
        mask_batch_size=args.mask_batch_size,
        mask_ngram_size=args.mask_ngram_size,
    )
    print(f"TSS done: {out['tss_path']}")


if __name__ == "__main__":
    main()
