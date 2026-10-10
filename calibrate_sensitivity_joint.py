"""Calibrate one modality-agnostic EpiCache-style layer profile for LLaVA.

Unlike ``calibrate_sensitivity.py``, this script does not identify image and
text spans when constructing its compression mask. It compares a normal
causal prefill with a sink-plus-recent block-prefill simulation over the whole
expanded multimodal sequence and writes one ``joint`` sensitivity per layer.

Example:
    python calibrate_sensitivity_joint.py \
        --model-path /lustre/hdd/LAS/jannesar-lab/tmahmud/models/llava-v1.5-7b \
        --data-path ./ConvBench/convbench.json \
        --image-path ./ConvBench/visit_bench_images \
        --calib-offset 100 --calib-samples 16 --probe-ratio 0.8 \
        --out sensitivity_joint_llava7b_r08.json
"""

import argparse
import json
import os

import torch

from calibrate_sensitivity import (
    forced_attention_mask,
    layer_key_similarity,
    load_image,
    tokenize_conversation_prompt,
)
from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.mm_utils import get_model_name_from_path, process_images
from llava.model.builder import load_pretrained_model


def choose_conversation_mode(model_name, requested_mode):
    """Use the same conversation-mode inference as the final evaluator."""
    lowered = model_name.lower()
    if "llama-2" in lowered:
        inferred = "llava_llama_2"
    elif "v1" in lowered:
        inferred = "llava_v1"
    elif "mpt" in lowered:
        inferred = "mpt"
    else:
        inferred = "llava_v0"
    return requested_mode or inferred


def context_limit(model, tokenizer, requested_limit=None):
    """Return the nominal context window used to validate calibration prompts."""
    if requested_limit is not None:
        if requested_limit <= 0:
            raise ValueError("--max-sequence-length must be positive")
        return int(requested_limit)

    candidates = []
    configurations = [getattr(model, "config", None)]
    if hasattr(model, "get_model"):
        configurations.append(getattr(model.get_model(), "config", None))
    for configuration in configurations:
        if configuration is None:
            continue
        for name in ("max_position_embeddings", "max_sequence_length", "max_seq_len"):
            value = getattr(configuration, name, None)
            if isinstance(value, int) and 0 < value < 1_000_000:
                candidates.append(value)

    tokenizer_limit = getattr(tokenizer, "model_max_length", None)
    if isinstance(tokenizer_limit, int) and 0 < tokenizer_limit < 1_000_000:
        candidates.append(tokenizer_limit)

    # LLaVA-1.5-7B uses a 2,048-token LLaMA context. Keep this conservative
    # fallback rather than silently sending an unchecked long prompt.
    return min(candidates) if candidates else 2048


def visual_token_count(model):
    """Return the number of visual tokens replacing one legacy <image> token."""
    if not hasattr(model, "get_vision_tower"):
        raise RuntimeError("The loaded model does not expose a LLaVA vision tower")
    tower = model.get_vision_tower()
    count = getattr(tower, "num_patches", None)
    if count is None:
        configuration = getattr(tower, "config", None)
        image_size = getattr(configuration, "image_size", None)
        patch_size = getattr(configuration, "patch_size", None)
        if image_size is not None and patch_size is not None:
            count = (int(image_size) // int(patch_size)) ** 2
    if count is None or int(count) <= 0:
        raise RuntimeError("Could not determine the expanded LLaVA visual-token count")
    return int(count)


def expanded_sequence_length(input_ids, image_tokens):
    """Compute length after each legacy image placeholder expands to patches."""
    raw_length = int(input_ids.shape[1])
    placeholders = int((input_ids == IMAGE_TOKEN_INDEX).sum().item())
    return raw_length + placeholders * (int(image_tokens) - 1)


def block_prefill_mask(length, keep_count, sink_size, device, dtype):
    """Build a causal mask retaining only sink and recent keys per query."""
    keep_count = min(max(1, int(keep_count)), int(length))
    sink_size = min(max(0, int(sink_size)), keep_count)
    blocked = torch.finfo(dtype).min
    mask = torch.full((length, length), blocked, dtype=dtype, device=device)

    for query in range(length):
        available = query + 1
        if available <= keep_count:
            mask[query, :available] = 0
            continue

        sink_count = min(sink_size, keep_count, available)
        recent_count = keep_count - sink_count
        if sink_count:
            mask[query, :sink_count] = 0
        if recent_count:
            recent_start = query - recent_count + 1
            mask[query, recent_start : query + 1] = 0

    return mask.view(1, 1, length, length)


def joint_sensitivity(model, input_ids, image_tensor, probe_ratio, sink_size):
    """Return per-layer cosine similarity for full versus joint compression."""
    with torch.no_grad():
        full = model(
            input_ids=input_ids,
            images=image_tensor,
            past_key_values=None,
            use_cache=True,
        )

    past_full = full.past_key_values
    length = past_full[0][0].size(2)
    device = past_full[0][0].device
    dtype = past_full[0][0].dtype
    keep_count = max(1, int(length * (1.0 - probe_ratio)))
    mask = block_prefill_mask(
        length, keep_count, sink_size, device=device, dtype=dtype
    )

    with torch.no_grad(), forced_attention_mask(model, mask):
        compressed = model(
            input_ids=input_ids,
            images=image_tensor,
            past_key_values=None,
            use_cache=True,
        )

    positions = torch.arange(length, device=device)
    return layer_key_similarity(
        past_full, compressed.past_key_values, positions
    )


def profile_stability(rows):
    """Return per-layer standard deviation and mean pairwise Spearman score."""
    per_layer_std = rows.std(dim=0, unbiased=False).tolist()
    if rows.shape[0] < 2:
        return per_layer_std, None
    ranks = torch.argsort(torch.argsort(rows, dim=1), dim=1).double()
    ranks -= ranks.mean(dim=1, keepdim=True)
    ranks /= ranks.norm(dim=1, keepdim=True).clamp_min(1e-12)
    correlations = ranks @ ranks.transpose(0, 1)
    upper = torch.triu(torch.ones_like(correlations, dtype=torch.bool), diagonal=1)
    return per_layer_std, float(correlations[upper].mean().item())


def main(args):
    if args.calib_samples <= 0:
        raise ValueError("--calib-samples must be positive")
    if args.calib_offset < 0:
        raise ValueError("--calib-offset must be non-negative")
    if not 0.0 <= args.probe_ratio < 1.0:
        raise ValueError("--probe-ratio must satisfy 0 <= ratio < 1")
    if args.probe_sink_size < 0:
        raise ValueError("--probe-sink-size must be non-negative")

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, _ = load_pretrained_model(
        args.model_path,
        args.model_base,
        model_name,
        args.load_8bit,
        args.load_4bit,
        device=args.device,
    )
    model.eval()
    conv_mode = choose_conversation_mode(model_name, args.conv_mode)
    max_sequence_length = context_limit(
        model, tokenizer, requested_limit=args.max_sequence_length
    )
    image_tokens = visual_token_count(model)

    # Hugging Face warns as soon as the unexpanded text is tokenized. We perform
    # the stricter expanded-length check below, so suppress only that redundant
    # tokenizer warning; no prompt is sent to the model before our own check.
    tokenizer.model_max_length = max(
        int(getattr(tokenizer, "model_max_length", 0)), 1_000_000_000
    )
    print(
        "Joint calibration length guard: "
        f"max expanded length={max_sequence_length}, "
        f"visual tokens per image={image_tokens}"
    )

    with open(args.data_path, "r") as handle:
        data = json.load(handle)

    similarities = []
    skipped_oversized = 0
    for item_index, item in enumerate(data):
        if item_index < args.calib_offset:
            continue
        if len(similarities) >= args.calib_samples:
            break
        if len(item.get("turns", [])) < 2:
            continue

        image_file = os.path.join(args.image_path, item["image"])
        if not os.path.exists(image_file):
            continue
        image = load_image(image_file)
        image_tensor = process_images([image], image_processor, args)
        image_tensor = image_tensor.to(
            model.device,
            dtype=getattr(model, "dtype", next(model.parameters()).dtype),
        )

        conv = conv_templates[conv_mode].copy()
        question_one = item["turns"][0]["question"]
        if DEFAULT_IMAGE_TOKEN not in question_one:
            question_one += "\n" + DEFAULT_IMAGE_TOKEN
        conv.append_message(conv.roles[0], question_one)
        conv.append_message(conv.roles[1], item["turns"][0]["answer"])
        conv.append_message(conv.roles[0], item["turns"][1]["question"])
        conv.append_message(conv.roles[1], item["turns"][1]["answer"])
        input_ids = tokenize_conversation_prompt(
            conv.get_prompt(), tokenizer, model.device
        )
        expanded_length = expanded_sequence_length(input_ids, image_tokens)
        if expanded_length > max_sequence_length:
            skipped_oversized += 1
            print(
                f"Skipping calibration item {item_index}: expanded sequence "
                f"length {expanded_length} exceeds {max_sequence_length}."
            )
            del image_tensor, input_ids
            if str(args.device).startswith("cuda"):
                torch.cuda.empty_cache()
            continue

        similarities.append(
            joint_sensitivity(
                model,
                input_ids,
                image_tensor,
                args.probe_ratio,
                args.probe_sink_size,
            )
        )
        del image_tensor
        if str(args.device).startswith("cuda"):
            torch.cuda.empty_cache()

    if not similarities:
        raise RuntimeError(
            "No usable two-turn calibration conversations with existing images."
        )
    if len(similarities) < args.calib_samples:
        raise RuntimeError(
            f"Only {len(similarities)} valid calibration conversations were found "
            f"after offset {args.calib_offset}; requested {args.calib_samples}. "
            f"Skipped {skipped_oversized} oversized conversations."
        )

    similarity_rows = torch.tensor(similarities, dtype=torch.float64)
    sigma_joint = similarity_rows.mean(dim=0)
    sensitivity_rows = (1.0 - similarity_rows).clamp_min(0.0)
    joint = sensitivity_rows.mean(dim=0)
    joint_std, rank_correlation = profile_stability(sensitivity_rows)

    profile = {
        "model": model_name,
        "layer_num": int(joint.numel()),
        "calib_samples_joint": len(similarities),
        "calib_offset": int(args.calib_offset),
        "probe_ratio": float(args.probe_ratio),
        "probe_sink_size": int(args.probe_sink_size),
        "max_expanded_sequence_length": int(max_sequence_length),
        "skipped_oversized": int(skipped_oversized),
        "probe": "modality_agnostic_sink_plus_recent_block_prefill",
        "sigma_joint": sigma_joint.tolist(),
        "joint": joint.tolist(),
        "joint_std": joint_std,
        "joint_mean_pairwise_spearman": rank_correlation,
    }
    with open(args.out, "w") as handle:
        json.dump(profile, handle, indent=2)

    print(
        f"Wrote {args.out}: {len(similarities)} samples, "
        f"{joint.numel()} layers"
    )
    print(f"Skipped oversized calibration items: {skipped_oversized}")
    print(f"Mean pairwise Spearman: {rank_correlation}")
    print(f"Most sensitive layers: {torch.argsort(joint, descending=True)[:5].tolist()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--image-path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--image-aspect-ratio", type=str, default="pad")
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--calib-samples", type=int, default=16)
    parser.add_argument("--calib-offset", type=int, default=100)
    parser.add_argument("--probe-ratio", type=float, default=0.8)
    parser.add_argument("--probe-sink-size", type=int, default=4)
    parser.add_argument(
        "--max-sequence-length",
        type=int,
        default=None,
        help=(
            "Maximum expanded multimodal length. By default this is inferred "
            "from the model/tokenizer (2,048 for LLaVA-1.5-7B). Oversized "
            "calibration conversations are skipped, never truncated."
        ),
    )
    parser.add_argument(
        "--out", type=str, default="sensitivity_joint_llava7b_r08.json"
    )
    main(parser.parse_args())