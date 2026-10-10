"""Modality-split per-layer sensitivity calibration for LLaVA.

Why this exists
---------------
EpiCache measures per-layer sensitivity by running two forward passes that
differ only in their attention mask, then comparing the resulting Key states
layer by layer:

    sigma_l = cos( K_l^full , K_l^masked )          s_l = 1 - sigma_l
    M_l     = ( s_l^alpha / sum_j s_j^alpha ) * (L * M)

Their sigma averages over every token, so it answers one question: "how much
does layer l depend on context at all?"  A VLM cache has two kinds of context,
and they are not interchangeable.  This script therefore runs the ablation
TWICE and keeps the two results separate:

    s_l^image : text Key states at layer l, measured after masking the visual
                entries that the probe would prune. How much does layer l
                depend on the removed image context?
    s_l^text  : current-turn text Key states at layer l, measured with the
                pruned subset of PRIOR-TURN text KV masked out. How much does
                layer l depend on the removed conversational history?

EpiCache found its text-only layer profile to be largely model-dependent.  That
cannot be assumed for a VLM, so this script also reports cross-sample profile
stability.  Use held-out calibration items, average their profiles once, and
freeze the result for evaluation.

How the masking works
---------------------
Pass A is an ordinary prefill, so LLaVA expands the single image placeholder
into its visual tokens as usual.  Pass B is the same call with
``_prepare_decoder_attention_mask`` temporarily replaced by a function that
returns a mask we built ourselves.  Expansion happens before the mask is
constructed, so the patched mask is applied to the already-expanded sequence
and nothing about LLaVA's multimodal path has to be reimplemented here.  The
Key states are read straight out of ``outputs.past_key_values``, so no hooks
are needed.

Usage
-----
    python calibrate_sensitivity.py \
        --model-path /lustre/hdd/LAS/jannesar-lab/tmahmud/models/llava-v1.5-7b \
        --data-path ./ConvBench/convbench.json \
        --image-path ./ConvBench/visit_bench_images \
        --calib-offset 100 \
        --calib-samples 16 \
        --probe-ratio 0.8 \
        --out sensitivity_llava7b_r08.json

Then point the cache at the result.  ``--sens-alpha`` is the sensitivity
allocation exponent; it is deliberately separate from ``--alpha``, which is
the existing token-attention weight:

    python eval_ppl_final.py --method turn_conditioned_layerwise_eviction \
        --allocation-signal sensitivity \
        --sensitivity-profile sensitivity_llava7b_r08.json \
        --sens-alpha 2.0 --alpha 1.0 --beta 0.8 --ratio 0.8 ...

Runtime is four prefills per usable two-turn calibration sample (full and
masked passes for each modality) and nothing during evaluation. Re-run
calibration when the model or the probed compression ratio changes;
sensitivity to mild and aggressive compression need not be identical.
"""

import argparse
import contextlib
import json
import os

import torch
from PIL import Image

from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.mm_utils import (
    get_model_name_from_path,
    process_images,
    tokenizer_image_token,
)
from llava.model.builder import load_pretrained_model


# ----------------------------------------------------------------------------
# Small helpers, kept byte-compatible with eval_ppl_final.py's conventions.
# ----------------------------------------------------------------------------
def tokenize_conversation_prompt(prompt, tokenizer, device):
    return tokenizer_image_token(
        prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0).to(device)


def infer_actual_image_span(input_ids, past_key_values, k_seq_dim=2):
    """Locate the expanded visual KV block.

    LLaVA puts ONE placeholder token in input_ids and expands it inside the
    model, so the visual length has to be recovered from the difference
    between the expanded KV length and the text token count.
    """
    image_positions = (input_ids[0] == IMAGE_TOKEN_INDEX).nonzero(as_tuple=False)
    if image_positions.numel() == 0:
        return None, 0
    if image_positions.shape[0] != 1:
        raise ValueError(
            "Sensitivity calibration expects exactly one image placeholder; "
            f"found {image_positions.shape[0]}."
        )
    expanded_seq_len = past_key_values[0][0].size(k_seq_dim)
    text_token_count = input_ids.shape[1] - image_positions.numel()
    image_token_len = int(expanded_seq_len - text_token_count)
    image_start = int(image_positions[0].item())
    if image_token_len <= 0:
        return None, 0
    return image_start, image_token_len


def load_image(path):
    return Image.open(path).convert("RGB")


@contextlib.contextmanager
def forced_attention_mask(model, mask_4d):
    """Temporarily make the language model use ``mask_4d`` verbatim.

    Transformers 4.31's ``LlamaModel._prepare_decoder_attention_mask`` builds
    the causal mask and adds the expanded padding mask.  Replacing it is the
    least invasive way to inject an arbitrary additive mask, and it leaves
    LLaVA's image-expansion path completely untouched.
    """
    language_model = model.get_model()
    original = language_model._prepare_decoder_attention_mask

    def patched(attention_mask, input_shape, inputs_embeds, past_key_values_length):
        query_length = int(input_shape[-1])
        key_length = query_length + int(past_key_values_length)
        expected = (1, 1, query_length, key_length)
        if tuple(mask_4d.shape) != expected:
            raise RuntimeError(
                "Custom sensitivity mask shape does not match LLaVA's expanded "
                f"sequence: mask={tuple(mask_4d.shape)}, expected={expected}."
            )
        return mask_4d.to(dtype=inputs_embeds.dtype, device=inputs_embeds.device)

    language_model._prepare_decoder_attention_mask = patched
    try:
        yield
    finally:
        language_model._prepare_decoder_attention_mask = original


def causal_mask(length, device, dtype):
    """Additive causal mask of shape [1, 1, length, length]."""
    blocked = torch.finfo(dtype).min
    mask = torch.full((length, length), blocked, device=device, dtype=dtype)
    mask = torch.triu(mask, diagonal=1)
    return mask.view(1, 1, length, length)


def block_columns(mask, query_rows, key_columns):
    """Additionally forbid ``query_rows`` from attending to ``key_columns``."""
    blocked = torch.finfo(mask.dtype).min
    out = mask.clone()
    if query_rows.numel() == 0 or key_columns.numel() == 0:
        return out
    rows = query_rows.view(-1, 1)
    cols = key_columns.view(1, -1)
    out[0, 0, rows, cols] = blocked
    return out


def uniformly_retained(indices, keep_count):
    """Choose ``keep_count`` approximately uniform entries from sorted indices."""
    keep_count = min(max(0, int(keep_count)), int(indices.numel()))
    if keep_count == 0:
        return indices[:0]
    if keep_count == indices.numel():
        return indices
    positions = torch.linspace(
        0,
        indices.numel() - 1,
        steps=keep_count,
        device=indices.device,
    ).round().long()
    return indices.index_select(0, positions)


def sink_recent_retained(indices, keep_count, sink_size):
    """Choose sink entries first and use remaining capacity for recent text."""
    keep_count = min(max(0, int(keep_count)), int(indices.numel()))
    if keep_count == 0:
        return indices[:0]
    if keep_count == indices.numel():
        return indices
    sink_count = min(int(sink_size), keep_count, int(indices.numel()))
    recent_count = keep_count - sink_count
    sink = indices[:sink_count]
    recent = indices[-recent_count:] if recent_count else indices[:0]
    return torch.unique(torch.cat((sink, recent)), sorted=True)


def complement(indices, retained):
    """Return values in ``indices`` that are not present in ``retained``."""
    if retained.numel() == 0:
        return indices
    keep = torch.isin(indices, retained)
    return indices[~keep]


def layer_key_similarity(pkv_full, pkv_masked, positions, k_seq_dim=2):
    """Per-layer mean cosine similarity of Key states at ``positions``.

    Cosine similarity is computed independently for every attention head and
    token position, then averaged over heads and positions. This matches the
    EpiCache sensitivity definition rather than concatenating heads first.
    """
    sims = []
    for (k_full, _), (k_masked, _) in zip(pkv_full, pkv_masked):
        a = k_full.float().index_select(k_seq_dim, positions)
        b = k_masked.float().index_select(k_seq_dim, positions)
        similarity = torch.nn.functional.cosine_similarity(a, b, dim=-1).mean()
        sims.append(float(similarity.clamp(min=-1.0, max=1.0).item()))
    return sims


# ----------------------------------------------------------------------------
# The two ablations
# ----------------------------------------------------------------------------
def image_sensitivity(model, tokenizer, base_ids, image_tensor, probe_ratio):
    """Measure text-Key changes after pruning visual context at ``probe_ratio``."""
    with torch.no_grad():
        full = model(
            input_ids=base_ids,
            images=image_tensor,
            past_key_values=None,
            use_cache=True,
        )
    pkv_full = full.past_key_values
    length = pkv_full[0][0].size(2)
    device = pkv_full[0][0].device
    dtype = pkv_full[0][0].dtype

    image_start, image_token_len = infer_actual_image_span(base_ids, pkv_full)
    if image_start is None or image_token_len <= 0:
        return None, None

    image_columns = torch.arange(
        image_start, image_start + image_token_len, device=device
    )
    image_keep_count = max(1, int(image_token_len * (1.0 - probe_ratio)))
    retained_image = uniformly_retained(image_columns, image_keep_count)
    dropped_image = complement(image_columns, retained_image)
    # Text positions AFTER the image; positions before it cannot see the image
    # under a causal mask, so including them would dilute the measurement with
    # guaranteed-zero changes.
    text_positions = torch.arange(image_start + image_token_len, length, device=device)
    if text_positions.numel() == 0:
        return None, None

    mask = causal_mask(length, device, dtype)
    mask = block_columns(mask, text_positions, dropped_image)

    with torch.no_grad(), forced_attention_mask(model, mask):
        masked = model(
            input_ids=base_ids,
            images=image_tensor,
            past_key_values=None,
            use_cache=True,
        )

    sims = layer_key_similarity(pkv_full, masked.past_key_values, text_positions)
    return sims, (image_start, image_token_len)


def text_sensitivity(
    model,
    tokenizer,
    two_turn_ids,
    turn_two_start_text,
    image_tensor,
    image_span,
    probe_ratio,
    sink_size,
):
    """Measure current-turn Key changes after pruning prior-turn text context.

    ``turn_two_start_text`` is the index, in UNEXPANDED token coordinates, of
    the first token belonging to turn 2.  It is shifted into expanded
    coordinates using the visual span measured by the image ablation.
    """
    with torch.no_grad():
        full = model(
            input_ids=two_turn_ids,
            images=image_tensor,
            past_key_values=None,
            use_cache=True,
        )
    pkv_full = full.past_key_values
    length = pkv_full[0][0].size(2)
    device = pkv_full[0][0].device
    dtype = pkv_full[0][0].dtype

    image_start, image_token_len = image_span
    # One placeholder token expanded into image_token_len tokens, so everything
    # after the placeholder shifts by image_token_len - 1.
    shift = image_token_len - 1
    turn_two_start = turn_two_start_text + shift
    if not (0 < turn_two_start < length):
        return None

    current_positions = torch.arange(turn_two_start, length, device=device)
    prior_text = torch.cat(
        (
            torch.arange(0, image_start, device=device),
            torch.arange(image_start + image_token_len, turn_two_start, device=device),
        )
    )
    if current_positions.numel() == 0 or prior_text.numel() == 0:
        return None

    text_keep_count = max(1, int(prior_text.numel() * (1.0 - probe_ratio)))
    retained_prior_text = sink_recent_retained(
        prior_text, text_keep_count, sink_size
    )
    dropped_prior_text = complement(prior_text, retained_prior_text)

    mask = causal_mask(length, device, dtype)
    mask = block_columns(mask, current_positions, dropped_prior_text)

    with torch.no_grad(), forced_attention_mask(model, mask):
        masked = model(
            input_ids=two_turn_ids,
            images=image_tensor,
            past_key_values=None,
            use_cache=True,
        )

    return layer_key_similarity(pkv_full, masked.past_key_values, current_positions)


# ----------------------------------------------------------------------------
def main(args):
    if args.calib_samples <= 0:
        raise ValueError("--calib-samples must be positive")
    if args.calib_offset < 0:
        raise ValueError("--calib-offset must be non-negative")
    if not 0.0 <= args.probe_ratio < 1.0:
        raise ValueError("--probe-ratio must satisfy 0 <= ratio < 1")
    if args.probe_sink_size < 0:
        raise ValueError("--probe-sink-size must be non-negative")

    device = torch.device(args.device)
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
    conv_mode = args.conv_mode or (
        "llava_v1" if "llava-v1.5" in model_name.lower() else "vicuna_v1"
    )

    with open(args.data_path, "r") as handle:
        data = json.load(handle)

    image_sims, text_sims = [], []
    used_image, used_text = 0, 0

    for item_index, item in enumerate(data):
        if item_index < args.calib_offset:
            continue
        if used_image >= args.calib_samples and used_text >= args.calib_samples:
            break
        if "turns" not in item or len(item["turns"]) == 0:
            continue
        image_file = os.path.join(args.image_path, item["image"])
        if not os.path.exists(image_file):
            continue

        image = load_image(image_file)
        image_tensor = process_images([image], image_processor, args)
        image_tensor = image_tensor.to(
            model.device, dtype=getattr(model, "dtype", next(model.parameters()).dtype)
        )

        # ---- completed Turn-1 prompt ---------------------------------------
        # Include the answer so image sensitivity is measured on a meaningful
        # downstream text span. With an empty assistant slot, the only text
        # after this evaluator's image placeholder is usually a short role
        # prefix, which produces a noisy and unrepresentative profile.
        conv_one = conv_templates[conv_mode].copy()
        question = item["turns"][0]["question"]
        if DEFAULT_IMAGE_TOKEN not in question:
            question = question + "\n" + DEFAULT_IMAGE_TOKEN
        conv_one.append_message(conv_one.roles[0], question)
        conv_one.append_message(conv_one.roles[1], item["turns"][0]["answer"])
        completed_turn_one_ids = tokenize_conversation_prompt(
            conv_one.get_prompt(), tokenizer, model.device
        )

        sims, image_span = image_sensitivity(
            model,
            tokenizer,
            completed_turn_one_ids,
            image_tensor,
            args.probe_ratio,
        )
        if sims is not None and used_image < args.calib_samples:
            image_sims.append(sims)
            used_image += 1

        # ---- two-turn prompt for the history ablation ------------------------
        if image_span is not None and len(item["turns"]) >= 2 and used_text < args.calib_samples:
            conv_two = conv_templates[conv_mode].copy()
            conv_two.append_message(conv_two.roles[0], question)
            conv_two.append_message(conv_two.roles[1], item["turns"][0]["answer"])
            committed_ids = tokenize_conversation_prompt(
                conv_two.get_prompt(), tokenizer, model.device
            )
            conv_two.append_message(conv_two.roles[0], item["turns"][1]["question"])
            conv_two.append_message(conv_two.roles[1], None)
            two_turn_ids = tokenize_conversation_prompt(
                conv_two.get_prompt(), tokenizer, model.device
            )
            prefix_length = committed_ids.shape[1]
            if (
                two_turn_ids.shape[1] < prefix_length
                or not torch.equal(two_turn_ids[:, :prefix_length], committed_ids)
            ):
                raise RuntimeError(
                    f"Calibration item {item_index}: conversation tokenization "
                    "is not prefix-stable between Turn 1 and Turn 2."
                )

            sims_text = text_sensitivity(
                model,
                tokenizer,
                two_turn_ids,
                int(committed_ids.shape[1]),
                image_tensor,
                image_span,
                args.probe_ratio,
                args.probe_sink_size,
            )
            if sims_text is not None:
                text_sims.append(sims_text)
                used_text += 1

        del image_tensor
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if not image_sims:
        raise RuntimeError("No usable calibration samples for the image ablation.")
    if not text_sims:
        raise RuntimeError(
            "No usable calibration samples for the history ablation; the dataset "
            "needs at least two turns per conversation."
        )

    layer_num = len(image_sims[0])

    def average(rows):
        return [sum(row[l] for row in rows) / len(rows) for l in range(layer_num)]

    sigma_image = average(image_sims)
    sigma_text = average(text_sims)
    s_image = [max(0.0, 1.0 - x) for x in sigma_image]
    s_text = [max(0.0, 1.0 - x) for x in sigma_text]

    image_sensitivity_rows = torch.tensor(
        [[max(0.0, 1.0 - x) for x in row] for row in image_sims],
        dtype=torch.float64,
    )
    text_sensitivity_rows = torch.tensor(
        [[max(0.0, 1.0 - x) for x in row] for row in text_sims],
        dtype=torch.float64,
    )

    def stability(rows):
        """Return per-layer std and mean pairwise rank correlation."""
        per_layer_std = rows.std(dim=0, unbiased=False).tolist()
        if rows.shape[0] < 2:
            return per_layer_std, None
        ranks = torch.argsort(torch.argsort(rows, dim=1), dim=1).double()
        ranks = ranks - ranks.mean(dim=1, keepdim=True)
        ranks = ranks / ranks.norm(dim=1, keepdim=True).clamp_min(1e-12)
        correlations = ranks @ ranks.transpose(0, 1)
        upper = torch.triu(
            torch.ones_like(correlations, dtype=torch.bool), diagonal=1
        )
        return per_layer_std, float(correlations[upper].mean().item())

    image_std, image_rank_correlation = stability(image_sensitivity_rows)
    text_std, text_rank_correlation = stability(text_sensitivity_rows)

    profile = {
        "model": model_name,
        "layer_num": layer_num,
        "calib_samples_image": len(image_sims),
        "calib_samples_text": len(text_sims),
        "sigma_image": sigma_image,
        "sigma_text": sigma_text,
        "image": s_image,
        "text": s_text,
        "image_std": image_std,
        "text_std": text_std,
        "image_mean_pairwise_spearman": image_rank_correlation,
        "text_mean_pairwise_spearman": text_rank_correlation,
        "calib_offset": int(args.calib_offset),
        "probe_ratio": float(args.probe_ratio),
        "probe_sink_size": int(args.probe_sink_size),
        "image_probe": "completed_turn_1_text_with_uniform_visual_retention",
        "text_probe": "turn_2_text_with_sink_plus_recent_history_retention",
    }
    with open(args.out, "w") as handle:
        json.dump(profile, handle, indent=2)

    print(f"\nwrote {args.out}  ({layer_num} layers, "
          f"{len(image_sims)} image / {len(text_sims)} history samples)\n")
    print(f"{'layer':>5}  {'s_image':>9}  {'s_text':>9}")
    for l in range(layer_num):
        print(f"{l:>5}  {s_image[l]:>9.5f}  {s_text[l]:>9.5f}")

    def summarize(name, values):
        order = sorted(range(layer_num), key=lambda i: values[i], reverse=True)
        spread = max(values) / max(min(values), 1e-9)
        print(f"\n{name}: most sensitive layers {order[:5]}, "
              f"least {order[-5:]}, max/min = {spread:.2f}")
        if spread < 1.5:
            print(f"  WARNING: {name} is nearly flat across layers. A flat "
                  "profile allocates a near-uniform budget, so it cannot "
                  "outperform uniform allocation. Raise --sens-alpha or treat "
                  "this as evidence that layer-wise allocation is not the "
                  "lever for this model.")

    summarize("s_image", s_image)
    summarize("s_text", s_text)
    print(
        "\nprofile stability (mean pairwise Spearman across calibration items):\n"
        f"  image = {image_rank_correlation}\n"
        f"  text  = {text_rank_correlation}\n"
        "Higher values mean the layer ordering is more input-stable."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="./models/llava-v1.5-7b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--image-path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--image-aspect-ratio", type=str, default="pad")
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--calib-samples", type=int, default=16)
    parser.add_argument(
        "--calib-offset",
        type=int,
        default=0,
        help=(
            "Skip this many dataset items before calibration. Use a disjoint "
            "range from the evaluation items when possible."
        ),
    )
    parser.add_argument(
        "--probe-ratio",
        type=float,
        default=0.8,
        help=(
            "Pruning ratio simulated by the masked pass. Match this to the "
            "evaluation --ratio (0.8 means retain 20%%)."
        ),
    )
    parser.add_argument(
        "--probe-sink-size",
        type=int,
        default=4,
        help="Number of earliest prior-text positions retained by the text probe.",
    )
    parser.add_argument("--out", type=str, default="sensitivity_profile.json")
    main(parser.parse_args())
