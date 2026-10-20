"""
Continuous multi-turn ROUGE evaluation for LLaVA + KV-cache pruning.

Expected dataset format (one image, several text turns per item):

[
    {
        "image": "example.jpg",
        "turns": [
            {"question": "Question 1", "answer": "Answer 1"},
            {"question": "Question 2", "answer": "Answer 2"},
            {"question": "Question 3", "answer": "Answer 3"}
        ]
    }
]

The important multi-turn behavior is:

    Turn 1: image + Q1 -> generate A1, score it against the reference
    Turn 2:         Q2 -> generate A2, reusing the KV cache from Turn 1
    Turn 3:         Q3 -> generate A3, reusing the same continuing KV cache

The model freely generates each assistant answer.  Its GENERATED answer (not
the reference) is kept in the continuing conversation and KV cache for the
following turn.  Only the first turn sends the image to LLaVA.  Later turns
send ONLY the newly appended text tokens.

``--max-token-generated`` is a per-turn generation ceiling (default 128).
Pass ``free`` to generate until the conversation stop token or the model's
remaining context capacity.  Values such as 1k, 2k, and 8k are accepted and
mean 1000, 2000, and 8000 tokens respectively.

Layer-budget allocation uses the same options as eval_ppl_final.py.  To use a
frozen profile written by calibrate_sensitivity.py:

    python eval_rouge_final.py --method turn_conditioned_layerwise_eviction \
        --allocation-signal sensitivity \
        --sensitivity-profile sensitivity_llava7b_r08.json \
        --sens-alpha 2.0 --alpha 1.0 --beta 0.8 --ratio 0.8 ...

The evaluator exposes the turn-boundary merge/eviction variants and the TGV-KV,
LocalCache, H2O, and ElasticCache comparison baselines.
"""

import argparse
import json
import os
from io import BytesIO

import requests
import torch
from PIL import Image
from rouge import Rouge
from tqdm import tqdm

# All cache methods used by this evaluator come through one final module.
from kv_cache_final import (
    ElasticCache,
    H2OCache,
    LocalCache,
    TurnConditionedLayerwiseKVCache,
    TurnConditionedLayerwiseEvictionKVCache,
    TGVKVCache,
    MEDACache,
    PyramidKVCache,
    enable_legacy_tgv_attention,
    set_legacy_logical_position_limit,
)
from kv_cache_coverage import TurnConditionedCoverageKVCache
from tgv_kv_legacy_nova import (
    enable_online_attention,
    online_attention,
)

from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import SeparatorStyle, conv_templates
from llava.mm_utils import (
    get_model_name_from_path,
    process_images,
    tokenizer_image_token,
)
from llava.model.builder import load_pretrained_model

def load_image(image_file):
    """Load a local image or URL and return RGB."""
    if image_file.startswith("http://") or image_file.startswith("https://"):
        response = requests.get(image_file, timeout=30)
        response.raise_for_status()
        return Image.open(BytesIO(response.content)).convert("RGB")
    return Image.open(image_file).convert("RGB")


def normalize_text(text, max_words=256):
    """Normalize generated/reference text only for ROUGE scoring."""
    text = (text or "").replace("</s>", "").replace("<AND>", " ").strip()
    return " ".join(text.split()[:max_words])


def parse_generated_token_limit(value):
    """Parse free/none, a positive integer, or a decimal k-token value."""
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"free", "none", "unlimited"}:
        return None

    multiplier = 1000 if text.endswith("k") else 1
    number = text[:-1] if multiplier == 1000 else text
    try:
        parsed = float(number) * multiplier
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Use 'free', a positive integer, or a value such as 1k, 2k, or 8k."
        ) from exc
    if parsed <= 0 or not parsed.is_integer():
        raise argparse.ArgumentTypeError(
            "--max-token-generated must resolve to a positive whole number."
        )
    return int(parsed)


def tokenize_conversation_prompt(prompt, tokenizer, device):
    """
    Tokenize a FULL LLaVA conversation prompt.

    We use tokenizer_image_token even for later turns because the full logical
    conversation still contains the original <image> placeholder.  These full
    token sequences are used only to determine which NEW text tokens were
    appended to the conversation.  We do NOT resend the full sequence to the
    model after Turn 1.
    """
    ids = tokenizer_image_token(
        prompt,
        tokenizer,
        IMAGE_TOKEN_INDEX,
        return_tensors="pt",
    )
    return ids.unsqueeze(0).to(device)


def require_prefix(full_ids, prefix_ids, description):
    """
    Return the suffix of full_ids after prefix_ids.

    Correct KV reuse requires the newly rendered conversation to extend the
    already-processed logical transcript.  If that is not true, silently using
    a string/token suffix would duplicate or skip tokens, so fail loudly.
    """
    prefix_len = prefix_ids.shape[1]

    if full_ids.shape[1] < prefix_len:
        raise RuntimeError(
            f"{description}: new token sequence is shorter than the transcript "
            "already stored in the KV cache."
        )

    if not torch.equal(full_ids[:, :prefix_len], prefix_ids):
        # Report the first mismatch to make tokenizer/template debugging easier.
        lhs = full_ids[:, :prefix_len]
        mismatch = (lhs != prefix_ids).nonzero(as_tuple=False)
        first_bad = int(mismatch[0, 1].item()) if mismatch.numel() else -1
        raise RuntimeError(
            f"{description}: conversation tokenization is not prefix-stable. "
            f"First mismatch at token position {first_bad}."
        )

    return full_ids[:, prefix_len:]


def infer_actual_image_span(input_ids, past_key_values, k_seq_dim=2):
    """
    Infer the ACTUAL image-token span in the expanded LLaVA KV sequence.

    tokenizer_image_token normally puts one IMAGE_TOKEN_INDEX placeholder in
    input_ids.  Inside LLaVA that placeholder is replaced by many visual tokens
    (commonly hundreds).  Therefore counting IMAGE_TOKEN_INDEX occurrences is
    NOT the visual KV length.

    For one image:
        expanded_KV_length
          = non-image text tokens + actual visual tokens

    so:
        actual_visual_tokens
          = expanded_KV_length - (input_length - number_of_placeholders)

    The start of the expanded visual block is the position of the image
    placeholder in the original token sequence.
    """
    image_positions = (input_ids[0] == IMAGE_TOKEN_INDEX).nonzero(as_tuple=True)[0]

    if image_positions.numel() == 0:
        return None, 0

    if image_positions.numel() != 1:
        raise ValueError(
            "This evaluator currently expects exactly one image per item. "
            f"Found {image_positions.numel()} image placeholders."
        )

    expanded_seq_len = past_key_values[0][0].size(k_seq_dim)
    text_token_count = input_ids.shape[1] - image_positions.numel()
    image_token_len = int(expanded_seq_len - text_token_count)
    image_start = int(image_positions[0].item())

    if image_token_len <= 0:
        raise RuntimeError(
            "Could not infer a positive expanded image-token length from LLaVA."
        )

    return image_start, image_token_len


def effective_context_length(model, loader_context_length):
    """Return the model context length used to cap free generation."""
    if isinstance(loader_context_length, (int, float)) and loader_context_length > 0:
        return int(loader_context_length)
    config = getattr(model, "config", None)
    if config is not None:
        for name in (
            "max_position_embeddings",
            "max_sequence_length",
            "tokenizer_model_max_length",
        ):
            value = getattr(config, name, None)
            if isinstance(value, (int, float)) and value > 0:
                return int(value)
    return None


def load_sensitivity_profile(path, mode, expected_layers, evaluation_ratio):
    """Load only the layer-sensitivity fields required by ``mode``."""
    if not path:
        raise ValueError(
            "--sensitivity-profile is required when --allocation-signal "
            "sensitivity is selected"
        )
    with open(path, "r") as handle:
        profile = json.load(handle)

    declared_layers = profile.get("layer_num")
    if declared_layers is not None and int(declared_layers) != expected_layers:
        raise ValueError(
            f"Sensitivity profile declares {declared_layers} layers; "
            f"this model expects {expected_layers}"
        )

    required = {
        "joint": ("joint",),
        "image": ("image",),
        "text": ("text",),
        "split": ("image", "text"),
    }[mode]
    validated = {"joint": None, "image": None, "text": None}
    for name in required:
        values = torch.as_tensor(profile.get(name, []), dtype=torch.float32).flatten()
        if values.numel() != expected_layers:
            raise ValueError(
                f"Sensitivity profile '{name}' has {values.numel()} values; "
                f"expected {expected_layers}"
            )
        if not torch.isfinite(values).all() or (values < 0).any():
            raise ValueError(
                f"Sensitivity profile '{name}' must contain finite, "
                "non-negative values"
            )
        validated[name] = values.tolist()

    probe_ratio = profile.get("probe_ratio")
    if probe_ratio is None:
        print(
            "[WARNING] sensitivity profile has no probe_ratio metadata; "
            "cannot verify that calibration and evaluation ratios match."
        )
    elif abs(float(probe_ratio) - float(evaluation_ratio)) > 1e-8:
        raise ValueError(
            f"Sensitivity profile was calibrated at ratio {probe_ratio}, but "
            f"evaluation requested ratio {evaluation_ratio}. Recalibrate at "
            "the evaluation ratio."
        )

    return validated, profile


def build_cache(
    args,
    model_name,
    image_start=None,
    image_token_len=0,
):
    """Create a fresh cache object for one dataset item/conversation."""
    layer_num = 32 if "7b" in model_name.lower() else 40
    common = dict(
        start_size=args.start_size,
        k_seq_dim=2,
        v_seq_dim=2,
        ratio=args.ratio,
    )
    baseline_common = dict(common, recent_size=args.recent_size)

    if args.method == "elastic_cache":
        return ElasticCache(layer_num=layer_num, **baseline_common)

    if args.method == "turn_conditioned_layerwise_merge":
        return TurnConditionedLayerwiseKVCache(
            layer_num=layer_num,
            image_start=image_start,
            image_token_len=image_token_len,
            attn_weight=args.alpha,
            conn_weight=args.beta,
            cross_turn_weight=args.cross_turn_weight,
            allocation_signal=args.allocation_signal,
            sensitivity_mode=args.sensitivity_mode,
            sensitivity_joint=args.sensitivity_joint,
            sensitivity_image=args.sensitivity_image,
            sensitivity_text=args.sensitivity_text,
            sensitivity_power=args.sens_alpha,
            **common,
        )

    if args.method == "turn_conditioned_layerwise_eviction":
        return TurnConditionedLayerwiseEvictionKVCache(
            layer_num=layer_num,
            image_start=image_start,
            image_token_len=image_token_len,
            attn_weight=args.alpha,
            conn_weight=args.beta,
            cross_turn_weight=args.cross_turn_weight,
            allocation_signal=args.allocation_signal,
            sensitivity_mode=args.sensitivity_mode,
            sensitivity_joint=args.sensitivity_joint,
            sensitivity_image=args.sensitivity_image,
            sensitivity_text=args.sensitivity_text,
            sensitivity_power=args.sens_alpha,
            **common,
        )

    if args.method == "turn_conditioned_coverage":
        return TurnConditionedCoverageKVCache(
            layer_num=layer_num,
            image_start=image_start,
            image_token_len=image_token_len,
            attn_weight=args.alpha,
            conn_weight=args.beta,
            cross_turn_weight=args.cross_turn_weight,
            allocation_signal=args.allocation_signal,
            sensitivity_mode=args.sensitivity_mode,
            sensitivity_joint=args.sensitivity_joint,
            sensitivity_image=args.sensitivity_image,
            sensitivity_text=args.sensitivity_text,
            sensitivity_power=args.sens_alpha,
            coverage_alpha=args.coverage_alpha,
            coverage_kappa=args.coverage_kappa,
            similarity=args.coverage_similarity,
            recent_window=args.coverage_recent_window,
            text_prior_frac=args.coverage_text_prior,
            prefill_compress=not args.coverage_no_prefill,
            relevance=args.coverage_relevance,
            merge_visual=args.coverage_merge,
            layer_floor=args.coverage_layer_floor,
            floor_stage=args.coverage_floor_stage,
            mass_preserving=not args.coverage_no_mass,
            mass_min_sim=args.coverage_mass_min_sim,
            alpha_auto=args.coverage_alpha_auto,
            **common,
        )

    if args.method == "tgv_kv":
        return TGVKVCache(
            layer_num=layer_num,
            image_start=image_start,
            image_token_len=image_token_len,
            start_size=args.start_size,
            k_seq_dim=2,
            v_seq_dim=2,
            ratio=args.ratio,
        )
    if args.method == "meda":
        return MEDACache(
            layer_num=layer_num,
            image_start=image_start,
            image_token_len=image_token_len,
            recent_fraction=args.meda_recent_fraction,
            entropy_mode=args.meda_entropy,
            **common,
        )
    if args.method == "pyramidkv":
        return PyramidKVCache(
            layer_num=layer_num,
            k_seq_dim=2,
            v_seq_dim=2,
            ratio=args.ratio,
            window_size=args.pyramid_window_size,
            kernel_size=args.pyramid_kernel_size,
            pooling=args.pyramid_pooling,
            beta=args.pyramid_beta,
        )

    if args.method == "local_cache":
        return LocalCache(**baseline_common)

    if args.method == "h2o":
        return H2OCache(**baseline_common)

    raise ValueError(f"Invalid method: {args.method}")


def apply_cache(kv_cache, past_key_values, num_of_token, attentions):
    """Apply the selected pruning/merging policy after a model forward pass."""
    if kv_cache is None:
        return past_key_values

    return kv_cache(
        past_key_values,
        num_of_token=num_of_token,
        attentions=attentions,
    )

def prefill_reads_attention(kv_cache):
    """Every current method except LocalCache scores the prefill."""
    return not isinstance(kv_cache, LocalCache)


def decode_reads_attention(kv_cache):
    """Return whether one-token continuation attention is required."""
    # LocalCache never reads attention.
    # ElasticCache scores only the initial prefill.
    return not isinstance(kv_cache, (LocalCache, ElasticCache))

def finalize_turn_cache(kv_cache, past_key_values):
    """Run boundary-only compression for a turn-isolated cache."""
    if kv_cache is None or not hasattr(kv_cache, "finalize_turn"):
        return past_key_values
    result = kv_cache.finalize_turn(past_key_values)
    if isinstance(kv_cache, TurnConditionedLayerwiseKVCache):
        kv_cache.begin_turn(kv_cache.current_turn + 1)
    return result


def forward_one_text_token(model, token_ids, past_key_values, kv_cache, num_of_token):
    """
    Feed exactly ONE text token into the continuing conversation.

    The token is either already known (user turn, template separator) or was
    just generated.  Feeding text one token at a time has two advantages:
      1. it exactly extends the existing KV cache without re-prefilling history;
      2. cache methods that distinguish prefill (query_len > 1) from decode
         (query_len == 1) see subsequent conversation tokens as continuing
         autoregressive tokens.

    Returns:
        last_logits, updated_past_key_values, updated_num_of_token
    """
    if token_ids.shape != (1, 1):
        raise ValueError(
            f"forward_one_text_token expects shape [1, 1], got {tuple(token_ids.shape)}"
        )

    # IMPORTANT: past_key_values may be physically shorter than the logical
    # conversation because the cache has been pruned/merged.  If position_ids
    # is omitted, LLaMA/LLaVA normally derives the next RoPE position from the
    # PHYSICAL past-KV length.  After compression that is wrong.
    #
    # `num_of_token` is the logical expanded conversation length BEFORE this
    # token is appended, so it is exactly the absolute position this token
    # should receive.  Keeping logical positions prevents RoPE position drift
    # across turn boundaries (and after any cache pruning).
    logical_position_ids = torch.tensor(
        [[num_of_token]],
        dtype=torch.long,
        device=token_ids.device,
    )

    # Every compression method may have a physical KV length shorter than the
    # logical conversation position.
    set_legacy_logical_position_limit(model, num_of_token + 1)

    outputs = model(
        input_ids=token_ids,
        position_ids=logical_position_ids,
        past_key_values=past_key_values,
        use_cache=True,
        output_attentions=decode_reads_attention(kv_cache),
    )

    # One logical text token has now been committed.
    num_of_token += outputs.logits.shape[1]
    past_key_values = outputs.past_key_values

    # Let the experimental cache method update scores and/or prune after this token.
    past_key_values = apply_cache(
        kv_cache,
        past_key_values,
        num_of_token,
        outputs.attentions,
    )

    last_logits = outputs.logits[:, -1, :]
    return last_logits, past_key_values, num_of_token


def feed_text_tokens(model, token_ids, past_key_values, kv_cache, num_of_token):
    """
    Commit a sequence of known text tokens to the cache, one token at a time.

    The returned logits are the logits AFTER the final supplied token.  They are
    therefore the logits that predict the next token in the conversation.
    """
    last_logits = None

    for pos in range(token_ids.shape[1]):
        one_token = token_ids[:, pos : pos + 1]
        last_logits, past_key_values, num_of_token = forward_one_text_token(
            model,
            one_token,
            past_key_values,
            kv_cache,
            num_of_token,
        )

    return last_logits, past_key_values, num_of_token


def answer_only_prompt(base_prompt, prompt_with_answer, answer):
    """
    Construct the prompt ending exactly at the answer, before its separator.

    LLaVA conversation templates append formatting after an assistant answer
    (for example a separator or EOS marker).  Those separator tokens must be in
    the KV cache before the next user turn, but they are not part of the
    generated answer that is scored.

    We therefore split the rendered assistant continuation into:
        [answer portion] [template separator portion]
    """
    if not prompt_with_answer.startswith(base_prompt):
        raise RuntimeError(
            "The conversation template did not extend the assistant-empty prompt "
            "when the answer was inserted."
        )

    continuation = prompt_with_answer[len(base_prompt) :]
    answer_pos = continuation.find(answer)

    if answer_pos < 0:
        raise RuntimeError(
            "Could not locate the answer inside the rendered assistant turn."
        )

    # Keep any template-added leading space before the answer, then stop exactly
    # after the answer text.  The remaining text is the turn separator/EOS part.
    answer_end = answer_pos + len(answer)
    return base_prompt + continuation[:answer_end]


def choose_conversation_mode(model_name, requested_mode):
    """Match the original LLaVA conversation-mode selection logic."""
    if "llama-2" in model_name.lower():
        inferred = "llava_llama_2"
    elif "v1" in model_name.lower():
        inferred = "llava_v1"
    elif "mpt" in model_name.lower():
        inferred = "mpt"
    else:
        inferred = "llava_v0"

    if requested_mode is not None and requested_mode != inferred:
        print(
            "[WARNING] auto-inferred conversation mode is {}, while --conv-mode "
            "is {}; using {}".format(inferred, requested_mode, requested_mode)
        )
        return requested_mode

    return inferred


def default_experiment_name(args):
    """
    Name a run after its method and ratio.

    Non-default allocation policies get a suffix so that, for example, a
    turn-allocation run and a sensitivity run at the same ratio do not write
    to the same score/prediction files.
    """
    name = f"{args.method}_ratio{args.ratio}"
    if args.allocation_signal == "sensitivity":
        name += f"_sens-{args.sensitivity_mode}_sa{args.sens_alpha}"
    elif args.allocation_signal != "turn":
        name += f"_{args.allocation_signal}"
    return name


def select_next_token(logits, temperature, top_p):
    """Greedy decoding at temperature 0; optional nucleus sampling otherwise."""
    if temperature <= 0:
        return torch.argmax(logits, dim=-1, keepdim=True)

    probabilities = torch.softmax(logits.float() / temperature, dim=-1)
    if top_p is not None and top_p < 1.0:
        sorted_probs, sorted_indices = torch.sort(probabilities, descending=True)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        remove = cumulative > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        sorted_probs = sorted_probs.masked_fill(remove, 0.0)
        sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
        sampled_sorted = torch.multinomial(sorted_probs, num_samples=1)
        return sorted_indices.gather(-1, sampled_sorted)
    return torch.multinomial(probabilities, num_samples=1)


def stop_token_sequences(conv, tokenizer):
    """Tokenize the conversation separator and EOS stop markers."""
    sequences = []
    stop_string = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    if stop_string:
        ids = tokenizer.encode(stop_string, add_special_tokens=False)
        if ids:
            sequences.append(ids)
    eos_id = tokenizer.eos_token_id
    eos_ids = [eos_id] if isinstance(eos_id, int) else (eos_id or [])
    for token_id in eos_ids:
        if [int(token_id)] not in sequences:
            sequences.append([int(token_id)])
    return sequences


def generate_one_turn(
    model,
    tokenizer,
    conv,
    next_token_logits,
    past_key_values,
    kv_cache,
    num_of_token,
    requested_limit,
    context_length,
    temperature,
    top_p,
):
    """
    Generate one assistant answer into the continuing KV cache.

    Every sampled token is fed exactly once, so it becomes part of the KV cache
    and produces the logits for the next step.  If generation stops on a
    separator/EOS sequence, those stop tokens are already committed and are
    returned separately so the caller does not feed them a second time.
    """
    separator_reserve = 8
    context_budget = None
    if context_length is not None:
        context_budget = max(0, context_length - num_of_token - separator_reserve)

    if requested_limit is None:
        if context_budget is None:
            raise RuntimeError(
                "Free generation needs a known context length. Pass "
                "--max-token-generated if the model does not expose one."
            )
        effective_limit = context_budget
    elif context_budget is None:
        effective_limit = requested_limit
    else:
        effective_limit = min(requested_limit, context_budget)

    raw_generated = []
    committed_stop = []
    stop_reason = (
        "context_limit"
        if context_budget is not None and effective_limit == context_budget
        else "max_token_generated"
    )
    stop_sequences = stop_token_sequences(conv, tokenizer)

    for _ in range(effective_limit):
        next_id = select_next_token(next_token_logits, temperature, top_p)
        raw_generated.append(int(next_id.item()))
        with torch.no_grad():
            next_token_logits, past_key_values, num_of_token = forward_one_text_token(
                model,
                next_id,
                past_key_values,
                kv_cache,
                num_of_token,
            )

        matched = next(
            (
                sequence
                for sequence in stop_sequences
                if len(raw_generated) >= len(sequence)
                and raw_generated[-len(sequence) :] == sequence
            ),
            None,
        )
        if matched is not None:
            committed_stop = matched
            raw_generated = raw_generated[: -len(matched)]
            stop_reason = "eos_or_separator"
            break

    answer_ids = torch.tensor(
        raw_generated,
        dtype=torch.long,
        device=next_token_logits.device,
    ).unsqueeze(0)
    answer = tokenizer.decode(
        raw_generated,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return (
        answer,
        answer_ids,
        committed_stop,
        stop_reason,
        next_token_logits,
        past_key_values,
        num_of_token,
        len(raw_generated) + len(committed_stop),
    )


def main(args):
    turn_conditioned_methods = {
        "turn_conditioned_layerwise_merge",
        "turn_conditioned_layerwise_eviction",
        "turn_conditioned_coverage",
    }
    if args.num_beams != 1:
        raise ValueError(
            "This continuous-cache evaluator currently supports --num-beams 1 only."
        )
    # Every cache method may have a physical KV length shorter than its logical
    # RoPE position after compression.
    enable_legacy_tgv_attention()

    # Nova/GPU path: keep device handling simple and explicit.
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"--device {args.device} was requested, but torch.cuda.is_available() is False."
        )

    device = torch.device(args.device)
    print(f"Using device: {device}")

    if args.sens_alpha <= 0:
        raise ValueError("--sens-alpha must be positive")
    if args.method not in turn_conditioned_methods and args.allocation_signal != "turn":
        raise ValueError(
            "--allocation-signal applies only to the turn-conditioned methods"
        )
    with open(args.data_path, "r") as f:
        data = json.load(f)

    model_name = get_model_name_from_path(args.model_path)
    expected_layers = 32 if "7b" in model_name.lower() else 40
    args.sensitivity_joint = None
    args.sensitivity_image = None
    args.sensitivity_text = None
    args.sensitivity_metadata = None
    if args.allocation_signal == "sensitivity":
        profiles, args.sensitivity_metadata = load_sensitivity_profile(
            args.sensitivity_profile,
            args.sensitivity_mode,
            expected_layers,
            args.ratio,
        )
        args.sensitivity_joint = profiles["joint"]
        args.sensitivity_image = profiles["image"]
        args.sensitivity_text = profiles["text"]
        print(
            "Loaded frozen sensitivity allocation profile: "
            f"{args.sensitivity_profile} (mode={args.sensitivity_mode}, "
            f"power={args.sens_alpha}). "
            "--cross-turn-weight is not used by this allocation policy."
        )

    tokenizer, model, image_processor, loader_context_length = load_pretrained_model(
        args.model_path,
        args.model_base,
        model_name,
        args.load_8bit,
        args.load_4bit,
        device=args.device,
    )

    # Install the per-layer hooks. They remain inactive unless online_attention()
    # supplies a cache observe_layer function.
    enable_online_attention(model)

    # LLaVA-v1.5 normally produces 576 visual tokens for one image.
    vision_tower = model.get_vision_tower()
    image_token_len_from_tower = int(vision_tower.num_patches)

    args.conv_mode = choose_conversation_mode(model_name, args.conv_mode)
    context_length = effective_context_length(model, loader_context_length)
    rouge_scorer = Rouge()

    # ROUGE-L F1 per generated turn, plus the same scores grouped by 1-based
    # conversation turn.
    all_turn_scores = []
    per_turn_scores = {}
    outputs_json = []
    # Turns whose generated tokens differ from re-tokenizing the decoded text.
    retokenization_mismatches = 0

    os.makedirs(args.output_dir, exist_ok=True)
    exp_name = args.exp_name or default_experiment_name(args)
    score_path = os.path.join(args.output_dir, f"{exp_name}.txt")
    pred_path = os.path.join(args.output_dir, f"{exp_name}_predictions.json")
    if args.eval_samples is not None and args.eval_samples < len(data):
        data = data[: args.eval_samples]

    # ================================================================
    # Each item is ONE continuous conversation associated with ONE image.
    # Cache state is reset only when we move to the next dataset item.
    # ================================================================
    for item_index, item in enumerate(tqdm(data)):
        if "turns" not in item or len(item["turns"]) == 0:
            continue

        # ----------------
        # Prepare the image
        # ----------------
        image_path = os.path.join(args.image_path, item["image"])
        image = load_image(image_path)
        image_tensor = process_images([image], image_processor, args)
        image_tensor = image_tensor.to(model.device, dtype=getattr(model, "dtype", next(model.parameters()).dtype))

        # ---------------------------------------------------------
        # Conversation and cache state belong to this ONE data item.
        # ---------------------------------------------------------
        conv = conv_templates[args.conv_mode].copy()
        past_key_values = None
        kv_cache = None

        # num_of_token tracks the number of tokens processed in the uncompressed
        # model stream.  The physical KV cache may be shorter after pruning.
        num_of_token = 0

        # `committed_full_ids` is the logical tokenization of everything that we
        # have ALREADY fed into the model, including previous answer separators.
        # It is used only to compute the exact suffix added by the next turn.
        committed_full_ids = None

        # `next_token_logits` are the logits after the most recently committed
        # token.  At the beginning of an assistant turn they predict the first
        # generated answer token.
        next_token_logits = None
        item_output = {"image": item["image"], "turns": []}

        for turn_id, turn in enumerate(item["turns"]):
            original_question = turn.get("question", "")
            question = original_question
            reference = turn.get("answer", "")

            # ============================================================
            # 1. Add the current USER turn to the logical conversation.
            #    The image placeholder is present ONLY in the first turn.
            # ============================================================
            if turn_id == 0 and DEFAULT_IMAGE_TOKEN not in question:
                question = question + "\n" + DEFAULT_IMAGE_TOKEN

            conv.append_message(conv.roles[0], question)
            conv.append_message(conv.roles[1], None)
            base_prompt = conv.get_prompt()
            base_ids = tokenize_conversation_prompt(
                base_prompt, tokenizer, model.device
            )

            if turn_id == 0:
                image_positions = (
                    base_ids[0] == IMAGE_TOKEN_INDEX
                ).nonzero(as_tuple=True)[0]

                if image_positions.numel() != 1:
                    raise ValueError(
                        f"Expected exactly one image placeholder, "
                        f"found {image_positions.numel()}"
                    )

                image_start = int(image_positions[0].item())
                image_token_len = image_token_len_from_tower

                # The cache must exist before the model runs so each layer can immediately
                # deliver its attention statistics.
                kv_cache = build_cache(
                    args,
                    model_name,
                    image_start=image_start,
                    image_token_len=image_token_len,
                )
                # Mass-preserving caches add log(mass) to the key logits of each layer.
                if hasattr(kv_cache, "attach_model"):
                    kv_cache.attach_model(model)

                sink = (
                    kv_cache.observe_layer
                    if prefill_reads_attention(kv_cache)
                    else None
                )

                # --------------------------------------------------------
                # FIRST AND ONLY multimodal prefill for this conversation.
                # --------------------------------------------------------
                # During this forward pass:
                #   layer 0 attention -> sink -> reduced -> discarded
                #   layer 1 attention -> sink -> reduced -> discarded
                #   ...
                # Therefore the model does not retain all layer attention matrices.
                with torch.no_grad(), online_attention(model, sink):
                    outputs = model(
                        input_ids=base_ids,
                        images=image_tensor,
                        past_key_values=None,
                        use_cache=True,
                        output_attentions=sink is not None,
                    )

                # Save the two outputs that are still needed.
                past_key_values = outputs.past_key_values
                next_token_logits = outputs.logits[:, -1, :].clone()

                # We no longer need outputs.attentions:
                # observe_layer() already received every layer's attention through the hook.
                # Deleting outputs allows the full-prompt logits and returned container to be released.
                del outputs

                # This is the expanded sequence length:
                # ordinary text tokens + all 576 visual tokens.
                effective_prefill_len = past_key_values[0][0].size(2)
                num_of_token = effective_prefill_len

                # Confirm that the image position and visual-token count supplied before
                # the forward agree with the actual expanded KV cache.
                actual_image_start, actual_image_token_len = infer_actual_image_span(
                    base_ids,
                    past_key_values,
                    k_seq_dim=2,
                )

                if (
                    actual_image_start != image_start
                    or actual_image_token_len != image_token_len
                ):
                    raise RuntimeError(
                        "The image span inferred from the expanded KV cache differs "
                        "from the vision tower's patch count: "
                        f"expected ({image_start}, {image_token_len}), "
                        f"got ({actual_image_start}, {actual_image_token_len})."
                    )

                if args.debug:
                    print(
                        f"\n[item {item_index} turn 0] "
                        f"text/input ids={base_ids.shape[1]}, "
                        f"expanded KV={effective_prefill_len}, "
                        f"image_start={image_start}, "
                        f"image_token_len={image_token_len}"
                    )

                # Apply the cache policy.
                # attentions=None is intentional. The cache already collected the required
                # statistics one layer at a time through observe_layer().
                past_key_values = apply_cache(
                    kv_cache,
                    past_key_values,
                    num_of_token,
                    None,
                )

            else:
                # ---------------------------------------------------------
                # Previous answer + separator are already in past_key_values.
                # Append ONLY the new user turn and assistant prefix.
                # ---------------------------------------------------------
                new_turn_ids = require_prefix(
                    base_ids,
                    committed_full_ids,
                    description=f"item {item_index}, turn {turn_id}",
                )

                if new_turn_ids.shape[1] == 0:
                    raise RuntimeError(
                        f"item {item_index}, turn {turn_id}: no new user tokens "
                        "were added to the conversation."
                    )

                # Send ONLY Q2/Q3/... plus template formatting.  No image and no
                # previous question/answer are resent.
                with torch.no_grad():
                    next_token_logits, past_key_values, num_of_token = feed_text_tokens(
                        model,
                        new_turn_ids,
                        past_key_values,
                        kv_cache,
                        num_of_token,
                    )

                if args.debug:
                    print(
                        f"[item {item_index} turn {turn_id}] "
                        f"new user/template tokens={new_turn_ids.shape[1]}"
                    )

            # ============================================================
            # 2. Generate the assistant answer into the continuing cache.
            # ============================================================
            (
                generated,
                generated_ids,
                committed_stop,
                stop_reason,
                next_token_logits,
                past_key_values,
                num_of_token,
                generation_steps,
            ) = generate_one_turn(
                model,
                tokenizer,
                conv,
                next_token_logits,
                past_key_values,
                kv_cache,
                num_of_token,
                args.max_token_generated,
                context_length,
                args.temperature,
                args.top_p,
            )

            # ============================================================
            # 3. Derive the RENDERED answer and separator tokens.
            #
            # The KV cache always keeps the tokens the model actually
            # generated.  The rendered text is re-tokenized only to find the
            # template separator and, next turn, the newly appended user
            # tokens.  Both diffs are taken between rendered transcripts, so
            # they stay prefix-stable even when the generated split differs.
            # ============================================================
            conv_with_answer = conv.copy()
            conv_with_answer.messages[-1][-1] = generated
            prompt_with_answer = conv_with_answer.get_prompt()

            # Prompt ending exactly after answer text, before separator/EOS.
            prompt_through_answer = answer_only_prompt(
                base_prompt,
                prompt_with_answer,
                generated,
            )

            ids_through_answer = tokenize_conversation_prompt(
                prompt_through_answer, tokenizer, model.device
            )
            ids_with_answer_and_separator = tokenize_conversation_prompt(
                prompt_with_answer, tokenizer, model.device
            )

            rendered_answer_ids = require_prefix(
                ids_through_answer,
                base_ids,
                description=f"item {item_index}, turn {turn_id}, generated answer",
            )
            # A heavily compressed model can emit a valid but non-canonical
            # token split (for example "▁" + "The" instead of "▁The").
            # Decoding then re-encoding yields the canonical split, so the two
            # differ even though the text is identical.  This is not a cache
            # error: num_of_token counts the tokens really fed, so RoPE
            # positions stay correct.  Record it instead of aborting.
            answer_retokenization_stable = torch.equal(
                rendered_answer_ids, generated_ids
            )
            if not answer_retokenization_stable:
                retokenization_mismatches += 1
                if args.debug:
                    print(
                        f"[item {item_index} turn {turn_id}] generated answer "
                        "re-tokenizes differently: "
                        f"generated={generated_ids.shape[1]} tokens, "
                        f"rendered={rendered_answer_ids.shape[1]} tokens"
                    )

            # The remaining tokens are conversation-template separators/EOS.
            separator_ids = require_prefix(
                ids_with_answer_and_separator,
                ids_through_answer,
                description=f"item {item_index}, turn {turn_id}, separator",
            )

            # ============================================================
            # 4. Commit the rest of the assistant-turn separator/EOS.
            #
            # Stop tokens that ended generation are already in the KV cache, so
            # only the remaining separator tokens are fed.  Every logical
            # conversation token is committed exactly once.
            # ============================================================
            already_committed = len(committed_stop)
            remaining_separator = separator_ids
            if already_committed:
                stop_tensor = torch.tensor(
                    committed_stop,
                    dtype=torch.long,
                    device=separator_ids.device,
                ).unsqueeze(0)
                if (
                    separator_ids.shape[1] >= already_committed
                    and torch.equal(
                        separator_ids[:, :already_committed], stop_tensor
                    )
                ):
                    remaining_separator = separator_ids[:, already_committed:]
                else:
                    # The template omits the separator after an EMPTY answer
                    # (model stopped on its first token).  The stop token is
                    # already in the KV cache, which is the separator's role,
                    # so nothing more is fed.
                    remaining_separator = separator_ids[:, :0]
                    if args.debug:
                        print(
                            f"[item {item_index} turn {turn_id}] stop tokens "
                            "already committed; template separator "
                            f"has {separator_ids.shape[1]} tokens, feeding none."
                        )
            if remaining_separator.shape[1] > 0:
                with torch.no_grad():
                    next_token_logits, past_key_values, num_of_token = feed_text_tokens(
                        model,
                        remaining_separator,
                        past_key_values,
                        kv_cache,
                        num_of_token,
                    )

            past_key_values = finalize_turn_cache(kv_cache, past_key_values)
            if args.debug and hasattr(kv_cache, "last_finalize_stats"):
                print(f"[item {item_index} turn {turn_id}] finalize: "
                      f"{kv_cache.last_finalize_stats}")

            # The model KV now represents the complete conversation through the
            # current generated answer and its separator.  Advance the logical state.
            conv = conv_with_answer
            committed_full_ids = ids_with_answer_and_separator

            # ============================================================
            # 5. Score the generated answer against the reference.
            # ============================================================
            generated_norm = normalize_text(generated, args.score_max_words)
            reference_norm = normalize_text(reference, args.score_max_words)
            try:
                score = rouge_scorer.get_scores(
                    generated_norm, reference_norm
                )[0]["rouge-l"]["f"] if generated_norm and reference_norm else 0.0
            except ValueError:
                # rouge raises "Hypothesis is empty." for answers made only of
                # punctuation (e.g. "."); such an answer scores 0.
                score = 0.0
            all_turn_scores.append(score)
            per_turn_scores.setdefault(turn_id + 1, []).append(score)
            item_output["turns"].append(
                {
                    "turn": turn_id + 1,
                    "question": original_question,
                    "reference": reference,
                    "prediction": generated,
                    "rouge_l_f": score,
                    "generated_answer_tokens": int(generated_ids.shape[1]),
                    "answer_retokenization_stable": bool(
                        answer_retokenization_stable
                    ),
                    "generation_steps_including_stop": generation_steps,
                    "stop_reason": stop_reason,
                }
            )

            if args.debug:
                physical_kv_len = past_key_values[0][0].size(2)
                print(
                    f"[item {item_index} turn {turn_id}] "
                    f"answer tokens={generated_ids.shape[1]}, "
                    f"separator tokens={separator_ids.shape[1]}, "
                    f"stop={stop_reason}, "
                    f"logical/effective processed={num_of_token}, "
                    f"physical KV={physical_kv_len}, "
                    f"position gap={num_of_token - physical_kv_len}"
                )

        if getattr(kv_cache, "alpha_auto", False):
            # alpha used at each compression and the novelty measured for each turn.
            item_output["coverage_alpha_history"] = kv_cache.alpha_history
        outputs_json.append(item_output)
        del past_key_values, kv_cache, image_tensor
        if device.type == "mps":
            torch.mps.empty_cache()

    average = sum(all_turn_scores) / len(all_turn_scores) if all_turn_scores else 0.0
    per_turn_average = {
        str(turn): sum(scores) / len(scores)
        for turn, scores in sorted(per_turn_scores.items())
    }

    # Keep the old one-number text output for compatibility.
    with open(score_path, "a") as f:
        f.write(f"{average}\n")

    # Also write a structured JSON result with every prediction.
    result = {
        "experiment": exp_name,
        "method": args.method,
        "ratio": float(args.ratio),
        "requested_eval_samples": int(args.eval_samples) if args.eval_samples is not None else None,
        "evaluated_items": int(len(data)),
        "max_token_generated_per_turn": args.max_token_generated,
        "free_generation": args.max_token_generated is None,
        "temperature": args.temperature,
        "model_context_length": context_length,
        "avg_rouge_l_f": average,
        "per_turn_rouge_l_f": per_turn_average,
        "num_items": len(outputs_json),
        "num_turns_scored": len(all_turn_scores),
        "retokenization_mismatch_turns": int(retokenization_mismatches),
        "alpha": float(args.alpha),
        "beta": float(args.beta),
        "cross_turn_weight": (
            float(args.cross_turn_weight)
            if args.method in turn_conditioned_methods
            and args.allocation_signal == "turn"
            else None
        ),
        "allocation_signal": (
            args.allocation_signal
            if args.method in turn_conditioned_methods
            else None
        ),
        "sensitivity_mode": (
            args.sensitivity_mode
            if args.allocation_signal == "sensitivity"
            else None
        ),
        "sensitivity_profile": (
            args.sensitivity_profile
            if args.allocation_signal == "sensitivity"
            else None
        ),
        "sens_alpha": (
            float(args.sens_alpha)
            if args.allocation_signal == "sensitivity"
            else None
        ),
        "results": outputs_json,
    }
    with open(pred_path, "w") as f:
        json.dump(result, f, indent=2)

    if retokenization_mismatches:
        print(
            f"[NOTE] {retokenization_mismatches} turn(s) generated a non-canonical "
            "token split; the KV cache kept the generated tokens."
        )
    print(f"Average ROUGE-L F1 over {len(all_turn_scores)} turns: {average:.6f}")
    for turn_number, turn_score in per_turn_average.items():
        print(f"Turn {turn_number} ROUGE-L F1: {turn_score:.6f}")
    print(f"Saved score to: {score_path}")
    print(f"Saved predictions to: {pred_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # LLaVA/model arguments retained from the original evaluator.
    parser.add_argument("--model-path", type=str, default="/lustre/hdd/LAS/jannesar-lab/tmahmud/models/llava-v1.5-7b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--data-path", type=str, default="/lustre/hdd/LAS/jannesar-lab/msamani/modality_aware_KVCache/MultiVerse_for_ppl/multiverse_ppl.json")
    parser.add_argument("--image-path", type=str, default="/lustre/hdd/LAS/jannesar-lab/msamani/modality_aware_KVCache/MultiVerse_raw/images")
    parser.add_argument("--device", type=str, default="cuda", help="Nova: cuda (default) or cuda:N")
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--image-aspect-ratio", type=str, default="pad")

    # Generation arguments (ROUGE only).
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="0 gives deterministic greedy decoding; values above 0 sample.",
    )
    parser.add_argument("--top_p", "--top-p", type=float, default=None)
    parser.add_argument("--num_beams", "--num-beams", type=int, default=1)
    parser.add_argument(
        "--max-token-generated",
        "--max_token_generated",
        "--max-new-tokens",
        type=parse_generated_token_limit,
        default=512,
        metavar="FREE|N|Nk",
        help=(
            "Maximum generated tokens for each turn (default 512). Pass 'free' "
            "to generate until EOS/separator or remaining model-context capacity."
        ),
    )

    # Cache/evaluation arguments retained from the original evaluator.
    parser.add_argument("--start-size", type=int, default=1)
    parser.add_argument("--recent-size", type=int, default=2047)
    parser.add_argument("--eval-samples", type=int, default=None)
    parser.add_argument(
        "--exp-name",
        type=str,
        default="",
        help=(
            "Output file stem. Default: <method>_ratio<ratio>, plus an "
            "allocation suffix when --allocation-signal is not 'turn'."
        ),
    )
    parser.add_argument(
        "--method",
        type=str,
        default="pyramidkv",
        choices=[
            "turn_conditioned_layerwise_merge",
            "turn_conditioned_layerwise_eviction",
            "turn_conditioned_coverage",
            "tgv_kv",
            "local_cache",
            "h2o",
            "elastic_cache",
            "meda",
            "pyramidkv",
        ],
    )
    parser.add_argument(
        "--ratio",
        type=float,
        default=0.9,
        help="Pruned fraction: 0.9 means prune 90%% and retain about 10%%.",
    )
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="Ordinary-attention weight for multimodal caches.")
    parser.add_argument("--beta", type=float, default=0.2,
                        help="Text-image connection weight for multimodal caches.")
    parser.add_argument(
        "--cross-turn-weight",
        type=float,
        default=1.0,
        help=(
            "Weight on current-turn attention to prior-turn KV entries for "
            "--allocation-signal turn; ignored for sensitivity allocation."
        ),
    )
    parser.add_argument(
        "--allocation-signal",
        choices=["uniform", "turn", "sensitivity"],
        default="turn",
        help=(
            "Layer-budget policy: equal budgets, dynamic grounding/history, "
            "or a frozen sensitivity profile."
        ),
    )
    parser.add_argument(
        "--sensitivity-mode",
        choices=["joint", "image", "text", "split"],
        default="text",
        help=(
            "Profile field used when --allocation-signal sensitivity. Split "
            "uses image at the first boundary and image+text afterward."
        ),
    )
    parser.add_argument(
        "--sensitivity-profile",
        type=str,
        default=None,
        help=(
            "JSON written by calibrate_sensitivity.py or "
            "calibrate_sensitivity_joint.py."
        ),
    )
    parser.add_argument(
        "--sens-alpha",
        type=float,
        default=1.0,
        help=(
            "Exponent applied to calibrated layer sensitivity. This is "
            "separate from --alpha, the token-attention score weight."
        ),
    )

    # Turn-conditioned coverage (kv_cache_coverage.py) arguments.
    parser.add_argument("--coverage-alpha", type=float, default=0.2,
                        help="Coverage: share of expected attention reserved for unseen questions (0 = relevance only).")
    parser.add_argument("--coverage-kappa", type=float, default=1.0,
                        help="Coverage: similarity bandwidth in units of the median nearest-neighbour key distance (0 = no coverage).")
    parser.add_argument("--coverage-similarity", choices=["key", "value", "key_value"], default="key",
                        help="Coverage: features used for visual-token similarity.")
    parser.add_argument("--coverage-recent-window", type=int, default=8,
                        help="Coverage: the most recent text entries always kept at every compression (shrinks to budget//4 at small budgets).")
    parser.add_argument("--coverage-text-prior", type=float, default=0.5,
                        help="Coverage: at the Turn-1 prefill, keep current-turn text up to this share of the layer budget.")
    parser.add_argument("--coverage-no-prefill", action="store_true",
                        help="Coverage ablation: do not compress after the Turn-1 prefill (boundary-only, like the base method).")
    parser.add_argument("--coverage-relevance", choices=["text_mean", "cumulative"], default="text_mean",
                        help="Coverage: debiased text-query relevance (default) or the base method's cumulative attention.")
    parser.add_argument("--coverage-merge", action="store_true", default=False,
                        help="Coverage: average each evicted visual entry into the anchor that covers it.")
    parser.add_argument("--coverage-layer-floor", type=float, default=0.5,
                        help="Coverage: every layer gets at least this share of the uniform per-layer budget (0 = allocation signal only).")
    parser.add_argument("--coverage-floor-stage", choices=["all", "prefill"], default="all",
                        help="Coverage: apply the layer floor at every compression or only at the Turn-1 prefill.")
    parser.add_argument("--coverage-no-mass", action="store_true",
                        help="Coverage ablation: plain eviction instead of moving evicted visual mass to the covering anchor (proportional attention).")
    parser.add_argument("--coverage-mass-min-sim", type=float, default=0.25,
                        help="Coverage: an evicted visual entry's mass moves to its most similar kept anchor only if their kernel similarity is at least this (0 = conserve all image mass).")
    parser.add_argument("--coverage-alpha-auto", action="store_true", default=False,
                        help="Coverage: choose coverage_alpha automatically for every conversation (ignores --coverage-alpha).")

    # ROUGE output arguments.
    parser.add_argument("--output-dir", type=str, default="logs_rouge_llava")
    parser.add_argument("--score-max-words", type=int, default=256)


    # MEDA-specific arguments.  The original paper used a 3:1 recent-to-history ratio.
    parser.add_argument(
        "--meda-recent-fraction",
        type=float,
        default=0.75,
        help="MEDA: share of each layer budget kept as the recent window (paper 3:1).",
    )
    parser.add_argument(
        "--meda-entropy",
        choices=["crossmodal", "full"],
        default="crossmodal",
        help="MEDA layer entropy: paper cross-modal (default) or released-code full-attention.",
    )


    # PyramidKV-specific arguments.  The original PyramidKV paper used a window size of 8 and a kernel size of 7.
    parser.add_argument("--pyramid-window-size", type=int, default=8,
                        help="PyramidKV: observation window (KVCache-Factory uses 8).")
    parser.add_argument("--pyramid-kernel-size", type=int, default=7,
                        help="PyramidKV: pooling kernel over keys (odd).")
    parser.add_argument("--pyramid-pooling", choices=["maxpool", "avgpool"], default="maxpool",
                        help="PyramidKV: score pooling.")
    parser.add_argument("--pyramid-beta", type=float, default=20,
                        help="PyramidKV: top/bottom budget spread; last layer gets M // beta.")

    # Debug mode prints per-turn token counts and physical KV lengths.  This is
    # particularly useful while validating a new multi-turn dataset/cache.
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()
    main(args)