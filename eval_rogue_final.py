"""Continuous multi-turn ROUGE evaluation for LLaVA KV-cache methods.

The model freely generates each assistant answer. Its generated answer is then
kept in the continuing conversation and KV cache for the following turn. The
image is processed only once, during Turn 1 prefill.

``--max-token-generated`` is a per-turn generation ceiling. If it is omitted,
generation is free-running until the conversation stop token or the model's
remaining context capacity is reached. Values such as 1k, 2k, and 8k are
accepted and mean 1000, 2000, and 8000 tokens respectively.
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

from kv_cache_final import (
    ElasticCache,
    H2OCache,
    LocalCache,
    TGVKVCache,
    TurnConditionedLayerwiseEvictionKVCache,
    TurnConditionedLayerwiseKVCache,
    enable_legacy_tgv_attention,
    set_legacy_logical_position_limit,
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
    """Load a local image or URL as RGB."""
    if image_file.startswith("http://") or image_file.startswith("https://"):
        response = requests.get(image_file, timeout=30)
        response.raise_for_status()
        return Image.open(BytesIO(response.content)).convert("RGB")
    return Image.open(image_file).convert("RGB")


def normalize_text(text, max_words=256):
    """Normalize generated/reference text only for ROUGE scoring."""
    text = (text or "").replace("</s>", "").replace("<AND>", " ").strip()
    return " ".join(text.split()[:max_words])


def infer_conv_mode(model_name, requested_conv_mode):
    if "llama-2" in model_name.lower():
        inferred = "llava_llama_2"
    elif "v1" in model_name.lower():
        inferred = "llava_v1"
    elif "mpt" in model_name.lower():
        inferred = "mpt"
    else:
        inferred = "llava_v0"

    if requested_conv_mode is not None and requested_conv_mode != inferred:
        print(
            "[WARNING] auto-inferred conversation mode is {}, while "
            "--conv-mode is {}; using {}".format(
                inferred, requested_conv_mode, requested_conv_mode
            )
        )
        return requested_conv_mode
    return inferred


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


def tokenize_prompt(prompt, tokenizer, device):
    ids = tokenizer_image_token(
        prompt,
        tokenizer,
        IMAGE_TOKEN_INDEX,
        return_tensors="pt",
    )
    return ids.unsqueeze(0).to(device)


def require_prefix(full_ids, prefix_ids, description):
    """Return the newly appended suffix, failing on transcript divergence."""
    prefix_len = prefix_ids.shape[1]
    if full_ids.shape[1] < prefix_len:
        raise RuntimeError(f"{description}: new transcript is shorter than its prefix.")
    if not torch.equal(full_ids[:, :prefix_len], prefix_ids):
        mismatch = (full_ids[:, :prefix_len] != prefix_ids).nonzero(as_tuple=False)
        first_bad = int(mismatch[0, 1].item()) if mismatch.numel() else -1
        raise RuntimeError(
            f"{description}: conversation tokenization is not prefix-stable; "
            f"first mismatch at token {first_bad}."
        )
    return full_ids[:, prefix_len:]


def infer_actual_image_span(input_ids, past_key_values, k_seq_dim=2):
    """Infer the expanded visual KV span rather than assuming 576 tokens."""
    image_positions = (input_ids[0] == IMAGE_TOKEN_INDEX).nonzero(as_tuple=True)[0]
    if image_positions.numel() == 0:
        return None, 0
    if image_positions.numel() != 1:
        raise ValueError(
            f"Expected one image placeholder, found {image_positions.numel()}."
        )

    expanded_len = past_key_values[0][0].size(k_seq_dim)
    text_len = input_ids.shape[1] - image_positions.numel()
    image_token_len = int(expanded_len - text_len)
    if image_token_len <= 0:
        raise RuntimeError("Could not infer a positive expanded image-token length.")
    return int(image_positions[0].item()), image_token_len


def effective_context_length(model, loader_context_length):
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


def build_kv_cache(args, model_name, image_start=None, image_token_len=0):
    """Build one cache policy per image/dialogue item."""
    layer_num = 32 if "7b" in model_name.lower() else 40
    common = dict(
        start_size=args.start_size,
        recent_size=args.recent_size,
        k_seq_dim=2,
        v_seq_dim=2,
        ratio=args.ratio,
    )

    if args.method == "turn_conditioned_layerwise_merge":
        return TurnConditionedLayerwiseKVCache(
            layer_num=layer_num,
            image_start=image_start,
            image_token_len=image_token_len,
            attn_weight=args.alpha,
            conn_weight=args.beta,
            cross_turn_weight=args.cross_turn_weight,
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
    if args.method == "local_cache":
        return LocalCache(**common)
    if args.method == "h2o":
        return H2OCache(**common)
    if args.method == "elastic_cache":
        return ElasticCache(layer_num=layer_num, **common)
    raise ValueError(f"Invalid method: {args.method}")


def apply_cache(kv_cache, past_key_values, num_of_token, attentions):
    return kv_cache(
        past_key_values,
        num_of_token=num_of_token,
        attentions=attentions,
    )


def finalize_turn_cache(kv_cache, past_key_values):
    """Run delayed boundary compression, then advance text provenance."""
    if not hasattr(kv_cache, "finalize_turn"):
        return past_key_values
    past_key_values = kv_cache.finalize_turn(past_key_values)
    if isinstance(kv_cache, TurnConditionedLayerwiseKVCache):
        kv_cache.begin_turn(kv_cache.current_turn + 1)
    return past_key_values


def forward_one_token(model, token_ids, past_key_values, kv_cache, logical_len):
    if token_ids.shape != (1, 1):
        raise ValueError(f"Expected one token [1, 1], got {tuple(token_ids.shape)}")

    position_ids = torch.tensor(
        [[logical_len]], dtype=torch.long, device=token_ids.device
    )
    if isinstance(kv_cache, (TGVKVCache, TurnConditionedLayerwiseKVCache)):
        set_legacy_logical_position_limit(model, logical_len + 1)

    outputs = model(
        input_ids=token_ids,
        position_ids=position_ids,
        past_key_values=past_key_values,
        use_cache=True,
        output_attentions=True,
    )
    logical_len += outputs.logits.shape[1]
    past_key_values = apply_cache(
        kv_cache,
        outputs.past_key_values,
        logical_len,
        outputs.attentions,
    )
    return outputs.logits[:, -1, :], past_key_values, logical_len


def feed_tokens(model, token_ids, past_key_values, kv_cache, logical_len):
    next_logits = None
    for pos in range(token_ids.shape[1]):
        next_logits, past_key_values, logical_len = forward_one_token(
            model,
            token_ids[:, pos : pos + 1],
            past_key_values,
            kv_cache,
            logical_len,
        )
    return next_logits, past_key_values, logical_len


def answer_only_prompt(base_prompt, prompt_with_answer, answer):
    """Return the rendered prompt ending before the assistant separator."""
    if not prompt_with_answer.startswith(base_prompt):
        raise RuntimeError("Conversation template did not extend the base prompt.")
    continuation = prompt_with_answer[len(base_prompt) :]
    answer_pos = continuation.find(answer)
    if answer_pos < 0:
        raise RuntimeError("Could not find the generated answer in the prompt.")
    return base_prompt + continuation[: answer_pos + len(answer)]


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
    next_logits,
    past_key_values,
    kv_cache,
    logical_len,
    requested_limit,
    context_length,
    temperature,
    top_p,
):
    """Generate one answer and return any separator already committed."""
    separator_reserve = 8
    context_budget = None
    if context_length is not None:
        context_budget = max(0, context_length - logical_len - separator_reserve)

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
        next_id = select_next_token(next_logits, temperature, top_p)
        raw_generated.append(int(next_id.item()))
        with torch.no_grad():
            next_logits, past_key_values, logical_len = forward_one_token(
                model,
                next_id,
                past_key_values,
                kv_cache,
                logical_len,
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
        device=next_logits.device,
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
        next_logits,
        past_key_values,
        logical_len,
        len(raw_generated) + len(committed_stop),
    )


def main(args):
    if args.num_beams != 1:
        raise ValueError("This continuous-cache evaluator currently supports --num-beams 1 only.")
    if args.method in {
        "tgv_kv",
        "turn_conditioned_layerwise_merge",
        "turn_conditioned_layerwise_eviction",
    }:
        enable_legacy_tgv_attention()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device {args.device} was requested but is unavailable.")

    with open(args.data_path, "r") as f:
        data = json.load(f)[: args.eval_samples]

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, loader_context_length = load_pretrained_model(
        args.model_path,
        args.model_base,
        model_name,
        args.load_8bit,
        args.load_4bit,
        device=args.device,
    )
    args.conv_mode = infer_conv_mode(model_name, args.conv_mode)
    context_length = effective_context_length(model, loader_context_length)
    rouge_scorer = Rouge()

    os.makedirs(args.output_dir, exist_ok=True)
    exp_name = args.exp_name or f"{args.method}_ratio{args.ratio}"
    score_path = os.path.join(args.output_dir, f"{exp_name}.txt")
    pred_path = os.path.join(args.output_dir, f"{exp_name}_predictions.json")
    all_turn_scores = []
    per_turn_scores = {}
    outputs_json = []

    for item_idx, item in enumerate(tqdm(data)):
        turns = item.get("turns", [])
        if not turns:
            continue

        image_path = os.path.join(args.image_path, item["image"])
        image = load_image(image_path)
        image_tensor = process_images([image], image_processor, args)
        image_tensor = image_tensor.to(
            model.device,
            dtype=getattr(model, "dtype", next(model.parameters()).dtype),
        )

        conv = conv_templates[args.conv_mode].copy()
        past_key_values = None
        kv_cache = None
        logical_len = 0
        committed_full_ids = None
        next_logits = None
        item_output = {"image": item["image"], "turns": []}

        for turn_idx, turn in enumerate(turns):
            original_question = turn.get("question", "")
            question = original_question
            reference = turn.get("answer", "")
            if turn_idx == 0 and DEFAULT_IMAGE_TOKEN not in question:
                question += "\n" + DEFAULT_IMAGE_TOKEN

            conv.append_message(conv.roles[0], question)
            conv.append_message(conv.roles[1], None)
            base_prompt = conv.get_prompt()
            base_ids = tokenize_prompt(base_prompt, tokenizer, model.device)

            if turn_idx == 0:
                with torch.no_grad():
                    model_outputs = model(
                        input_ids=base_ids,
                        images=image_tensor,
                        past_key_values=None,
                        use_cache=True,
                        output_attentions=True,
                    )
                past_key_values = model_outputs.past_key_values
                next_logits = model_outputs.logits[:, -1, :]
                logical_len = past_key_values[0][0].size(2)
                image_start, image_token_len = infer_actual_image_span(
                    base_ids, past_key_values
                )
                kv_cache = build_kv_cache(
                    args,
                    model_name,
                    image_start=image_start,
                    image_token_len=image_token_len,
                )
                past_key_values = apply_cache(
                    kv_cache,
                    past_key_values,
                    logical_len,
                    model_outputs.attentions,
                )
            else:
                new_turn_ids = require_prefix(
                    base_ids,
                    committed_full_ids,
                    f"item {item_idx}, turn {turn_idx}",
                )
                if new_turn_ids.shape[1] == 0:
                    raise RuntimeError(
                        f"item {item_idx}, turn {turn_idx}: no new user tokens."
                    )
                with torch.no_grad():
                    next_logits, past_key_values, logical_len = feed_tokens(
                        model,
                        new_turn_ids,
                        past_key_values,
                        kv_cache,
                        logical_len,
                    )

            (
                generated,
                generated_ids,
                committed_stop,
                stop_reason,
                next_logits,
                past_key_values,
                logical_len,
                generation_steps,
            ) = generate_one_turn(
                model,
                tokenizer,
                conv,
                next_logits,
                past_key_values,
                kv_cache,
                logical_len,
                args.max_token_generated,
                context_length,
                args.temperature,
                args.top_p,
            )

            conv_with_answer = conv.copy()
            conv_with_answer.messages[-1][-1] = generated
            prompt_with_answer = conv_with_answer.get_prompt()
            prompt_through_answer = answer_only_prompt(
                base_prompt, prompt_with_answer, generated
            )
            ids_through_answer = tokenize_prompt(
                prompt_through_answer, tokenizer, model.device
            )
            ids_with_separator = tokenize_prompt(
                prompt_with_answer, tokenizer, model.device
            )
            rendered_answer_ids = require_prefix(
                ids_through_answer,
                base_ids,
                f"item {item_idx}, turn {turn_idx}, generated answer",
            )
            if not torch.equal(rendered_answer_ids, generated_ids):
                raise RuntimeError(
                    f"item {item_idx}, turn {turn_idx}: generated answer was not "
                    "tokenization-stable in the conversation template."
                )
            separator_ids = require_prefix(
                ids_with_separator,
                ids_through_answer,
                f"item {item_idx}, turn {turn_idx}, separator",
            )

            already_committed = len(committed_stop)
            if already_committed:
                stop_tensor = torch.tensor(
                    committed_stop,
                    dtype=torch.long,
                    device=separator_ids.device,
                ).unsqueeze(0)
                if (
                    separator_ids.shape[1] < already_committed
                    or not torch.equal(
                        separator_ids[:, :already_committed], stop_tensor
                    )
                ):
                    raise RuntimeError(
                        f"item {item_idx}, turn {turn_idx}: generated stop tokens "
                        "do not match the conversation separator."
                    )
            remaining_separator = separator_ids[:, already_committed:]
            if remaining_separator.shape[1] > 0:
                with torch.no_grad():
                    next_logits, past_key_values, logical_len = feed_tokens(
                        model,
                        remaining_separator,
                        past_key_values,
                        kv_cache,
                        logical_len,
                    )

            past_key_values = finalize_turn_cache(kv_cache, past_key_values)
            conv = conv_with_answer
            committed_full_ids = ids_with_separator

            generated_norm = normalize_text(generated, args.score_max_words)
            reference_norm = normalize_text(reference, args.score_max_words)
            if generated_norm and reference_norm:
                score = rouge_scorer.get_scores(
                    generated_norm, reference_norm
                )[0]["rouge-l"]["f"]
            else:
                score = 0.0
            all_turn_scores.append(score)
            per_turn_scores.setdefault(turn_idx + 1, []).append(score)
            item_output["turns"].append(
                {
                    "turn": turn_idx + 1,
                    "question": original_question,
                    "reference": reference,
                    "prediction": generated,
                    "rouge_l_f": score,
                    "generated_answer_tokens": int(generated_ids.shape[1]),
                    "generation_steps_including_stop": generation_steps,
                    "stop_reason": stop_reason,
                }
            )

            if args.debug:
                print(
                    f"[item {item_idx} turn {turn_idx + 1}] "
                    f"answer_tokens={generated_ids.shape[1]}, stop={stop_reason}, "
                    f"logical={logical_len}, physical_KV={past_key_values[0][0].size(2)}"
                )

        outputs_json.append(item_output)
        del past_key_values, kv_cache, image_tensor
        if model.device.type == "mps":
            torch.mps.empty_cache()

    average = sum(all_turn_scores) / len(all_turn_scores) if all_turn_scores else 0.0
    per_turn_average = {
        str(turn): sum(scores) / len(scores)
        for turn, scores in sorted(per_turn_scores.items())
    }
    with open(score_path, "a") as f:
        f.write(f"{average}\n")
    with open(pred_path, "w") as f:
        json.dump(
            {
                "method": args.method,
                "ratio": args.ratio,
                "max_token_generated_per_turn": args.max_token_generated,
                "free_generation": args.max_token_generated is None,
                "temperature": args.temperature,
                "model_context_length": context_length,
                "avg_rouge_l_f": average,
                "per_turn_rouge_l_f": per_turn_average,
                "num_items": len(outputs_json),
                "num_turns_scored": len(all_turn_scores),
                "results": outputs_json,
            },
            f,
            indent=2,
        )

    print(f"Average ROUGE-L F1 over {len(all_turn_scores)} turns: {average:.6f}")
    for turn, score in per_turn_average.items():
        print(f"Turn {turn} ROUGE-L F1: {score:.6f}")
    print(f"Saved score to: {score_path}")
    print(f"Saved predictions to: {pred_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="./models/llava-v1.5-7b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument(
        "--data-path", type=str, default="./playground/data/mm-vet/mm-vet.json"
    )
    parser.add_argument(
        "--image-path", type=str, default="./playground/data/mm-vet/images"
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--conv-mode", type=str, default=None)
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
        default=None,
        metavar="FREE|N|Nk",
        help=(
            "Maximum generated tokens for each turn. Default: free until "
            "EOS/separator or remaining model-context capacity."
        ),
    )
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--image-aspect-ratio", type=str, default="pad")
    parser.add_argument("--start-size", type=int, default=1)
    parser.add_argument("--recent-size", type=int, default=2047)
    parser.add_argument("--eval-samples", type=int, default=218)
    parser.add_argument("--exp-name", type=str, default="")
    parser.add_argument(
        "--method",
        type=str,
        default="turn_conditioned_layerwise_merge",
        choices=[
            "turn_conditioned_layerwise_merge",
            "turn_conditioned_layerwise_eviction",
            "tgv_kv",
            "local_cache",
            "h2o",
            "elastic_cache",
        ],
    )
    parser.add_argument(
        "--ratio",
        type=float,
        default=0.2,
        help="Pruned fraction: 0.8 means prune 80%% and retain about 20%%.",
    )
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.8)
    parser.add_argument("--cross-turn-weight", type=float, default=1.0)
    parser.add_argument("--output-dir", type=str, default="logs_rouge_llava")
    parser.add_argument("--score-max-words", type=int, default=256)
    args = parser.parse_args()
    main(args)
