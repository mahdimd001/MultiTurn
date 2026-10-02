"""
Correct multi-turn multimodal perplexity evaluation for LLaVA + KV-cache pruning.

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

    Turn 1: image + Q1 -> score A1
    Turn 2:         Q2 -> score A2, reusing the KV cache from Turn 1
    Turn 3:         Q3 -> score A3, reusing the same continuing KV cache

Only the first turn sends the image to LLaVA.  Later turns send ONLY the newly
appended text tokens.  Previous questions/answers are already represented by
past_key_values and are therefore never fed a second time.

This version uses the corrected multimodal cache implementations from
kv_cache_multiturn_fixed.py.  The non-multimodal baseline classes are still
imported unchanged from the original kv_cache.py through that module.
"""

import argparse
import json
import os
from io import BytesIO

import requests
import torch
from PIL import Image
from torch.nn import CrossEntropyLoss
from tqdm import tqdm

# Non-multimodal baselines remain unchanged; the two multimodal classes
# come from the corrected multi-turn implementation.
from kv_cache_nova_similarity_eviction import (
    ElasticCache,
    H2OCache,
    HelasticCache,
    LocalCache,
    MultimodalElasticCache,
    MultimodalElasticSimilarityCache,
    MultimodalHelasticCache,
    TurnIsolatedMultimodalCache,
    PersistentVisualTurnIsolatedCache,
    DelayedFirstCompressionSimilarityCache,
    DelayedFirstCompressionEvictionCache,
)

from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.mm_utils import (
    get_model_name_from_path,
    process_images,
    tokenizer_image_token,
)
from llava.model.builder import load_pretrained_model
from tgv_kv_legacy_nova import (
    TGVKVCache,
    enable_legacy_tgv_attention,
    set_legacy_logical_position_limit,
)

def load_image(image_file):
    """Load a local image or an image URL and always return RGB."""
    if image_file.startswith("http://") or image_file.startswith("https://"):
        response = requests.get(image_file, timeout=30)
        response.raise_for_status()
        return Image.open(BytesIO(response.content)).convert("RGB")

    return Image.open(image_file).convert("RGB")


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


def common_prefix_length(a, b):
    """Return the identical leading-token count of two [1, N] tensors."""
    length = min(a.shape[1], b.shape[1])
    if length == 0:
        return 0
    mismatch = (a[:, :length] != b[:, :length]).nonzero(as_tuple=False)
    return int(mismatch[0, 1].item()) if mismatch.numel() else length


def build_cache(
    args,
    model_name,
    image_start=None,
    image_token_len=0,
    initial_protected_text_len=0,
):
    """Create a fresh cache object for one dataset item/conversation."""
    layer_num = 32 if "7b" in model_name.lower() else 40
    common = dict(
        start_size=args.start_size,
        recent_size=args.recent_size,
        k_seq_dim=2,
        v_seq_dim=2,
        ratio=args.ratio,
    )

    if args.method == "elastic":
        return ElasticCache(layer_num=layer_num, **common)

    if args.method == "helastic":
        return HelasticCache(layer_num=layer_num, **common)

    if args.method == "multimodal_elastic":
        return MultimodalElasticCache(
            layer_num=layer_num,
            image_start=image_start,
            image_token_len=image_token_len,
            attn_weight=args.alpha,
            conn_weight=args.beta,
            **common,
        )

    if args.method == "multimodal_elastic_similarity":
        return MultimodalElasticSimilarityCache(
            layer_num=layer_num,
            image_start=image_start,
            image_token_len=image_token_len,
            attn_weight=args.alpha,
            conn_weight=args.beta,
            **common,
        )

    if args.method == "delayed_first_compression_similarity":
        return DelayedFirstCompressionSimilarityCache(
            layer_num=layer_num,
            image_start=image_start,
            image_token_len=image_token_len,
            attn_weight=args.alpha,
            conn_weight=args.beta,
            **common,
        )

    if args.method == "delayed_first_compression_eviction":
        return DelayedFirstCompressionEvictionCache(
            layer_num=layer_num,
            image_start=image_start,
            image_token_len=image_token_len,
            attn_weight=args.alpha,
            conn_weight=args.beta,
            **common,
        )

    if args.method == "turn_isolated_mm":
        return TurnIsolatedMultimodalCache(
            layer_num=layer_num,
            image_start=image_start,
            image_token_len=image_token_len,
            initial_protected_text_len=initial_protected_text_len,
            attn_weight=args.alpha,
            conn_weight=args.beta,
            **common,
        )

    if args.method == "persistent_visual_mm":
        return PersistentVisualTurnIsolatedCache(
            layer_num=layer_num,
            image_start=image_start,
            image_token_len=image_token_len,
            initial_protected_text_len=initial_protected_text_len,
            attn_weight=args.alpha,
            conn_weight=args.beta,
            **common,
        )

    if args.method == "multimodal_helastic":
        return MultimodalHelasticCache(
            layer_num=layer_num,
            image_start=image_start,
            image_token_len=image_token_len,
            attn_weight=args.alpha,
            conn_weight=args.beta,
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

    if args.method == "local":
        return LocalCache(**common)

    if args.method == "h2o":
        return H2OCache(**common)

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


def finalize_turn_cache(kv_cache, past_key_values):
    """Run boundary-only compression for a turn-isolated cache."""
    if kv_cache is None or not hasattr(kv_cache, "finalize_turn"):
        return past_key_values
    return kv_cache.finalize_turn(past_key_values)


def forward_one_text_token(model, token_ids, past_key_values, kv_cache, num_of_token):
    """
    Feed exactly ONE already-known text token into the continuing conversation.

    Feeding later text one token at a time has two advantages here:
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

    # TGV-KV can assign different physical cache lengths to different layers.
    # Its legacy attention compatibility path therefore needs the logical RoPE
    # limit explicitly for every appended token. Other methods continue using
    # the normal Nova/Transformers RoPE patch already installed in the env.
    if isinstance(kv_cache, TGVKVCache):
        set_legacy_logical_position_limit(model, num_of_token + 1)

    outputs = model(
        input_ids=token_ids,
        position_ids=logical_position_ids,
        past_key_values=past_key_values,
        use_cache=True,
        output_attentions=True,
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
    Construct the prompt ending exactly at the gold answer, before its separator.

    LLaVA conversation templates append formatting after an assistant answer
    (for example a separator or EOS marker).  Those separator tokens must be in
    the KV cache before the next user turn, but they should not be counted as
    answer tokens when computing answer perplexity.

    We therefore split the rendered assistant continuation into:
        [answer portion] [template separator portion]
    """
    if not prompt_with_answer.startswith(base_prompt):
        raise RuntimeError(
            "The conversation template did not extend the assistant-empty prompt "
            "when the gold answer was inserted."
        )

    continuation = prompt_with_answer[len(base_prompt) :]
    answer_pos = continuation.find(answer)

    if answer_pos < 0:
        raise RuntimeError(
            "Could not locate the gold answer inside the rendered assistant turn."
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


def main(args):
    # ----------------------------
    # Load dataset and LLaVA model
    # ----------------------------
    # TGV-KV reallocates the visual budget across layers, so layers can have
    # unequal physical KV lengths. Transformers 4.31 assumes equal lengths;
    # enable the compatibility attention implementation only for this method.
    if args.method == "tgv_kv":
        enable_legacy_tgv_attention()

    # Nova/GPU path: keep device handling simple and explicit.
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"--device {args.device} was requested, but torch.cuda.is_available() is False."
        )

    device = torch.device(args.device)
    print(f"Using device: {device}")

    with open(args.data_path, "r") as f:
        data = json.load(f)

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, _ = load_pretrained_model(
        args.model_path,
        args.model_base,
        model_name,
        args.load_8bit,
        args.load_4bit,
        device=args.device,
    )

    args.conv_mode = choose_conversation_mode(model_name, args.conv_mode)

    # We compute cross entropy token-by-token only on GOLD ASSISTANT ANSWERS.
    loss_fn = CrossEntropyLoss(reduction="none")
    nlls = []
    # Token-level NLLs grouped by 1-based conversation turn.
    # This adds no model passes; it only groups losses already computed.
    turn_nlls = {}

    os.makedirs("logs_ppl_llava", exist_ok=True)
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
        static_ids = tokenize_conversation_prompt(
            conv.get_prompt(), tokenizer, model.device
        )
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
        # gold answer token.
        next_token_logits = None

        for turn_id, turn in enumerate(item["turns"]):
            question = turn["question"]
            answer = turn["answer"]

            # ============================================================
            # 1. Add the current USER turn to the logical conversation.
            #    The image placeholder is present ONLY in the first turn.
            # ============================================================
            if turn_id == 0:
                if DEFAULT_IMAGE_TOKEN not in question:
                    question = question + "\n" + DEFAULT_IMAGE_TOKEN

                conv.append_message(conv.roles[0], question)
                conv.append_message(conv.roles[1], None)
                base_prompt = conv.get_prompt()
                base_ids = tokenize_conversation_prompt(
                    base_prompt, tokenizer, model.device
                )

                # --------------------------------------------------------
                # FIRST AND ONLY multimodal prefill for this conversation.
                # --------------------------------------------------------
                with torch.no_grad():
                    outputs = model(
                        input_ids=base_ids,
                        images=image_tensor,
                        past_key_values=None,
                        use_cache=True,
                        output_attentions=True,
                    )

                past_key_values = outputs.past_key_values
                next_token_logits = outputs.logits[:, -1, :]

                # Use the expanded model sequence length rather than the raw
                # input_ids length, because one image placeholder expands into
                # many visual tokens inside LLaVA.
                effective_prefill_len = past_key_values[0][0].size(2)
                num_of_token = effective_prefill_len

                # Infer where the ACTUAL visual KV block is located.
                image_start, image_token_len = infer_actual_image_span(
                    base_ids, past_key_values, k_seq_dim=2
                )
                initial_protected_text_len = common_prefix_length(
                    base_ids, static_ids
                )

                # Build the cache only AFTER the first forward pass so that the
                # multimodal cache receives the true visual-token length.
                kv_cache = build_cache(
                    args,
                    model_name,
                    image_start=image_start,
                    image_token_len=image_token_len,
                    initial_protected_text_len=initial_protected_text_len,
                )

                if args.debug:
                    print(
                        f"\n[item {item_index} turn 0] "
                        f"text/input ids={base_ids.shape[1]}, "
                        f"expanded KV={effective_prefill_len}, "
                        f"image_start={image_start}, "
                        f"image_token_len={image_token_len}, "
                        f"protected_static_prefix={initial_protected_text_len}"
                    )

                # Apply pruning/merging to the first multimodal prefill.
                past_key_values = apply_cache(
                    kv_cache,
                    past_key_values,
                    num_of_token,
                    outputs.attentions,
                )

            else:
                # ---------------------------------------------------------
                # Previous answer + separator are already in past_key_values.
                # Append ONLY the new user turn and assistant prefix.
                # ---------------------------------------------------------
                conv.append_message(conv.roles[0], question)
                conv.append_message(conv.roles[1], None)
                base_prompt = conv.get_prompt()
                base_ids = tokenize_conversation_prompt(
                    base_prompt, tokenizer, model.device
                )

                # Exact token suffix added since the previous turn was fully
                # committed.  This is the central fix for answer duplication.
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
            # 2. Derive the GOLD ANSWER TOKENS IN CONVERSATION CONTEXT.
            #
            # Do not use tokenizer.encode(answer) here.  Standalone answer
            # tokenization can differ at whitespace/template boundaries.
            # ============================================================
            conv_with_answer = conv.copy()
            conv_with_answer.messages[-1][-1] = answer
            prompt_with_answer = conv_with_answer.get_prompt()

            # Prompt ending exactly after answer text, before separator/EOS.
            prompt_through_answer = answer_only_prompt(
                base_prompt,
                prompt_with_answer,
                answer,
            )

            ids_through_answer = tokenize_conversation_prompt(
                prompt_through_answer, tokenizer, model.device
            )
            ids_with_answer_and_separator = tokenize_conversation_prompt(
                prompt_with_answer, tokenizer, model.device
            )

            # Gold answer tokens are exactly what was appended after the
            # assistant-empty base prompt and before the turn separator.
            answer_ids = require_prefix(
                ids_through_answer,
                base_ids,
                description=f"item {item_index}, turn {turn_id}, answer",
            )

            # The remaining tokens are conversation-template separators/EOS.
            separator_ids = require_prefix(
                ids_with_answer_and_separator,
                ids_through_answer,
                description=f"item {item_index}, turn {turn_id}, separator",
            )

            if answer_ids.shape[1] == 0:
                # Empty answers cannot contribute an answer-token NLL, but the
                # template separator still needs to be committed for next turn.
                if separator_ids.shape[1] > 0:
                    with torch.no_grad():
                        next_token_logits, past_key_values, num_of_token = feed_text_tokens(
                            model,
                            separator_ids,
                            past_key_values,
                            kv_cache,
                            num_of_token,
                        )

                past_key_values = finalize_turn_cache(kv_cache, past_key_values)
                if args.debug and hasattr(kv_cache, "last_finalize_stats"):
                    print(f"[item {item_index} turn {turn_id}] finalize: "
                          f"{kv_cache.last_finalize_stats}")

                conv = conv_with_answer
                committed_full_ids = ids_with_answer_and_separator
                continue

            # ============================================================
            # 3. Teacher-force the answer WITHOUT duplicating it.
            #
            # Before feeding answer token i, the current logits predict it.
            # Then token i is fed exactly once so it becomes part of the KV
            # cache and produces logits for token i+1.
            # ============================================================
            for answer_pos in range(answer_ids.shape[1]):
                gold_token = answer_ids[:, answer_pos]

                # Score this gold answer token using logits from the context
                # immediately BEFORE that token.
                token_nll = loss_fn(next_token_logits.float(), gold_token)
                token_nll_cpu = token_nll.detach().cpu()
                nlls.append(token_nll_cpu)
                turn_nlls.setdefault(turn_id + 1, []).append(token_nll_cpu)

                # Feed the gold token so the next prediction sees it in the KV.
                with torch.no_grad():
                    next_token_logits, past_key_values, num_of_token = forward_one_text_token(
                        model,
                        answer_ids[:, answer_pos : answer_pos + 1],
                        past_key_values,
                        kv_cache,
                        num_of_token,
                    )

            # ============================================================
            # 4. Commit the assistant-turn separator/EOS.
            #
            # The original implementation omitted this from the KV cache and
            # then tried to reconstruct later turns from strings.  Here every
            # logical conversation token is committed exactly once.
            # ============================================================
            if separator_ids.shape[1] > 0:
                with torch.no_grad():
                    next_token_logits, past_key_values, num_of_token = feed_text_tokens(
                        model,
                        separator_ids,
                        past_key_values,
                        kv_cache,
                        num_of_token,
                    )

            past_key_values = finalize_turn_cache(kv_cache, past_key_values)
            if args.debug and hasattr(kv_cache, "last_finalize_stats"):
                print(f"[item {item_index} turn {turn_id}] finalize: "
                      f"{kv_cache.last_finalize_stats}")

            # The model KV now represents the complete conversation through the
            # current gold answer and its separator.  Advance the logical state.
            conv = conv_with_answer
            committed_full_ids = ids_with_answer_and_separator

            if args.debug:
                physical_kv_len = past_key_values[0][0].size(2)
                print(
                    f"[item {item_index} turn {turn_id}] "
                    f"answer tokens={answer_ids.shape[1]}, "
                    f"separator tokens={separator_ids.shape[1]}, "
                    f"logical/effective processed={num_of_token}, "
                    f"physical KV={physical_kv_len}, "
                    f"position gap={num_of_token - physical_kv_len}"
                )

        del past_key_values, kv_cache, image_tensor
        if device.type == "mps":
            torch.mps.empty_cache()

    if not nlls:
        raise RuntimeError("No answer tokens were evaluated; cannot compute perplexity.")

    # Concatenate scalar-per-token losses and compute ordinary token-level PPL.
    mean_nll = torch.cat([x.reshape(-1) for x in nlls]).mean()
    ppl = torch.exp(mean_nll)

    # Per-turn PPL uses the same token-level definition as overall PPL, but
    # pools only answer-token NLLs from that turn across all evaluated items.
    per_turn_ppl = {}
    per_turn_token_count = {}
    for turn_number in sorted(turn_nlls):
        turn_losses = torch.cat([x.reshape(-1) for x in turn_nlls[turn_number]])
        per_turn_ppl[str(turn_number)] = float(torch.exp(turn_losses.mean()).item())
        per_turn_token_count[str(turn_number)] = int(turn_losses.numel())

    # Keep the old one-number text output for compatibility.
    output_path = os.path.join("logs_ppl_llava", f"{args.exp_name}.txt")
    with open(output_path, "a") as f:
        f.write(f"{ppl.item()}\n")

    # Also write a structured JSON result for paper/evaluation scripts.
    json_output_path = os.path.join("logs_ppl_llava", f"{args.exp_name}.json")
    result = {
        "experiment": args.exp_name,
        "method": args.method,
        "ratio": float(args.ratio),
        "requested_eval_samples": int(args.eval_samples),
        "evaluated_items": int(len(data)),
        "overall_ppl": float(ppl.item()),
        "overall_answer_tokens": int(sum(x.numel() for x in nlls)),
        "per_turn_ppl": per_turn_ppl,
        "per_turn_answer_tokens": per_turn_token_count,
        "alpha": float(args.alpha),
        "beta": float(args.beta),
    }
    with open(json_output_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Perplexity: {ppl.item():.6f}")
    for turn_number, turn_ppl in per_turn_ppl.items():
        print(f"Turn {turn_number} PPL: {turn_ppl:.6f}")
    print(f"Saved overall PPL to: {output_path}")
    print(f"Saved overall + per-turn PPL JSON to: {json_output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # LLaVA/model arguments retained from the original evaluator.
    parser.add_argument("--model-path", type=str, default="./models/llava-v1.5-7b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--data-path", type=str, default="./playground/data/mm-vet/mm-vet.json")
    parser.add_argument("--image-path", type=str, default="./playground/data/mm-vet/images")
    parser.add_argument("--device", type=str, default="cuda", help="Nova: cuda (default) or cuda:N")
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--image-aspect-ratio", type=str, default="pad")

    # Cache/evaluation arguments retained from the original evaluator.
    parser.add_argument("--start-size", type=int, default=1)
    parser.add_argument("--recent-size", type=int, default=2047)
    parser.add_argument("--eval-samples", type=int, default=100)
    parser.add_argument("--exp-name", type=str, default="llava-7b-multiturn-fixed")
    parser.add_argument(
        "--method",
        type=str,
        default="elastic",
        choices=[
            "elastic",
            "helastic",
            "multimodal_elastic",
            "multimodal_elastic_similarity",
            "delayed_first_compression_similarity",
            "delayed_first_compression_eviction",
            "multimodal_helastic",
            "turn_isolated_mm",
            "persistent_visual_mm",
            "tgv_kv",
            "local",
            "h2o",
        ],
    )
    parser.add_argument("--ratio", type=float, default=0.2)
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="Ordinary-attention weight for multimodal caches.")
    parser.add_argument("--beta", type=float, default=0.8,
                        help="Text-image connection weight for multimodal caches.")

    # Debug mode prints per-turn token counts and physical KV lengths.  This is
    # particularly useful while validating a new multi-turn dataset/cache.
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()
    main(args)
