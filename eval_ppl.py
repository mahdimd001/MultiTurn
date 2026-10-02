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
import statistics
import time
from io import BytesIO

import requests
import torch
from PIL import Image
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import CausalLMOutputWithPast
from tqdm import tqdm

# Non-multimodal baselines remain unchanged; the two multimodal classes
# come from the corrected multi-turn implementation.
from kv_cache import (
    ElasticCache,
    H2OCache,
    HelasticCache,
    LocalCache,
    MultimodalElasticCache,
    MultimodalHelasticCache,
    TurnIsolatedMultimodalCache,
    PersistentVisualTurnIsolatedCache,
)

from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.mm_utils import (
    get_model_name_from_path,
    process_images,
    tokenizer_image_token,
)
from llava.model.builder import load_pretrained_model
from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM
from kv_cache import (
    TGVKVCache,
    enable_legacy_tgv_attention,
    set_legacy_logical_position_limit,
)


def resolve_device(requested="auto"):
    requested = str(requested or "auto").lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    return torch.device(requested)


def inference_dtype(device):
    device = torch.device(device)
    if device.type == "cpu":
        return torch.float32
    if device.type == "mps":
        return torch.float16
    return torch.bfloat16


def synchronize(device):
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def _nova_llava_forward(
    self,
    input_ids=None,
    attention_mask=None,
    position_ids=None,
    past_key_values=None,
    inputs_embeds=None,
    labels=None,
    use_cache=None,
    output_attentions=None,
    output_hidden_states=None,
    images=None,
    return_dict=None,
):
    """LLaVA-1.5 forward with explicit logical positions after KV pruning."""
    output_attentions = (
        output_attentions
        if output_attentions is not None
        else self.config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states
        if output_hidden_states is not None
        else self.config.output_hidden_states
    )
    return_dict = (
        return_dict if return_dict is not None else self.config.use_return_dict
    )
    (
        input_ids,
        attention_mask,
        past_key_values,
        inputs_embeds,
        labels,
    ) = self.prepare_inputs_labels_for_multimodal(
        input_ids,
        attention_mask,
        past_key_values,
        labels,
        images,
    )
    outputs = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )
    logits = self.lm_head(outputs[0])
    loss = None
    if labels is not None:
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = CrossEntropyLoss()(
            shift_logits.view(-1, self.config.vocab_size),
            shift_labels.view(-1).to(shift_logits.device),
        )
    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output
    return CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )


# Keep the Nova transfer self-contained: older LLaVA checkouts do not forward
# explicit position_ids, which compressed multi-turn caches require.
LlavaLlamaForCausalLM.forward = _nova_llava_forward


def kv_cache_bytes(past_key_values):
    """Return bytes occupied by the key/value tensors currently retained."""
    if past_key_values is None:
        return 0
    return sum(
        tensor.numel() * tensor.element_size()
        for layer in past_key_values
        for tensor in layer[:2]
    )


class RuntimeMetrics:
    """Collect synchronized PPL-pass latency and memory measurements.

    These are teacher-forced measurements. In particular, ``ttft`` is the
    multimodal model-prefill component of time-to-first-token, and ``tpot`` is
    the time to commit known answer tokens, not free-running sampled decoding.
    """

    def __init__(self, device, enabled=False):
        self.device = torch.device(device)
        self.enabled = enabled
        self.item_records = []
        self._item = None
        self.peak_materialized_kv_bytes = 0
        self.peak_retained_kv_bytes = 0
        self.cuda_allocated_after_model_load = None

    def reset_device_peak(self):
        if self.enabled and self.device.type == "cuda":
            synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
            self.cuda_allocated_after_model_load = torch.cuda.memory_allocated(
                self.device
            )

    def begin_item(self, item_index, item_id=None):
        if not self.enabled:
            return
        self._item = {
            "id": item_id if item_id is not None else item_index,
            "item_index": item_index,
            "ttft_model_seconds": None,
            "prompt_forward_seconds": [],
            "answer_forward_seconds": [],
            "answer_tokens": 0,
            "peak_materialized_kv_bytes": 0,
            "peak_retained_kv_bytes": 0,
        }

    def timed_forward(self, model, phase, **kwargs):
        if not self.enabled:
            return model(**kwargs)
        synchronize(self.device)
        started = time.perf_counter()
        outputs = model(**kwargs)
        synchronize(self.device)
        elapsed = time.perf_counter() - started
        if phase == "prefill":
            self._item["ttft_model_seconds"] = elapsed
        elif phase == "answer":
            self._item["answer_forward_seconds"].append(elapsed)
            self._item["answer_tokens"] += int(outputs.logits.shape[1])
        else:
            self._item["prompt_forward_seconds"].append(elapsed)
        return outputs

    def observe_kv(self, past_key_values, stage="retained"):
        if not self.enabled:
            return
        size = kv_cache_bytes(past_key_values)
        if stage == "materialized":
            self.peak_materialized_kv_bytes = max(
                self.peak_materialized_kv_bytes, size
            )
            self._item["peak_materialized_kv_bytes"] = max(
                self._item["peak_materialized_kv_bytes"], size
            )
        else:
            self.peak_retained_kv_bytes = max(self.peak_retained_kv_bytes, size)
            self._item["peak_retained_kv_bytes"] = max(
                self._item["peak_retained_kv_bytes"], size
            )

    def finish_item(self, mean_nll, ppl):
        if not self.enabled:
            return
        answer_seconds = sum(self._item["answer_forward_seconds"])
        answer_tokens = self._item["answer_tokens"]
        self._item.update(
            {
                "mean_nll": float(mean_nll),
                "ppl": float(ppl),
                "teacher_forced_tpot_seconds": (
                    answer_seconds / answer_tokens if answer_tokens else None
                ),
                "teacher_forced_answer_tokens_per_second": (
                    answer_tokens / answer_seconds if answer_seconds else None
                ),
            }
        )
        self.item_records.append(self._item)
        self._item = None

    @staticmethod
    def _mean(values):
        values = [value for value in values if value is not None]
        return statistics.fmean(values) if values else None

    def summary(self):
        ttft = [x["ttft_model_seconds"] for x in self.item_records]
        tpot = [x["teacher_forced_tpot_seconds"] for x in self.item_records]
        throughput = [
            x["teacher_forced_answer_tokens_per_second"] for x in self.item_records
        ]
        result = {
            "measurement_scope": "teacher_forced_ppl_pass",
            "ttft_scope": "multimodal_model_prefill_only",
            "tpot_scope": "known_answer_token_commit_forwards",
            "num_items": len(self.item_records),
            "mean_ttft_model_seconds": self._mean(ttft),
            "mean_teacher_forced_tpot_seconds": self._mean(tpot),
            "mean_teacher_forced_answer_tokens_per_second": self._mean(throughput),
            "peak_materialized_kv_bytes": self.peak_materialized_kv_bytes,
            "peak_retained_kv_bytes": self.peak_retained_kv_bytes,
        }
        if self.device.type == "cuda":
            result.update(
                {
                    "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(
                        self.device
                    ),
                    "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(
                        self.device
                    ),
                    "cuda_allocated_after_model_load_bytes": (
                        self.cuda_allocated_after_model_load
                    ),
                }
            )
        return result


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


def forward_one_text_token(
    model, token_ids, past_key_values, kv_cache, num_of_token,
    metrics=None, phase="prompt",
):
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

    logical_position_ids = torch.tensor(
        [[num_of_token]], dtype=torch.long, device=token_ids.device
    )
    set_legacy_logical_position_limit(model, num_of_token + 1)

    forward_kwargs = dict(
        input_ids=token_ids,
        position_ids=logical_position_ids,
        past_key_values=past_key_values,
        use_cache=True,
        output_attentions=True,
    )
    outputs = (
        metrics.timed_forward(model, phase, **forward_kwargs)
        if metrics is not None
        else model(**forward_kwargs)
    )

    # For a one-token text input this should increase the effective sequence by 1.
    num_of_token += outputs.logits.shape[1]
    past_key_values = outputs.past_key_values
    if metrics is not None:
        metrics.observe_kv(past_key_values, stage="materialized")

    # Let the experimental cache method update scores and/or prune after this token.
    past_key_values = apply_cache(
        kv_cache,
        past_key_values,
        num_of_token,
        outputs.attentions,
    )
    if metrics is not None:
        metrics.observe_kv(past_key_values)

    last_logits = outputs.logits[:, -1, :]
    return last_logits, past_key_values, num_of_token


def feed_text_tokens(
    model, token_ids, past_key_values, kv_cache, num_of_token,
    metrics=None, phase="prompt",
):
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
            metrics=metrics,
            phase=phase,
        )

    return last_logits, past_key_values, num_of_token


def forward_text_block(
    model, token_ids, past_key_values, kv_cache, num_of_token,
    metrics=None, phase="prompt",
):
    """Commit a text block in one forward for boundary-only cache methods.

    Turn-isolated caches collect attention evidence during a turn and compress
    only in ``finalize_turn``.  A causal block forward is therefore equivalent
    to one-token forwards, but avoids launching every LLaMA layer separately
    for every token (which is especially costly on MPS).

    Returns all block logits so answer-token NLLs can still be computed with
    the exact teacher-forcing shift.
    """
    block_len = token_ids.shape[1]
    if block_len == 0:
        raise ValueError("forward_text_block requires at least one token.")

    position_ids = torch.arange(
        num_of_token,
        num_of_token + block_len,
        dtype=torch.long,
        device=token_ids.device,
    ).unsqueeze(0)
    set_legacy_logical_position_limit(model, num_of_token + block_len)

    forward_kwargs = dict(
        input_ids=token_ids,
        position_ids=position_ids,
        past_key_values=past_key_values,
        use_cache=True,
        output_attentions=True,
    )
    outputs = (
        metrics.timed_forward(model, phase, **forward_kwargs)
        if metrics is not None
        else model(**forward_kwargs)
    )
    num_of_token += block_len
    if metrics is not None:
        metrics.observe_kv(outputs.past_key_values, stage="materialized")
    past_key_values = apply_cache(
        kv_cache,
        outputs.past_key_values,
        num_of_token,
        outputs.attentions,
    )
    if metrics is not None:
        metrics.observe_kv(past_key_values)
    return outputs.logits, past_key_values, num_of_token


def uses_turn_boundary_compression(kv_cache):
    """Whether text can be safely committed as causal blocks."""
    return kv_cache is not None and hasattr(kv_cache, "finalize_turn")


def commit_text_tokens(
    model, token_ids, past_key_values, kv_cache, num_of_token,
    metrics=None, phase="prompt",
):
    """Commit known text using the fastest path allowed by the cache policy."""
    if uses_turn_boundary_compression(kv_cache):
        logits, past_key_values, num_of_token = forward_text_block(
            model, token_ids, past_key_values, kv_cache, num_of_token,
            metrics=metrics, phase=phase,
        )
        return logits[:, -1, :], past_key_values, num_of_token
    return feed_text_tokens(
        model, token_ids, past_key_values, kv_cache, num_of_token,
        metrics=metrics, phase=phase,
    )


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
    # Every compressed cache needs explicit logical RoPE positions. TGV also
    # needs this attention implementation for unequal per-layer cache lengths.
    if args.ratio > 0:
        enable_legacy_tgv_attention()

    device = resolve_device(args.device)
    args.device = str(device)
    dtype = inference_dtype(device)
    print(f"Using device: {device} ({dtype})")

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

    metrics = RuntimeMetrics(device, enabled=args.collect_metrics)
    # Record the post-load baseline and reset peak tracking before evaluation.
    # CUDA peak allocation still includes the resident model, as standard VRAM
    # reporting does; the separate baseline makes incremental overhead visible.
    metrics.reset_device_peak()

    # We compute cross entropy token-by-token only on GOLD ASSISTANT ANSWERS.
    loss_fn = CrossEntropyLoss(reduction="none")
    nlls = []

    os.makedirs("logs_ppl_llava", exist_ok=True)
    data = data[: args.eval_samples]

    # ================================================================
    # Each item is ONE continuous conversation associated with ONE image.
    # Cache state is reset only when we move to the next dataset item.
    # ================================================================
    for item_index, item in enumerate(tqdm(data)):
        if "turns" not in item or len(item["turns"]) == 0:
            continue
        metrics.begin_item(item_index, item.get("id", item.get("image")))
        item_nll_start = len(nlls)

        # ----------------
        # Prepare the image
        # ----------------
        image_path = os.path.join(args.image_path, item["image"])
        image = load_image(image_path)
        image_tensor = process_images([image], image_processor, args)
        image_tensor = image_tensor.to(model.device, dtype=dtype)

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
                set_legacy_logical_position_limit(model, 0)
                with torch.no_grad():
                    outputs = metrics.timed_forward(
                        model,
                        "prefill",
                        input_ids=base_ids,
                        images=image_tensor,
                        past_key_values=None,
                        use_cache=True,
                        output_attentions=True,
                    )

                past_key_values = outputs.past_key_values
                metrics.observe_kv(past_key_values, stage="materialized")
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
                metrics.observe_kv(past_key_values)

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
                    next_token_logits, past_key_values, num_of_token = commit_text_tokens(
                        model,
                        new_turn_ids,
                        past_key_values,
                        kv_cache,
                        num_of_token,
                        metrics=metrics,
                        phase="prompt",
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
                        next_token_logits, past_key_values, num_of_token = commit_text_tokens(
                            model,
                            separator_ids,
                            past_key_values,
                            kv_cache,
                            num_of_token,
                            metrics=metrics,
                            phase="prompt",
                        )

                past_key_values = finalize_turn_cache(kv_cache, past_key_values)
                metrics.observe_kv(past_key_values)
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
            if uses_turn_boundary_compression(kv_cache):
                # The context logits predict answer token 0. Within the causal
                # answer block, logit i predicts answer token i+1.
                with torch.no_grad():
                    answer_logits, past_key_values, num_of_token = forward_text_block(
                        model,
                        answer_ids,
                        past_key_values,
                        kv_cache,
                        num_of_token,
                        metrics=metrics,
                        phase="answer",
                    )
                shifted_logits = torch.cat(
                    [next_token_logits.unsqueeze(1), answer_logits[:, :-1, :]],
                    dim=1,
                )
                block_nll = loss_fn(
                    shifted_logits.reshape(-1, shifted_logits.shape[-1]),
                    answer_ids.reshape(-1),
                )
                nlls.extend(block_nll.detach().cpu().unbind())
                next_token_logits = answer_logits[:, -1, :]
            else:
                for answer_pos in range(answer_ids.shape[1]):
                    gold_token = answer_ids[:, answer_pos]

                    # Score this gold answer token using logits from the context
                    # immediately BEFORE that token.
                    token_nll = loss_fn(next_token_logits, gold_token)
                    nlls.append(token_nll.detach().cpu())

                    # Feed the gold token so the next prediction sees it in the KV.
                    with torch.no_grad():
                        next_token_logits, past_key_values, num_of_token = forward_one_text_token(
                            model,
                            answer_ids[:, answer_pos : answer_pos + 1],
                            past_key_values,
                            kv_cache,
                            num_of_token,
                            metrics=metrics,
                            phase="answer",
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
                    next_token_logits, past_key_values, num_of_token = commit_text_tokens(
                        model,
                        separator_ids,
                        past_key_values,
                        kv_cache,
                        num_of_token,
                        metrics=metrics,
                        phase="prompt",
                    )

            past_key_values = finalize_turn_cache(kv_cache, past_key_values)
            metrics.observe_kv(past_key_values)
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

        item_values = nlls[item_nll_start:]
        if item_values:
            item_losses = torch.cat([value.reshape(-1) for value in item_values])
            item_mean_nll = item_losses.mean().item()
            metrics.finish_item(
                item_mean_nll, torch.exp(item_losses.mean()).item()
            )

        del past_key_values, kv_cache, image_tensor
        if device.type == "mps":
            torch.mps.empty_cache()

    if not nlls:
        raise RuntimeError("No answer tokens were evaluated; cannot compute perplexity.")

    # Concatenate scalar-per-token losses and compute ordinary token-level PPL.
    mean_nll = torch.cat([x.reshape(-1) for x in nlls]).mean()
    ppl = torch.exp(mean_nll)

    output_path = os.path.join("logs_ppl_llava", f"{args.exp_name}.txt")
    with open(output_path, "a") as f:
        f.write(f"{ppl.item()}\n")

    if args.collect_metrics:
        results_dir = args.metrics_output_dir
        os.makedirs(results_dir, exist_ok=True)
        summary = metrics.summary()
        summary.update(
            {
                "experiment": args.exp_name,
                "method": args.method,
                "ratio": args.ratio,
                "overall_mean_nll": mean_nll.item(),
                "overall_ppl": ppl.item(),
                "device": str(device),
            }
        )
        metrics_path = os.path.join(results_dir, f"{args.exp_name}.metrics.json")
        items_path = os.path.join(results_dir, f"{args.exp_name}.items.jsonl")
        with open(metrics_path, "w") as f:
            json.dump(summary, f, indent=2)
        with open(items_path, "w") as f:
            for record in metrics.item_records:
                f.write(json.dumps(record) + "\n")
        print(f"Metrics: {metrics_path}")
        print(f"Per-item metrics: {items_path}")

    print(f"Perplexity: {ppl.item():.6f}")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # LLaVA/model arguments retained from the original evaluator.
    parser.add_argument("--model-path", type=str, default="./models/llava-v1.5-7b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--data-path", type=str, default="./playground/data/mm-vet/mm-vet.json")
    parser.add_argument("--image-path", type=str, default="./playground/data/mm-vet/images")
    parser.add_argument("--device", type=str, default="auto", help="auto, mps, cpu, or cuda")
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
    parser.add_argument(
        "--collect-metrics",
        action="store_true",
        help=(
            "Collect synchronized prefill/teacher-forced token latency, "
            "throughput, peak KV bytes, and CUDA peak allocation."
        ),
    )
    parser.add_argument(
        "--metrics-output-dir",
        default="evaluation/results",
        help="Directory for JSON summary and per-item JSONL metrics.",
    )

    args = parser.parse_args()
    main(args)
