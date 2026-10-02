"""TGV-KV adapted to the legacy Transformers 4.31 LLaVA implementation.

The cache policy follows the official TGV-KV implementation vendored under
``third_party/TGV-KV``.  Two compatibility changes are necessary here:

* legacy LLaVA exposes one negative image placeholder which expands internally,
  so the evaluator supplies the actual expanded visual-token span;
* TGV-KV's token-preservation reallocation gives layers different cache
  lengths.  Transformers 4.31 assumes all layers have the first layer's cache
  length, so ``enable_legacy_tgv_attention`` makes its LLaMA attention accept
  the correct mask length for each layer.
"""

import math

import numpy as np
import torch
import torch.nn.functional as F
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    apply_rotary_pos_emb,
    repeat_kv,
)


_ORIGINAL_LLAMA_ATTENTION_FORWARD = LlamaAttention.forward


def _tgv_compatible_llama_attention_forward(
    self,
    hidden_states,
    attention_mask=None,
    position_ids=None,
    past_key_value=None,
    output_attentions=False,
    use_cache=False,
):
    """Transformers 4.31 LLaMA attention with unequal-layer-cache support."""
    bsz, q_len, _ = hidden_states.size()

    if self.pretraining_tp > 1:
        kv_slice = (self.num_key_value_heads * self.head_dim) // self.pretraining_tp
        q_slices = self.q_proj.weight.split(
            (self.num_heads * self.head_dim) // self.pretraining_tp, dim=0
        )
        k_slices = self.k_proj.weight.split(kv_slice, dim=0)
        v_slices = self.v_proj.weight.split(kv_slice, dim=0)
        query_states = torch.cat(
            [F.linear(hidden_states, weight) for weight in q_slices], dim=-1
        )
        key_states = torch.cat(
            [F.linear(hidden_states, weight) for weight in k_slices], dim=-1
        )
        value_states = torch.cat(
            [F.linear(hidden_states, weight) for weight in v_slices], dim=-1
        )
    else:
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

    query_states = query_states.view(
        bsz, q_len, self.num_heads, self.head_dim
    ).transpose(1, 2)
    key_states = key_states.view(
        bsz, q_len, self.num_key_value_heads, self.head_dim
    ).transpose(1, 2)
    value_states = value_states.view(
        bsz, q_len, self.num_key_value_heads, self.head_dim
    ).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value[0].shape[-2]

    # position_ids are derived from the first layer by legacy LlamaModel.  A
    # more heavily pruned later layer can be shorter, so its rotary table must
    # still cover that shared logical position.
    # Legacy LlamaModel constructs this mask from layer 0. Its shape is already
    # a host-side integer and covers the shared position_ids. Using
    # position_ids.max().item() here would force an MPS/CUDA synchronization in
    # every layer for every generated token, making multi-turn decoding appear
    # to hang.
    rotary_seq_len = max(
        kv_seq_len, int(getattr(self, "_logical_position_limit", 0))
    )
    if attention_mask is not None:
        rotary_seq_len = max(rotary_seq_len, attention_mask.shape[-1])
    cos, sin = self.rotary_emb(value_states, seq_len=rotary_seq_len)
    query_states, key_states = apply_rotary_pos_emb(
        query_states, key_states, cos, sin, position_ids
    )

    if past_key_value is not None:
        key_states = torch.cat([past_key_value[0], key_states], dim=2)
        value_states = torch.cat([past_key_value[1], value_states], dim=2)
    new_past = (key_states, value_states) if use_cache else None

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)
    attn_weights = torch.matmul(
        query_states, key_states.transpose(2, 3)
    ) / math.sqrt(self.head_dim)

    if attention_mask is not None:
        # The mask is constructed using layer 0's cache length.  Past tokens
        # are unmasked in this batch-size-one evaluator, so retain the causal
        # tail and resize the all-zero past portion for this layer.
        if attention_mask.shape[-1] >= kv_seq_len:
            layer_mask = attention_mask[..., -kv_seq_len:]
        else:
            padding = torch.zeros(
                (*attention_mask.shape[:-1], kv_seq_len - attention_mask.shape[-1]),
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )
            layer_mask = torch.cat([padding, attention_mask], dim=-1)
        attn_weights = attn_weights + layer_mask

    attn_weights = torch.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
        query_states.dtype
    )
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous().reshape(
        bsz, q_len, self.hidden_size
    )

    if self.pretraining_tp > 1:
        chunks = attn_output.split(self.hidden_size // self.pretraining_tp, dim=2)
        weights = self.o_proj.weight.split(
            self.hidden_size // self.pretraining_tp, dim=1
        )
        attn_output = sum(F.linear(chunks[i], weights[i]) for i in range(self.pretraining_tp))
    else:
        attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None
    return attn_output, attn_weights, new_past


def enable_legacy_tgv_attention():
    """Enable the compatibility patch once, only for a TGV-KV evaluation."""
    if LlamaAttention.forward is not _tgv_compatible_llama_attention_forward:
        LlamaAttention.forward = _tgv_compatible_llama_attention_forward


def set_legacy_logical_position_limit(model, logical_length):
    """Size every legacy LLaMA RoPE table for an explicit logical position."""
    inner = model.get_model() if hasattr(model, "get_model") else model.model
    for layer in inner.layers:
        layer.self_attn._logical_position_limit = int(logical_length)


class TGVKVCache:
    """Official TGV-KV scoring/allocation adapted to expanded LLaVA images."""

    def __init__(
        self,
        layer_num,
        image_start,
        image_token_len,
        start_size=4,
        k_seq_dim=2,
        v_seq_dim=2,
        ratio=0.0,
        batch_size=1,
    ):
        self.layer_num = layer_num
        self.image_start = int(image_start)
        self.image_token_len = int(image_token_len)
        self.start_size = start_size
        self.protect_size = 1
        self.k_seq_dim = k_seq_dim
        self.v_seq_dim = v_seq_dim
        self.ratio = ratio
        self.batch_size = batch_size
        self.ratios = np.full(layer_num, ratio, dtype=np.float32)
        self.initial_text_len_list = [self.protect_size] * layer_num

    def __call__(self, past_key_values, num_of_token=None, attentions=None):
        if past_key_values is None:
            return None
        if attentions is None:
            raise ValueError("TGV-KV requires output_attentions=True.")
        if attentions[0].shape[-2] > 1:
            return self._prefill(past_key_values, num_of_token, attentions)
        return self._decode(past_key_values, num_of_token, attentions)

    def _prefill(self, past_key_values, num_of_token, attentions):
        seq_lens = np.array([p[0].size(self.k_seq_dim) for p in past_key_values])
        seq_len = int(seq_lens[0])
        total_forget = int(seq_len - num_of_token * (1 - self.ratio)) * self.layer_num
        if total_forget <= 0:
            return past_key_values

        image_start = self.image_start
        text_start = image_start + self.image_token_len
        all_attns = torch.stack([x.squeeze(0).mean(0).float() for x in attentions])
        text_image_attns = all_attns[:, text_start:, image_start:text_start]
        text_text_attns = all_attns[:, text_start:, text_start:]

        text_text_score = text_text_attns.sum(1, keepdim=True)
        divisor = torch.arange(
            1, text_text_score.shape[-1] + 1, device=text_text_score.device
        ).flip([0]).view(1, 1, -1)
        text_text_score = text_text_score / divisor
        text_image_score = (
            text_image_attns * text_text_score.transpose(-1, -2)
        ).sum(1, keepdim=True)
        pre_score = all_attns[..., :image_start].sum(1, keepdim=True) + 100
        post_score = all_attns[..., text_start:].sum(1, keepdim=True) + 100
        score_sum = torch.cat([pre_score, text_image_score, post_score], dim=2)

        layer_mass = text_image_attns.reshape(self.layer_num, -1).sum(dim=1)
        if float(layer_mass.sum().item()) <= 0:
            normalized = torch.full_like(layer_mass, 1.0 / self.layer_num)
        else:
            normalized = layer_mass / layer_mass.sum()
        layer_ratio = (
            seq_len - self.layer_num * seq_len * (1 - self.ratio) * normalized
        ) / seq_len
        self.ratios = layer_ratio.cpu().numpy()
        forget_nums = np.maximum((self.ratios * seq_lens).round().astype(np.int32), 0)

        result = []
        self.initial_text_len_list = []
        for idx, (k, v) in enumerate(past_key_values):
            forget_num = min(int(forget_nums[idx]), max(0, seq_len - self.start_size - self.protect_size))
            candidate = score_sum[idx, :, self.start_size : seq_len - self.protect_size]
            selected = torch.argsort(candidate, dim=1)[:, forget_num:] + self.start_size
            selected = selected.sort().values
            pre = torch.arange(self.start_size, device=selected.device).view(1, -1)
            post = torch.tensor([seq_len - 1], device=selected.device).view(1, -1)
            selected = torch.cat([pre, selected, post], dim=-1)
            self.initial_text_len_list.append(
                max(int((selected[0] >= text_start).sum().item()), self.protect_size)
            )
            gather = selected.to(k.device).view(self.batch_size, 1, -1, 1)
            result.append((
                k.gather(-2, gather.expand(-1, k.shape[1], -1, k.shape[-1])),
                v.gather(-2, gather.expand(-1, v.shape[1], -1, v.shape[-1])),
            ))
        return tuple(result)

    def _decode(self, past_key_values, num_of_token, attentions):
        seq_lens = np.array([p[0].size(self.k_seq_dim) for p in past_key_values])
        forget_nums = (seq_lens - num_of_token * (1 - self.ratios)).astype(np.int32)
        forget_nums = np.maximum(forget_nums, 0)
        result = []
        for idx, (k, v) in enumerate(past_key_values):
            if forget_nums[idx] <= 0:
                result.append((k, v))
                continue
            seq_len = int(seq_lens[idx])
            evict_start = self.start_size
            evict_end = seq_len - self.initial_text_len_list[idx]
            if evict_start >= evict_end:
                result.append((k, v))
                continue
            score = attentions[idx].mean(1).squeeze(0).sum(0)
            # Keep the eviction index on the accelerator. Calling .item() here
            # creates one device-to-host synchronization per layer per token.
            pruned = score[evict_start:evict_end].argmin() + evict_start
            indices = torch.arange(seq_len, device=k.device)
            retained = indices[indices != pruned.to(k.device)]
            result.append((
                k.index_select(self.k_seq_dim, retained),
                v.index_select(self.v_seq_dim, retained),
            ))
        return tuple(result)
