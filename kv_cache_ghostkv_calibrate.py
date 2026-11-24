"""
Find GhostKV's locator heads (once per model, ~20 conversations, no training).

Run any evaluator with ``--method ghostkv --ghostkv-calibrate FILE``.  The
model runs with the FULL cache and its output is unchanged.  On answer tokens:

  * every head: how much attention it puts on its top image patches
    ("top mass") and how many patches hold 90% of its image attention;
  * the 32 heads with the most concentrated image attention get a causal test:
    hide the patches the head looks at most and measure how much the
    next-token distribution moves (KL).  That is the head's score.

FILE holds ``score`` [layers, heads], ``top_mass_quantiles`` (50/75/90/95%)
and ``footprint`` (patches a head reads in its sharpest 10% of tokens); see
``kv_cache_ghostkv.load_locators``.
"""
import math
import os

import torch

from kv_cache_seedkv import SeedKVCache

QUANTILES = (0.5, 0.75, 0.9, 0.95)
CANDIDATES = 32        # heads that get the causal test
WARMUP = 4             # conversations used to pick them
DENSE, STRIDE = 8, 8   # test the first 8 answer tokens of a turn, then every 8th
HIDE = 0.03            # share of the image hidden per test
COVER = 0.9

_STATS = {}            # path -> running statistics (shared by the caches of one run)


def concentration(attention):
    """[heads, patches] -> [heads]: attention mass on the image x how peaked it is."""
    mass = attention.sum(dim=-1)
    prob = attention / mass.clamp_min(1e-9).unsqueeze(-1)
    entropy = -(prob * torch.log(prob.clamp_min(1e-12))).sum(dim=-1)
    return mass * (1.0 - entropy / math.log(max(2, attention.shape[-1]))).clamp_min(0.0)


class GhostKVCalibrator(SeedKVCache):
    """Has the cache interface, keeps the full cache, and writes the locator file."""

    def __init__(self, layer_num, image_start, image_token_len, path):
        self.path = path
        super().__init__(layer_num, 0.0, image_start, image_token_len, regenerate=False)
        self.image_end = self.image_start + self.image_len

    def reset(self):
        super().reset()
        if getattr(self, "_hook", None) is not None:
            self._hook.remove()
        self._hook, self._kwargs, self._step = None, None, 0

    def attach_model(self, model):
        super().attach_model(model)
        self._model = model
        stats = _STATS.setdefault(self.path, {"prior": None, "steps": 0, "conversations": 0, "candidates": None,
                                              "kl": None, "kl_n": None, "top_mass": [], "spread": []})
        stats["conversations"] += 1
        self._hook = self._inner.register_forward_pre_hook(self._capture, with_kwargs=True)

    def _capture(self, module, args, kwargs):
        if not self._busy and kwargs.get("past_key_values") is not None:
            self._kwargs = dict(kwargs)                  # to replay this step with patches hidden
        return None

    @torch.no_grad()
    def _logp(self, mask):
        kwargs = dict(self._kwargs)
        kwargs.update(attention_mask=mask, use_cache=False, output_attentions=False,
                      output_hidden_states=False, return_dict=True)
        self._busy = True
        try:
            hidden = self._inner(**kwargs).last_hidden_state[:, -1]
        finally:
            self._busy = False
        return torch.log_softmax(self._model.lm_head(hidden).float(), dim=-1)[0]

    def observe_layer(self, layer_idx, attention):
        return None

    def __call__(self, past_key_values, num_of_token=None, attentions=None):
        if attentions is None or attentions[0] is None or not self._answer_begun:
            return past_key_values
        if attentions[0].shape[-2] != 1 or attentions[0].shape[-1] < self.image_end:
            return past_key_values
        stats = _STATS[self.path]
        image = [a[0, :, 0, self.image_start: self.image_end].float() for a in attentions]
        prior = torch.stack([concentration(a).cpu() for a in image])
        stats["prior"] = prior if stats["prior"] is None else stats["prior"] + prior
        stats["steps"] += 1
        hide = max(1, int(round(HIDE * self.image_len)))
        if len(stats["top_mass"]) < 4096:
            stats["top_mass"].append(torch.stack([a.topk(hide, dim=-1).values.sum(-1).cpu() for a in image]))
            spread = []
            for a in image:
                ordered = a.sort(dim=-1, descending=True).values.cumsum(-1)
                spread.append((ordered < COVER * ordered[:, -1:]).sum(-1).add(1).cpu())
            stats["spread"].append(torch.stack(spread))
        if stats["conversations"] <= WARMUP or self._kwargs is None:
            return past_key_values
        if stats["candidates"] is None:
            order = stats["prior"].flatten().argsort(descending=True)[:CANDIDATES].tolist()
            heads = stats["prior"].shape[1]
            stats["candidates"] = [(i // heads, i % heads) for i in order]
            stats["kl"], stats["kl_n"] = torch.zeros_like(stats["prior"]), torch.zeros_like(stats["prior"])
        self._step += 1
        if self._step > DENSE and self._step % STRIDE:
            return past_key_values
        full = torch.ones((1, attentions[0].shape[-1]), dtype=torch.long, device=attentions[0].device)
        base = self._logp(full)
        for layer, head in stats["candidates"]:
            mask = full.clone()
            mask[0, image[layer][head].topk(hide).indices + self.image_start] = 0
            stats["kl"][layer, head] += float((base.exp() * (base - self._logp(mask))).sum())
            stats["kl_n"][layer, head] += 1
        return past_key_values

    def prepare_turn(self, past_key_values, num_of_token=None):
        return past_key_values

    def finalize_turn(self, past_key_values):
        """Turn boundary: write the file (rewritten after every turn)."""
        self._step, self._answer_begun = 0, False
        stats = _STATS.get(self.path)
        if not stats or not stats["steps"]:
            return past_key_values
        prior = stats["prior"] / stats["steps"]
        tested = stats["kl"] is not None and float(stats["kl_n"].sum()) > 0
        score = stats["kl"] / stats["kl_n"].clamp_min(1) if tested else prior
        top_mass = torch.stack(stats["top_mass"]).float()                       # [steps, layers, heads]
        quantiles = torch.quantile(top_mass, torch.tensor(QUANTILES), dim=0)
        spread = torch.stack(stats["spread"]).float()
        sharp = top_mass >= quantiles[2][None]
        footprint = torch.where(sharp, spread, torch.full_like(spread, float("nan"))).nanmedian(dim=0).values
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        torch.save({"score": score, "prior": prior, "top_mass_quantiles": quantiles,
                    "footprint": torch.nan_to_num(footprint, nan=0.0).round().long(),
                    "steps": stats["steps"], "conversations": stats["conversations"]}, self.path)
        return past_key_values
