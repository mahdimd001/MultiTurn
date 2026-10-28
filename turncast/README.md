# TurnCast-KV: learned forecast of what a conversation will read

TurnCast-KV is a multi-turn KV-cache eviction method for LLaVA. A small trained
model (about 15k parameters) predicts which image patches and earlier text
tokens the rest of the conversation will attend to. The cache keeps those
entries. Nothing extra is generated or run at inference: the forecaster reads
statistics that the normal forward passes already produce.

## Why

The evaluators compress the cache at two kinds of point, always before the
next question is known:

* right after the Turn-1 prefill;
* at every turn boundary.

The Turn-1 compression decides which of the 576 visual tokens survive for the
whole conversation. PyramidKV makes that decision from the attention of the
last 8 prompt tokens. The prompt is `USER: Q1\n<image> ASSISTANT:`, so those
8 tokens are 3 image patches and ` ASSISTANT:`. Because Q1 comes before the
image, none of those 8 tokens knows what Q1 asked or what later turns will ask.

## How

1. **Collect** (`collect.py`). Run one teacher-forced pass of the frozen LLaVA
   per conversation (no generation). At every compression point, record two
   things for each cached entry:
   * the free inputs: attention from this turn's question rows, answer rows
     and last 8 rows; the peak attention in earlier turns; the same signals
     averaged over heads and over layers, and max-pooled over image/text
     neighbours; key and value norms; segment, turn age, recency and image
     position.
   * the target: the attention that the REST of the conversation actually
     paid to that entry.

   Use 200 conversations per dataset. The other conversations become
   `heldout.json`, which the end-to-end runs use.
2. **Train** (`train.py`). Fit a 2-layer MLP (64 hidden units) with a
   per-(layer, head) embedding, using listwise KL against the target. The
   training settings are fixed: 8 epochs, AdamW, lr 2e-3, keeping the best
   validation epoch. Training takes minutes.
3. **Evict** (`kv_cache_turncast.py`, `--method turncast_kv`).
   * At the Turn-1 compression, every head keeps its top-B_l entries by
     forecast score. The budget B_l is PyramidKV's linear pyramid, with an
     exact total of L × C. The attention sink is always kept.
   * At turn boundaries, the default (`--turncast-boundary window`) applies
     PyramidKV's window rule with the same budgets. `--turncast-boundary
     forecast` uses the forecaster there too; see the results below.

Hyperparameters: `--ratio` only. The forecaster's training settings are fixed,
and the window size (8) and pooling (7) are PyramidKV's standard values, used
only at boundaries.

## Cost

| | PyramidKV | TurnCast-KV |
|---|---|---|
| Model passes | prefill + 1 per token | same |
| Attention maps read | prefill + decoding | same |
| Extra work | none | features + MLP at each compression point |

By default the forecaster runs once per conversation. For LLaVA-7B that is
about 0.67M entries and about 9 GFLOPs, which is less than one decoding step.
`bench_overhead.py` measures the cache code alone at LLaVA-7B shapes.

## Run on Nova

```bash
sbatch --export=ALL,STAGE=prepare run-job-turncast           # collect + train + offline check (~30 min)
sbatch --export=ALL,STAGE=eval,RATIOS=0.9 run-job-turncast   # one job per ratio, in parallel
```

The job runs the following on the held-out conversations of both datasets:

* PyramidKV, PyramidKV with an exact budget, TurnCast-KV, and TurnCast-KV with
  `--turncast-boundary forecast`;
* ROUGE, with paired per-turn bootstrap (`compare_rouge_runs.py`), and PPL;
* `eval_efficiency.py` for the overhead.

## Tests (`test_turncast.py`, CPU)

* **Train/inference parity.** The features computed online while
  `eval_ppl_final.py` feeds a conversation token by token match the
  collector's single-pass features (max difference 6e-4).
* **Budgets.** Every compression keeps at most L × C entries, and exactly
  L × C when every layer is long enough. The sink is always kept, and
  positions stay sorted and unique.
* **No extra passes.** The number of model forwards is identical to PyramidKV's.

## Results so far (CPU sandbox; LLaVA weights were not reachable)

These come from an 8-layer LLaVA-architecture model trained on synthetic
multi-turn "needle" conversations. Each needle is one of 144 visual tokens,
and there are 3 turns. The forecaster was trained on 400 conversations and
tested on 100 held-out ones. The metric is the perplexity of the reference
answers; full cache is 1.59.

| ratio | PyramidKV | PyramidKV (exact budget) | **TurnCast-KV** | forecaster at boundaries too | oracle (true future attention) |
|---|---|---|---|---|---|
| 0.9 | 12.87 | 12.29 | **4.86** | 7.26 | 5.72 |
| 0.8 | 1.99 | 1.96 | **1.82** | 1.93 | 1.90 |
| 0.5 | 1.589 | 1.589 | 1.590 | 1.590 | 1.592 |

Per turn at ratio 0.9 (PyramidKV → TurnCast-KV): turn 1 34.2 → 2.35,
turn 2 8.40 → 5.79, turn 3 7.41 → 8.44.  Reproduce with `diagnose.py`.

Lessons that shaped the design:

* **The forecaster learns its target.** On held-out conversations it captures
  97–99% of the future attention that an oracle with the true future
  attention captures. PyramidKV with an exact budget captures 66–94% (66% at
  ratio 0.9).
* **The win is at the Turn-1 compression.** The turn-1 answer perplexity at
  ratio 0.9 is 2.4 for TurnCast-KV and 33 for PyramidKV. The forecaster keeps
  Q1's words and the patches that later turns read; PyramidKV's window cannot
  see them.
* **At turn boundaries, attention mass is the wrong target in this test.**
  Even the TRUE future attention lost to PyramidKV's window rule on turn 3.
  The window rule keeps the just-finished answer, which later answers copy
  from. Hence the default `boundary=window`.
* **Two variants were rejected:**
  * Allocating layer budgets by forecast mass kept more attention mass but
    starved some layers, and was worse end-to-end at every ratio.
  * Splitting the boundary budget between the two rules was worse than either
    rule alone.

The toy model never learned to read its image (chance accuracy with full
cache), so it cannot show the benefit of keeping image regions for later
turns at boundaries. Only LLaVA can settle `boundary=forecast` versus
`boundary=window`.

## Related work

Learned eviction scorers already exist:

* ForesightKV (arXiv 2602.03203) uses an MLP over key, value and attention
  features, trained on future attention and with RL, for long reasoning
  traces.
* LookaheadKV (arXiv 2603.10899) uses PEFT modules that predict importance
  during prefill.
* Learning to Evict (arXiv 2602.10238) and TrimKV.

TurnCast-KV differs in three ways:

* **Setting.** It targets multi-turn VLM conversations, and the target is what
  the later TURNS read.
* **Inputs.** It uses no raw keys or values. It uses free dialogue-structure
  signals: turn-wise question and answer attention, cross-turn history, the
  2-D image neighbourhood, and cross-head and cross-layer consensus. The
  forecaster is therefore ~15k parameters and is trained from 400
  conversations with one forward pass each.
* **Where it is used.** It is applied at the compression that fixes the visual
  tokens, and it keeps PyramidKV's rule where the toy showed attention mass
  misleads.
