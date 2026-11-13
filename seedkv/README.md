# SeedKV: evict the image to a seed, regrow it when the next question arrives

SeedKV is a multi-turn KV-cache policy for LLaVA. Between turns it keeps **no
image KV at all**, only a small seed (the vision features and the prompt's
token ids). When the next question arrives it regenerates the image KV
exactly, reads the question with the whole image visible, and then selects
image and text entries together down to the usual budget.

Every other method in this repo only retains. It decides which visual entries
survive before the next question exists, and an evicted entry never comes
back.

## Why

Your own logs say that better scoring has little left to give:

* `turncast_data/turncast_scorer_report.json`, ConvBench, ratio 0.9: PyramidKV
  captures 91% of the attention later turns pay, and an oracle that knows the
  true future attention captures 94%.
* The 2026 audit "Seen, Said, or Forgotten?" (arXiv 2607.25467, on ConvBench
  and VisDial) reports that current attention is a poor predictor of which
  visual entries a later turn needs, and that the damage from deleting them is
  irreversible and grows over turns.

So the lever SeedKV pulls is different: do not predict the next question,
wait for it. That is affordable because the prefill prompt
(`<system> USER: Q1 <image> ASSISTANT:`) sits at the start of the context.
Its KV depends on nothing that comes later, so one batched forward over its
input embeddings rebuilds it exactly, at any turn.

## How

| Step | When | What |
|---|---|---|
| 1 | Turn-1 prefill | Keep the seed. Compress the prefill to the budget (the question is in view). |
| 2 | Turn boundary (`finalize_turn`) | Drop every prefill entry. Compress the dialogue text to the budget. |
| 3 | Next turn (`prepare_turn`) | Regenerate the full prefill KV from the seed: one forward of the prefill length. |
| 4 | Commit (first cache call after `begin_answer`) | The question has been read with the whole image visible. Select image and text entries together from its attention. |

* **Scoring** is PyramidKV's rule: attention of the last 8 queries, max-pool 7,
  per-head top-k. Nothing is trained. What is new is where and when the rule
  runs.
* **Layer budgets** follow PyramidKV's linear pyramid with an exact total.
* **Heads** keep their own entries. All heads of a layer keep the same number
  of image entries and the same number of text entries, so the tensors stay
  rectangular after the image is dropped.
* **The attention sink** (position 0) is kept whenever prefill entries are
  resident.

## Budget

`C(n) = int(n * (1 - ratio))` entries per layer on average, like the other
methods, **minus the seed's own size**, so KV + seed never exceeds `L * C`.

| Moment | Entries held |
|---|---|
| After the Turn-1 prefill | `L * C(prefill length)` |
| At rest between turns | at most `L * C(logical length)`, text only |
| After a turn's commit, during generation | `L * (C(length at last boundary) + tokens appended since)` |
| While a later turn's question is read | the full prefill KV + the text (the size every method holds during the Turn-1 prefill, before its first compression) |

Seed size for LLaVA-1.5-7B: 576 x 1024 CLIP features in 16-bit floats =
1.2 MB, against about 300 MB for the image KV over 32 layers. That is charged
as about 72 KV entries, 2.25 per layer (the budget is 64 per layer after a
640-token prefill at ratio 0.9).

**PyramidKVCache holds more than `L * C`.** `kv_cache_pyramidkv.py` rounds the
pyramid's slope down, which leaves extra entries in the deep layers. For 32
layers and a 640-token prefill it keeps 2288 entries at ratio 0.9 against a
nominal 2048 (+11.7%), and +23% at ratio 0.95. SeedKV keeps exactly `L * C`
by default. For a like-for-like run against that baseline use
`--seedkv-allocation pyramidkv`, which gives every layer the same number of
entries PyramidKVCache gives it (minus the seed).

## Cost

One extra forward over the prefill per later turn. In `results.txt` the
Turn-1 prefill of LLaVA-1.5-7B takes about 103 ms and TTFT of later turns is
about 900 ms, so expect roughly +0.1 s per later turn. `eval_efficiency.py`
measures it: the regeneration is inside TTFT of turns 2+, and the
question-reading peak shows up in `kv_peak_bytes` / `kv_transient_peak_bytes`.

## Options

| Flag | Default | Meaning |
|---|---|---|
| `--seedkv-window-size` | 8 | Observation window, as in PyramidKV. |
| `--seedkv-allocation` | `pyramid` | Per-layer budgets. `pyramid`: PyramidKV's schedule with an exact total. `pyramidkv`: the same entries per layer as this repo's PyramidKVCache. `uniform`: equal. |
| `--seedkv-seed` | `auto` | What is stored: `auto` picks the smallest that rebuilds the prefill exactly. |
| `--seedkv-refresh-every` | 0 | Also regenerate and reselect every N answer tokens (the full image is resident for 8 tokens each time). |
| `--seedkv-no-seed-charge` | off | Ablation: do not subtract the seed from the budget. |
| `--seedkv-no-regenerate` | off | Ablation: same code, scoring and budgets, but retain-only. |

`--ratio` is the only setting you need.

## Files

| File | Purpose |
|---|---|
| `kv_cache_seedkv.py` | The method. |
| `seedkv/test_seedkv.py` | Correctness checks through the unchanged evaluators. |
| `seedkv/make_standin.py` | Builds a small LLaVA-architecture model for CPU tests. |
| `seedkv/run_compare.py` | Runs methods x ratios x splits on that model. |
| `seedkv/bench_overhead.py` | Cache-code overhead at LLaVA-1.5-7B shapes. |
| `seedkv/harness.py` | Shared helpers for the three scripts above. |
| `seedkv/standin_results.json` | Every number in the tables below. |
| `run-job-seedkv` | Nova job: PPL, ROUGE and efficiency against PyramidKV and TurnCast-KV. |

## What was tested, and where

LLaVA-1.5-7B could not be run where this was developed (no GPU, no access to
the weights). Everything below ran on CPU on a **stand-in**: an 8-layer
LLaVA-architecture model built by `make_standin.py`, loaded by
`llava.model.builder` and driven by the unchanged `eval_ppl_final.py` and
`eval_rouge_final2.py`. It answers 99% of questions correctly with a full
cache, so it does read its image. **None of these numbers are LLaVA numbers.**

### Correctness (`python seedkv/test_seedkv.py --standin DIR`): all checks pass

* The regenerated prefill KV equals the original bit for bit (max difference 0).
* KV + seed is within `L * (C + appended)` at every cache call outside question
  reading (534 calls checked per ratio). While a question is read the cache is
  above budget for about 10 tokens per later turn.
* Per-head positions are sorted and unique, head counts are equal, the sink is
  kept (4464 layer states checked per ratio).
* Ratio 0 gives the same perplexity as the full cache.
* The evaluator runs the same number of forwards as with PyramidKV, plus
  exactly one regeneration per later turn.
* Both fallbacks work (no `prepare_turn`, no `begin_answer`); two runs give
  identical perplexity.

### Quality on the stand-in, ratio 0.9, 100 conversations x 4 turns

Three test sets: **shift** (every later turn asks about an object no earlier
turn touched), **same** (every later turn asks about the Turn-1 object again),
**mixed** (the training mix; a quarter of the first turns are "describe the
image"). Perplexity of the reference answers, lower is better.

| Method | shift: all | turn 1 | later | same: all | turn 1 | later | mixed: all | turn 1 | later |
|---|---|---|---|---|---|---|---|---|---|
| Full cache | 1.00 | 1.00 | 1.00 | 1.01 | 1.01 | 1.01 | 1.04 | 1.11 | 1.00 |
| PyramidKV | 1.78 | 1.62 | 1.84 | 1.50 | 1.65 | 1.46 | 1.98 | 2.49 | 1.73 |
| TurnCast-KV (scorer trained on the stand-in) | 1.22 | 1.09 | 1.27 | 1.04 | 1.03 | 1.04 | 1.27 | 1.32 | 1.24 |
| Turn-conditioned coverage | 1.38 | 1.59 | 1.32 | 1.30 | 1.56 | 1.24 | 1.54 | 1.80 | 1.41 |
| Layerwise eviction (full cache in turn 1) | 1.57 | 1.00 | 1.83 | 1.48 | 1.01 | 1.65 | 1.36 | 1.11 | 1.53 |
| Layerwise merge (full cache in turn 1) | 1.48 | 1.00 | 1.69 | 1.48 | 1.01 | 1.65 | 1.44 | 1.11 | 1.69 |
| AtlasKV | 1.61 | 2.84 | 1.32 | 1.47 | 2.39 | 1.28 | 1.83 | 3.21 | 1.30 |
| TGV-KV | 2.21 | 1.82 | 2.37 | 2.62 | 2.17 | 2.77 | 5.31 | 16.67 | 2.67 |
| MEDA | 3.07 | 14.87 | 1.80 | 2.78 | 15.89 | 1.70 | 3.76 | 12.28 | 1.85 |
| LocalCache | 2.80 | 5.05 | 2.29 | 2.37 | 4.25 | 2.00 | 5.29 | 21.84 | 2.26 |
| ElasticCache | 2.79 | 4.39 | 2.39 | 6.02 | 4.49 | 6.54 | 6.67 | 24.17 | 3.08 |
| H2O | 18.10 | 9.18 | 22.79 | 15.37 | 4.60 | 21.70 | 19.61 | 13.26 | 24.79 |
| **SeedKV** | 1.36 | 1.82 | 1.23 | 1.39 | 1.98 | 1.25 | 2.11 | 4.69 | 1.31 |
| SeedKV, seed not charged | 1.30 | 1.70 | 1.19 | 1.33 | 1.79 | 1.22 | 1.88 | 3.73 | 1.25 |
| SeedKV with regeneration off (ablation) | 1.84 | 1.70 | 1.89 | 1.57 | 1.79 | 1.51 | 2.35 | 3.73 | 1.77 |

Share of generated answers that equal the reference (%), ratio 0.9:

| Method | shift: all | turn 1 | later | same: all | turn 1 | later | mixed: all | turn 1 | later |
|---|---|---|---|---|---|---|---|---|---|
| Full cache | 100 | 100 | 99 | 98 | 99 | 97 | 94 | 76 | 99 |
| PyramidKV | 38 | 45 | 35 | 54 | 54 | 53 | 38 | 35 | 40 |
| TurnCast-KV | 76 | 84 | 74 | 94 | 97 | 92 | 69 | 59 | 72 |
| Turn-conditioned coverage | 71 | 66 | 73 | 69 | 67 | 70 | 63 | 42 | 70 |
| **SeedKV** | 58 | 30 | 67 | 57 | 33 | 65 | 51 | 22 | 61 |
| SeedKV with regeneration off (ablation) | 35 | 36 | 34 | 47 | 43 | 49 | 36 | 28 | 39 |

Later-turn perplexity by ratio (shift / same / mixed):

| Method | 0.8 | 0.9 | 0.95 |
|---|---|---|---|
| PyramidKV | 1.48 / 1.21 / 1.38 | 1.84 / 1.46 / 1.73 | 2.32 / 1.98 / 2.26 |
| TurnCast-KV | 1.07 / 1.02 / 1.05 | 1.27 / 1.04 / 1.24 | 2.24 / 1.88 / 2.18 |
| Turn-conditioned coverage | 1.08 / 1.10 / 1.07 | 1.32 / 1.24 / 1.41 | 2.09 / 2.12 / 2.11 |
| **SeedKV** | 1.05 / 1.14 / 1.10 | 1.23 / 1.25 / 1.31 | 1.56 / 1.44 / 1.59 |
| SeedKV with regeneration off | 1.52 / 1.25 / 1.43 | 1.89 / 1.51 / 1.77 | 2.26 / 1.96 / 2.25 |

How to read this:

* **Regeneration is what helps.** With identical code, scoring and budget,
  switching it on takes later-turn perplexity from 1.89 to 1.23 (shift), 1.51
  to 1.25 (same) and 1.77 to 1.31 (mixed) at ratio 0.9, and later-turn exact
  match from 34% to 67% on shift.
* **SeedKV beats PyramidKV on later turns at every ratio and on every set.**
  At ratio 0.95 its later turns are also ahead of TurnCast-KV and coverage.
* **SeedKV does not beat TurnCast-KV or coverage overall on the stand-in at
  ratio 0.9.** Its first turn is worse than PyramidKV's (exact match 30% vs
  45% on shift). Turn 1 is PyramidKV's rule with a smaller budget: the seed
  takes 19 of this model's 144 entries (13%), where on LLaVA-1.5-7B it takes
  about 72 of 2048 (3.5%), and PyramidKVCache holds 17% more than nominal here.
* **The stand-in overrates scorers.** It reads its image when it writes the
  answer, not while it reads the question, so question-time attention is a
  weak signal and a learned scorer can memorise "objects, not background".
  The same happened before in this repo: on the earlier CPU model TurnCast-KV
  reached perplexity 4.86 against PyramidKV's 12.87 (`turncast/README.md`),
  and on LLaVA the two tied (ROUGE-L 0.2698 vs 0.2661, ConvBench, ratio 0.9). Treat the ablation row as the
  evidence for the mechanism and the cross-method ranking as provisional.
* Regeneration and scoring are independent. Using TurnCast-KV's forecaster or
  the coverage selector at SeedKV's commit is the obvious next step.

`--seedkv-refresh-every 8` brings the first-turn perplexity on **mixed** from
4.69 to 2.32 (its long "describe" answers), but in this model that keeps the
full image resident for half of the answer, so it is not a budget-matched
result.

### Cost

* Cache code at LLaVA-1.5-7B shapes (`bench_overhead.py`, 2 CPU cores, float32):
  prefill compression 37 ms (PyramidKV 38 ms), decode step 0.12 ms (0.38 ms),
  and about 0.11 s more per later turn, almost all of it copying the
  regenerated KV. On a GPU that copy is milliseconds.
* The regeneration forward is not in those numbers. It is one batched forward
  over the prefill, which `results.txt` puts at about 103 ms for LLaVA-1.5-7B.
* On the CPU stand-in a 100-conversation perplexity run took 25.7 s with
  SeedKV and 19.7 s with PyramidKV (+30%). A CPU is far slower at one batched
  186-token forward relative to single-token steps than a GPU is, so this
  overstates the GPU overhead.

## Run it on LLaVA

    sbatch run-job-seedkv

runs PyramidKV, TurnCast-KV and SeedKV on ConvBench and MultiVerse at ratios
0.9, 0.8 and 0.5 (PPL, ROUGE-L, efficiency). Things to look at first:

1. **Per-turn perplexity of turns 2+**, SeedKV against PyramidKV. That is
   where the mechanism acts.
2. `--seedkv-no-regenerate` at the same ratio. It isolates regeneration from
   everything else.
3. Ratio 0.95. Your logs show turn-1 ROUGE-L with a full cache (0.2501 on
   ConvBench) inside the range of the compressed methods at 0.9
   (0.2467-0.2558), so ROUGE-L against the references has under 0.01 of
   headroom there.

## Reproduce the stand-in numbers (CPU, about an hour)

    pip install torch==2.1.2 transformers==4.31.0 tokenizers==0.13.3 sentencepiece accelerate==0.21.0 rouge
    python seedkv/make_standin.py --out /tmp/standin          # ~25 min on 2 cores
    python seedkv/test_seedkv.py --standin /tmp/standin
    python seedkv/run_compare.py --standin /tmp/standin --metric ppl \
        --methods full pyramidkv turn_conditioned_coverage seedkv seedkv_retain_only \
        --ratios 0.9 --out /tmp/ppl.json
    python seedkv/bench_overhead.py

`make_standin.py` trains with three aids (fewer-object scenes, a per-patch
word loss, an attention guide on one head in each of two layers). Without
them a model this small did not learn to read the image in 6000 steps.

## Related work

* **SparseVILA** (arXiv 2510.17777) keeps the visual cache and retrieves a
  question-specific subset of it each turn. SeedKV keeps no visual KV at rest.
* **Conduit** (arXiv 2609.05821) recomputes stale cached visual tokens so a
  cache can be reused under a changed prefix.
* **HCache**, **CachedAttention** restore or offload a whole conversation's
  state between turns.
* **"Seen, Said, or Forgotten?"** (arXiv 2607.25467) is the audit that
  motivates not deleting visual KV irreversibly; it does not test re-encoding.

I found no paper that evicts the visual KV to a seed and regenerates it per
turn inside a fixed eviction budget, but this field moves fast and that is a
search result, not a guarantee.
