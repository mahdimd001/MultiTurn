# Image-first layout and image-based long context (November 2026)

## What changed

| | Before | Now |
|---|---|---|
| Turn-1 prompt | `USER: <question>\n<image>` | `USER: <image>\n<question>` (LLaVA's own layout) |
| Long context (`--context-lengths N`) | text passage before the question | extra images after the conversation's own image |
| Images per item | exactly one | one or more, as one contiguous block |

* `mm_layout.py` is the single place that builds the Turn-1 message. Every
  evaluator has `--image-position first|last`; `last` reproduces the old
  numbers exactly (checked: SeedKV 1.399, same as before).
* `make_long_context.py --mode images` (default) adds images from other
  conversations and writes them to `extra_images`. One image is a fixed number
  of tokens (576 for LLaVA-1.5), so N cannot be hit exactly: the number of
  images whose prefill is closest to N is used, and the table's `prefill`
  column shows the real length (on LLaVA-1.5 roughly: 2048 -> 3 images, about
  1,790 tokens; 3584 -> 6 images, about 3,520). `--mode text` is the old behaviour. Files are
  named `<data>_imgN.json`, so old `_ctxN.json` files are not reused by mistake.
* `eval_ppl_final.py`, `eval_rouge_final2.py` and `eval_efficiency.py` load
  `extra_images` and pass `image_token_len = images x 576` to every cache.
* `turncast/collect.py` uses the same layout (not run here).

Not changed: the older scripts (`eval_ppl.py`, `eval_rouge.py`,
`eval_rogue_final.py`, `eval_latency.py`, ...) still build the old layout.

**Everything measured with the old layout has to be re-run**, and two files
have to be rebuilt because they were made in the old layout:

* the GhostKV locator file (`--ghostkv-calibrate`);
* the TurnCast-KV scorer (`turncast/collect.py`, `turncast/train.py`).

LLaVA-1.5 has 4,096 positions. Seven images already fill them (7 x 576 =
4,032 plus the text), so 6 images (`--context-lengths 3584`) is the practical
ceiling for a conversation that still has room to answer.

## Baselines: review against the new layout

I read the six baselines for anything that depends on where the image is.
None needs a code change.

| Method | Verdict | Notes |
|---|---|---|
| `tgv_kv` | correct, and the new layout is the one it was written for | It scores image tokens by the attention from the text AFTER the image. In the old layout that text was only `ASSISTANT:`, so the "text guidance" did not see the question. Now it does. As in the official code there is no turn-boundary step: while decoding it evicts one entry per token, chosen from everything except the first `start_size` and the last few entries, so later turns can lose recent dialogue. |
| `pyramidkv` | correct | Its observation window is the last 8 prefill tokens. Old layout: the tail of the image plus `ASSISTANT:`. New layout: the end of the question, as in the paper. |
| `meda` | correct | Cross-modal entropy now uses real question-to-image attention. Its rule ranks every text entry above every image entry, so with one image at ratio 0.9 almost no image entry survives selection (only merged). That is the paper's rule, not a bug. |
| `elastic_cache` | matches the official code | Including the official code's `torch.argsort(...) > forget_num` line, which compares sort indices instead of ranks, so the kept set is not a true importance ranking. I left it as is so the baseline stays the official one. |
| `h2o` | correct | Layout-independent. |
| `local_cache` | correct | Keeps the start and the most recent entries; with the image first and a long context the image is dropped entirely. |

The older comparison was tilted: with the question before the image, the
question-aware methods (TGV-KV, PyramidKV, MEDA) could not see the question
when they scored the image.

## SeedKV and GhostKV

* SeedKV: the seed now holds the features of all images (`[images, 576, 1024]`
  flattened); the text ids before and after the image block are found for any
  number of images. All 29 checks pass with the image first.
* GhostKV: same seed fix; the vectorised detection and fault handling
  (`ghostkv_speed_update.patch`) are included. Two defaults changed:
  * `--ghostkv-fault-on answer` (was `all`): page in only while answering.
  * `--ghostkv-victims image` (new): a paged-in patch overwrites image entries
    only, so paging never displaces text. `any` is the previous behaviour.
  All 27 checks pass with the image first.
* With the image first, the text in front of the image is the system prompt
  (about 35 tokens on LLaVA), so one fault recomputes about 35 tokens plus the
  missing patches instead of the whole passage.

## Running time (CPU stand-in, not LLaVA)

`seedkv/run_efficiency_cpu.py` runs the unchanged `eval_efficiency.py` on the
stand-in. Ratio 0.9, 4 turns, 12 generated tokens per turn, 8 conversations,
contexts of 1, 4, 8 and 13 images (144 tokens each):

| Prefill | Method | TTFT2+ ms | TPOT ms | Conversation s | KV peak MiB |
|---|---|---|---|---|---|
| 188 | PyramidKV | 36.3 | 3.90 | 0.34 | 0.4 |
| 188 | SeedKV | 53.8 | 3.72 | 0.38 | 1.7 |
| 188 | GhostKV | 33.8 | 6.21 | 0.45 | 0.3 |
| 620 | PyramidKV | 37.4 | 4.18 | 0.41 | 0.7 |
| 620 | SeedKV | 101.9 | 4.14 | 0.60 | 5.1 |
| 620 | GhostKV | 38.2 | 6.69 | 0.54 | 0.6 |
| 1196 | PyramidKV | 39.9 | 4.35 | 0.52 | 1.2 |
| 1196 | SeedKV | 195.0 | 4.85 | 1.00 | 9.6 |
| 1196 | GhostKV | 35.5 | 7.04 | 0.64 | 0.9 |
| 1916 | PyramidKV | 41.5 | 4.73 | 0.93 | 1.7 |
| 1916 | SeedKV | 375.9 | 4.69 | 1.74 | 15.3 |
| 1916 | GhostKV | 40.8 | 7.89 | 0.93 | 1.4 |

* Time to first token of later turns: GhostKV equals PyramidKV at every
  context size. SeedKV's grows with the context (it re-runs the whole prefill)
  and is 9x GhostKV's at 13 images.
* Per token GhostKV is slower than both (about +2.5 ms on this tiny model):
  1.5 to 2 faults per turn at about 12 ms each, spread over only 12 generated
  tokens, plus 0.35 ms of detection per token.
* Whole conversation: GhostKV is faster than SeedKV from 4 images up and
  slower with a single image. On LLaVA I expect the same shape: about equal at
  the native context, clearly faster with several images. That is an
  expectation, not a measurement.

Raw table: `seedkv/image_first_standin_efficiency.csv`.

## Quality on a stand-in retrained with the image first

100 conversations x 4 turns, perplexity of the reference answers, all turns
(shift / same):

| Method | ratio 0.5 | 0.8 | 0.9 |
|---|---|---|---|
| Full cache | 1.00 / 1.00 | | |
| PyramidKV | 1.18 / 1.12 | 1.69 / 1.69 | 2.41 / 2.75 |
| TGV-KV | 2.25 / 3.37 | 3.42 / 3.00 | 9.54 / 8.50 |
| SeedKV | 1.14 / 1.16 | 1.52 / 1.57 | 2.28 / 2.52 |
| GhostKV | 1.03 / 1.01 | 1.59 / 1.54 | 2.87 / 4.42 |
| GhostKV, paging off | 1.19 / 1.14 | 1.73 / 1.75 | 2.54 / 3.27 |

Exact answers on shift: ratio 0.5: GhostKV 96%, SeedKV 86%, PyramidKV 85%;
ratio 0.9: SeedKV 50%, GhostKV 35%, PyramidKV 29%.

**This is weaker than the earlier (old-layout) GhostKV result, and you should
know why before relying on either:**

* At ratio 0.5 and 0.8 GhostKV is the best or tied. At 0.9 it is worse than
  PyramidKV in perplexity. On this stand-in the budget at 0.9 is 19 entries
  per layer and GhostKV's seed plus ghost keys take 19% of it (about 5% on
  LLaVA). With no faults at all that overhead alone costs 2.6 -> 3.0.
* On this stand-in, compression at 0.8 to 0.9 mostly breaks the *wording* of
  the answers for every method, not the image lookup. Paging fixes lookups
  (wrong-attribute answers: 39 with PyramidKV, 5 with GhostKV on 120 later
  turns) but cannot fix wording.
* Paging while the question is read, and letting patches overwrite text, both
  made this stand-in much worse (later-turn perplexity 14 instead of 3). That
  is why the two defaults changed. On the old stand-in they were harmless.
* Two small synthetic models now disagree about how much GhostKV helps at
  high ratios. Only LLaVA runs can settle it.

Raw numbers: `seedkv/image_first_standin_results.json`.

## Run it

    # long-context files with extra images (also made automatically by eval_efficiency.py)
    python make_long_context.py --model-path $MODEL --data-path ConvBench/convbench.json \
        --context-lengths 2048 3584 --mode images

    # GhostKV locators for the new layout (once)
    python eval_ppl_final.py --model-path $MODEL --data-path $DATA --image-path $IMAGES \
        --eval-samples 20 --method ghostkv --ratio 0.9 \
        --ghostkv-calibrate locators_llava15_7b_imagefirst.pt --output-dir logs --exp-name calib

    python eval_efficiency.py --methods full pyramidkv seedkv ghostkv --ratios 0.9 \
        --context-lengths native 2048 3584 --long-context-mode images \
        --ghostkv-locators locators_llava15_7b_imagefirst.pt --gen-tokens 64 --num-turns 3

    # PPL / ROUGE on a long-context file: pass it as --data-path
    python eval_ppl_final.py ... --data-path ConvBench/long_context/convbench_img2048.json

    # the old layout, to reproduce earlier numbers
    python eval_ppl_final.py ... --image-position last

Stand-in for the new layout: `python seedkv/make_standin.py --out DIR`
(`--image-position last` builds the old one).
