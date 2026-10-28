#!/usr/bin/env python
"""Generate long-context versions of a multi-turn image-dialogue dataset.

For every target length N, each conversation's turn-1 user message gets a
background passage in front of the question, so that the turn-1 prefill is
exactly N tokens.  The turn-1 prefill is everything LLaVA processes in the first
forward pass: system prompt, expanded image tokens and text, using the model's
own tokenizer and conversation template.  Turns 2+ are unchanged; the passage
stays in the conversation history and KV cache for them.

The output keeps the input format ({"image", "turns": [{"question", "answer"}]}),
so eval_rogue_final.py and eval_efficiency.py both read it directly.  ROUGE and
efficiency can therefore be measured on identical samples.

Turn-1 layout (eval_rogue_final.py appends the image after the question):

    <system> USER: Background notes:\n<passage>\n\n<question>\n<image> ASSISTANT:

The passage is built from the reference answers of OTHER conversations in the
dataset, never the conversation's own answers (those would leak into ROUGE),
shuffled with a fixed per-conversation seed.  The order does not depend on N,
so a longer context extends the shorter one's passage.  Pass --filler-file to
use your own text instead.

Each output item carries a "long_context" block with the target, the measured
prefill length, the native length and the original question.

Usage (from the MultiTurn root):
    python make_long_context.py --model-path ./models/llava-v1.5-7b \
        --data-path ConvBench/convbench.json --context-lengths 1024 2048 3072
    # -> ConvBench/long_context/convbench_ctx1024.json, ..._ctx2048.json, ...
eval_efficiency.py calls the same code automatically when a file is missing.
"""

import argparse
import json
import os
import random
import unicodedata

DEFAULT_HEADER = "Background notes:\n"
IMAGE_PLACEHOLDER = "<image>"

FALLBACK_TEXT = (
    "Field notes from a long walk along the river. The path starts beside an old "
    "brick warehouse that now houses a bakery, a bicycle repair shop and a small "
    "printing studio. Past the warehouse the trail follows the water under three "
    "bridges, each built in a different decade and each painted a different color. "
    "Rowers practice in the mornings, and in the afternoons families gather on the "
    "grass near the boathouse to feed the ducks and watch the barges move slowly "
    "upstream. A wooden sign lists the birds seen here last spring: herons, "
    "kingfishers, swallows, two kinds of geese and a single osprey that nested on "
    "a light pole near the rail yard. Further on, the city gives way to fields of "
    "corn and soybeans, a grain elevator, and a narrow gravel road that leads to a "
    "farm stand selling sweet corn, tomatoes, honey and late-season apples. The "
    "walk ends at a small park with a shelter, a water fountain and a bench where "
    "someone has carved the names of a dozen friends and the year they graduated."
)


# --------------------------------------------------------------------------
# Dataset helpers
# --------------------------------------------------------------------------


def infer_conv_mode(model_name):
    """Same rule as eval_rogue_final.infer_conv_mode."""
    name = model_name.lower()
    if "llama-2" in name:
        return "llava_llama_2"
    if "v1" in name:
        return "llava_v1"
    if "mpt" in name:
        return "mpt"
    return "llava_v0"


def turns_of(item):
    if item.get("turns"):
        return [dict(turn) for turn in item["turns"]]
    if "question" in item:
        return [{"question": item["question"], "answer": item.get("answer", "")}]
    return []


def clean_text(text):
    """ASCII prose without special tokens, so token trimming never splits bytes."""
    text = unicodedata.normalize("NFKD", text or "")
    text = text.encode("ascii", "ignore").decode("ascii")
    for marker in (IMAGE_PLACEHOLDER, "</s>", "<s>", "<AND>"):
        text = text.replace(marker, " ")
    return " ".join(text.split())


def passage_pool(raw_items, filler_file=None):
    """List of (owner_item_index, text). Owner -1 means usable by every item."""
    pool = []
    if filler_file:
        with open(filler_file, "r") as handle:
            paragraphs = handle.read().split("\n\n")
        pool = [(-1, clean_text(p)) for p in paragraphs if clean_text(p)]
    else:
        for index, item in enumerate(raw_items):
            for turn in turns_of(item):
                text = clean_text(turn.get("answer", ""))
                if text:
                    pool.append((index, text))
    if not pool:
        pool = [(-1, clean_text(FALLBACK_TEXT))]
    return pool


def long_context_path(out_dir, data_path, target):
    stem = os.path.splitext(os.path.basename(data_path))[0] if data_path else "synthetic"
    return os.path.join(out_dir, f"{stem}_ctx{int(target)}.json")


# --------------------------------------------------------------------------
# Exact-length construction
# --------------------------------------------------------------------------


class PrefillCounter:
    """Counts turn-1 prefill tokens exactly as LLaVA will see them."""

    def __init__(self, tokenizer, template, image_tokens, conv_mode, header=DEFAULT_HEADER):
        from llava.constants import IMAGE_TOKEN_INDEX
        from llava.mm_utils import tokenizer_image_token

        self.tokenizer = tokenizer
        self.template = template
        self.image_tokens = int(image_tokens)
        self.conv_mode = conv_mode
        self.header = header
        self._image_index = IMAGE_TOKEN_INDEX
        self._tokenize = tokenizer_image_token
        # Prompts longer than model_max_length (2048 in LLaVA's tokenizer config)
        # are expected here; mark the one-off "indexing errors" warning as shown.
        warned = getattr(tokenizer, "deprecation_warnings", None)
        if isinstance(warned, dict):
            warned["sequence-length-is-longer-than-the-specified-maximum"] = True

    def question(self, original, passage, padding=""):
        if not passage:
            return original
        return self.header + passage + padding + "\n\n" + original

    def count(self, question):
        if IMAGE_PLACEHOLDER not in question:  # same rule as eval_rogue_final.py
            question = question + "\n" + IMAGE_PLACEHOLDER
        conv = self.template.copy()
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        ids = self._tokenize(conv.get_prompt(), self.tokenizer, self._image_index)
        placeholders = sum(1 for token in ids if token == self._image_index)
        return len(ids) - placeholders + placeholders * self.image_tokens


def item_pool_ids(pool, item_index, need_tokens, tokenizer, seed):
    """Token ids of this item's shuffled passage source, at least need_tokens long."""
    entries = [text for owner, text in pool if owner != item_index] or [t for _, t in pool]
    order = list(entries)
    random.Random(seed * 1000003 + item_index).shuffle(order)
    target_chars = need_tokens * 4 + 512
    while True:
        chunks, chars, position = [], 0, 0
        while chars < target_chars:
            text = order[position % len(order)]
            chunks.append(text)
            chars += len(text) + 2
            position += 1
        ids = tokenizer.encode("\n\n".join(chunks), add_special_tokens=False)
        if len(ids) >= need_tokens:
            return ids
        target_chars *= 2


PADDING_UNITS = ("\n", ".", ",", " .", " -")


def fit_passage(counter, original, pool_ids, target, max_repeats=4):
    """Return (question_text, prefill_tokens) with prefill_tokens == target if possible.

    Cut the passage source after k tokens and re-measure (a cut can re-tokenize
    unevenly at word boundaries).  If no cut lands exactly, take the longest
    one below the target and top it up at the end of the passage, newlines
    first (one token each in the Llama tokenizer).  Every candidate is
    measured again rather than assumed.
    """
    passage, length = _fit_cut(counter, original, pool_ids, target)
    if length < target and passage:
        for unit in PADDING_UNITS:
            for repeats in range(1, max_repeats + 1):
                candidate = counter.question(original, passage, padding=unit * repeats)
                measured = counter.count(candidate)
                if measured == target:
                    return candidate, measured
                if measured > target:
                    break
    return counter.question(original, passage), length


def _fit_cut(counter, original, pool_ids, target):
    tokenizer = counter.tokenizer
    seen = {}

    def measure(k):
        if k not in seen:
            text = tokenizer.decode(
                pool_ids[:k], skip_special_tokens=True, clean_up_tokenization_spaces=False
            ).strip()
            seen[k] = (text, counter.count(counter.question(original, text)))
        return seen[k]

    base = counter.count(original)
    if base == target:
        return "", base
    k = max(1, target - base)
    for _ in range(12):  # Newton-style: token count is ~linear in k
        k = max(1, min(k, len(pool_ids)))
        text, length = measure(k)
        if length == target:
            return text, length
        step = target - length
        if k + step == k or (k + step) in seen:
            break
        k += step
    for distance in range(1, 9):  # boundary merges: search nearby cut points
        for candidate in (k - distance, k + distance):
            if 1 <= candidate <= len(pool_ids):
                text, length = measure(candidate)
                if length == target:
                    return text, length
    under = [value for value in seen.values() if value[1] <= target]
    return max(under, key=lambda v: v[1]) if under else min(seen.values(), key=lambda v: v[1])


def build_long_context_dataset(raw_items, counter, target, seed=0, filler_file=None, log=print):
    """Return (items, report) where every item's turn-1 prefill is `target` tokens."""
    pool = passage_pool(raw_items, filler_file)
    source = os.path.basename(filler_file) if filler_file else "reference answers of other conversations"
    output, too_long, inexact = [], [], []
    for index, item in enumerate(raw_items):
        turns = turns_of(item)
        if not turns:
            continue
        original = turns[0].get("question", "")
        native = counter.count(original)
        if native > target:
            too_long.append(index)
            continue
        pool_ids = item_pool_ids(pool, index, target - native + 64, counter.tokenizer, seed)
        question, measured = fit_passage(counter, original, pool_ids, target)
        if measured != target:
            inexact.append((index, measured))
        turns[0]["question"] = question
        new_item = {k: v for k, v in item.items() if k not in ("turns", "question", "answer")}
        new_item["turns"] = turns
        new_item["long_context"] = {
            "target_prefill_tokens": int(target),
            "prefill_tokens": int(measured),
            "native_prefill_tokens": int(native),
            "added_tokens": int(measured - native),
            "image_tokens": counter.image_tokens,
            "conv_mode": counter.conv_mode,
            "passage_source": source,
            "seed": seed,
            "original_question": original,
        }
        output.append(new_item)
    report = {
        "target": int(target),
        "written": len(output),
        "skipped_native_longer_than_target": len(too_long),
        "not_exact": len(inexact),
    }
    if too_long:
        log(
            f"  ctx={target}: skipped {len(too_long)} conversation(s) whose native "
            f"turn-1 prompt is already longer than {target} tokens"
        )
    if inexact:
        log(f"  ctx={target}: {len(inexact)} item(s) could not hit the exact length: {inexact[:5]}")
    return output, report


def write_dataset(path, items):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as handle:
        json.dump(items, handle, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model-path", default="./models/llava-v1.5-7b",
                        help="Only the tokenizer is loaded (no weights).")
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--context-lengths", nargs="+", type=int, required=True,
                        help="Turn-1 prefill lengths in tokens, e.g. 1024 2048 3072.")
    parser.add_argument("--out-dir", default=None,
                        help="Default: <data-path directory>/long_context")
    parser.add_argument("--image-tokens", type=int, default=576,
                        help="Expanded image tokens (576 for LLaVA-1.5 CLIP-L/336).")
    parser.add_argument("--conv-mode", default=None, help="Default: inferred like eval_rogue_final.py")
    parser.add_argument("--filler-file", default=None)
    parser.add_argument("--header", default=DEFAULT_HEADER)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    from transformers import AutoTokenizer
    from llava.conversation import conv_templates
    from llava.mm_utils import get_model_name_from_path

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False)  # as llava's builder
    conv_mode = args.conv_mode or infer_conv_mode(get_model_name_from_path(args.model_path))
    counter = PrefillCounter(tokenizer, conv_templates[conv_mode], args.image_tokens, conv_mode, args.header)

    with open(args.data_path, "r") as handle:
        raw_items = json.load(handle)
    if isinstance(raw_items, dict):
        raw_items = list(raw_items.values())
    out_dir = args.out_dir or os.path.join(os.path.dirname(args.data_path) or ".", "long_context")
    print(f"conv_mode={conv_mode} image_tokens={args.image_tokens} items={len(raw_items)}")
    for target in args.context_lengths:
        items, report = build_long_context_dataset(
            raw_items, counter, target, seed=args.seed, filler_file=args.filler_file
        )
        path = long_context_path(out_dir, args.data_path, target)
        write_dataset(path, items)
        print(f"ctx={target}: wrote {report['written']} conversations -> {path}")


if __name__ == "__main__":
    main()
