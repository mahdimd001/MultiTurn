"""Drive the unchanged evaluators from Python on CPU (shared by the SeedKV tests)."""
import contextlib
import io
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "turncast"))

from run_cpu import patched_module  # noqa: E402  (turncast/run_cpu.py: CPU shims only)

_MODULES = {}


def _cpu_turncast():
    """kv_cache_turncast.load_scorer defaults to CUDA; on a CPU-only machine load on CPU."""
    if torch.cuda.is_available():
        return
    import kv_cache_turncast
    original = kv_cache_turncast.load_scorer
    if getattr(original, "_cpu_shim", False):
        return

    def load_scorer(path, device="cpu"):
        return original(path, "cpu")

    load_scorer._cpu_shim = True
    kv_cache_turncast.load_scorer = load_scorer


def evaluator(script):
    """The evaluator as a module with parse_cli() and main(args)."""
    if script not in _MODULES:
        _MODULES[script] = patched_module(script)
        _cpu_turncast()
    return _MODULES[script]


def run(script, argv, quiet=True):
    """Run one evaluation; returns the parsed arguments it ran with."""
    module = evaluator(script)
    old_argv = sys.argv
    sys.argv = [script] + [str(x) for x in argv]
    try:
        args = module.parse_cli()
        sink = io.StringIO()
        with contextlib.redirect_stdout(sink if quiet else sys.stdout), \
                contextlib.redirect_stderr(sink if quiet else sys.stderr):
            module.main(args)
    finally:
        sys.argv = old_argv
    return args


def standin_paths(standin, split):
    return {
        "model": os.path.join(standin, "llava-v1-standin"),
        "data": os.path.join(standin, "data", f"eval_{split}.json"),
        "images": os.path.join(standin, "data", "images"),
    }


def base_argv(standin, split, method, ratio, samples, out_dir, name):
    paths = standin_paths(standin, split)
    return [
        "--model-path", paths["model"], "--data-path", paths["data"], "--image-path", paths["images"],
        "--device", "cpu", "--method", method, "--ratio", ratio, "--eval-samples", samples,
        "--output-dir", out_dir, "--exp-name", name,
    ]


def run_ppl(standin, split, method, ratio, samples, out_dir, name, extra=()):
    run("eval_ppl_final.py", base_argv(standin, split, method, ratio, samples, out_dir, name) + list(extra))
    with open(os.path.join(out_dir, name + ".json")) as handle:
        return json.load(handle)


def run_rouge(standin, split, method, ratio, samples, out_dir, name, extra=(), max_tokens=24):
    argv = base_argv(standin, split, method, ratio, samples, out_dir, name)
    run("eval_rouge_final2.py", argv + ["--max-token-generated", max_tokens] + list(extra))
    with open(os.path.join(out_dir, name + "_predictions.json")) as handle:
        return json.load(handle)


def exact_match(result):
    """Share of generated answers equal to the reference, overall and per turn."""
    per_turn, total = {}, []
    for item in result["results"]:
        for turn in item["turns"]:
            hit = float(" ".join(turn["prediction"].split()) == " ".join(turn["reference"].split()))
            per_turn.setdefault(str(turn["turn"]), []).append(hit)
            total.append(hit)
    mean = lambda values: sum(values) / max(1, len(values))  # noqa: E731
    return mean(total), {turn: mean(values) for turn, values in sorted(per_turn.items())}


class Recorder:
    """Record the physical cache size after every cache call of one evaluator run.

    events: dicts with kind ("call", "finalize" or "prepare"), the conversation's
    cache index, the logical length, the entries per layer and the cache state;
    caches: every cache object the run built.
    """

    def __init__(self, script, inspect=None, on_build=None):
        self.module = evaluator(script)
        self.events = []
        self.caches = []
        self.forwards = 0
        self.inspect = inspect      # fn(kind, cache, past_before, past_after)
        self.on_build = on_build    # fn(cache), right after the evaluator builds one

    @staticmethod
    def _sizes(past):
        return [int(k.shape[2]) for k, _ in past]

    def __enter__(self):
        module, recorder = self.module, self
        self._saved = (module.apply_cache, module.finalize_turn_cache, module.build_cache)
        apply_cache, finalize, build = self._saved

        def note(kind, kv_cache, num_of_token, before, after):
            recorder.events.append({
                "kind": kind, "cache": len(recorder.caches) - 1,
                "logical": None if num_of_token is None else int(num_of_token),
                "sizes": recorder._sizes(after), "state": getattr(kv_cache, "state", None),
            })
            if recorder.inspect is not None:
                recorder.inspect(kind, kv_cache, before, after)

        def apply_recorded(kv_cache, past, num_of_token, attentions):
            after = apply_cache(kv_cache, past, num_of_token, attentions)
            note("call", kv_cache, num_of_token, past, after)
            return after

        def finalize_recorded(kv_cache, past):
            after = finalize(kv_cache, past)
            note("finalize", kv_cache, None, past, after)
            return after

        def build_recorded(*args, **kwargs):
            cache = build(*args, **kwargs)
            recorder.caches.append(cache)
            if recorder.on_build is not None:
                recorder.on_build(cache)
            if hasattr(cache, "prepare_turn"):
                prepare = cache.prepare_turn

                def prepare_recorded(past, num_of_token=None):
                    after = prepare(past, num_of_token)
                    note("prepare", cache, num_of_token, past, after)
                    return after

                cache.prepare_turn = prepare_recorded
            return cache

        module.apply_cache, module.finalize_turn_cache, module.build_cache = (
            apply_recorded, finalize_recorded, build_recorded)

        from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM
        self._model_class = LlavaLlamaForCausalLM
        self._forward = LlavaLlamaForCausalLM.forward

        def counting_forward(model, *args, **kwargs):
            recorder.forwards += 1
            return recorder._forward(model, *args, **kwargs)

        LlavaLlamaForCausalLM.forward = counting_forward
        return self

    def __exit__(self, *exc):
        module = self.module
        module.apply_cache, module.finalize_turn_cache, module.build_cache = self._saved
        self._model_class.forward = self._forward
        return False


def set_threads():
    torch.set_num_threads(int(os.environ.get("THREADS", 2)))
