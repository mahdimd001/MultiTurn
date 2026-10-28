"""Run eval_rouge_final2.py / eval_ppl_final.py on CPU with a small test model.

Test-only shims (the evaluator logic itself is unchanged):
  1. cast the loaded model to float32 (CPU has no fast fp16 kernels);
  2. read the layer count from the model config instead of the "7b" -> 32 rule,
     so small test models build correctly sized caches.
Usage: python turncast/run_cpu.py eval_ppl_final.py <normal evaluator arguments>
       (or import patched_module() to drive an evaluator from Python)
"""
import os
import sys
import types

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)


def patched_module(script, run_main=False):
    os.chdir(REPO)
    source = open(os.path.join(REPO, script)).read()
    old = 'layer_num = 32 if "7b" in model_name.lower() else 40'
    assert source.count(old) == 1
    source = source.replace(old, "layer_num = _TEST_LAYER_NUM")
    loader = "= load_pretrained_model("
    assert source.count(loader) == 1
    source = source.replace(loader, "= _test_load(")
    name = os.path.splitext(script)[0] + "_cpu"
    module = types.ModuleType(name)
    module.__file__ = os.path.join(REPO, script)
    sys.modules[name] = module

    def _test_load(*args, **kwargs):
        from llava.model.builder import load_pretrained_model

        tokenizer, model, processor, ctx = load_pretrained_model(*args, **kwargs)
        model = model.float()
        model.get_vision_tower().to(dtype=torch.float32)
        model.eval()
        module._TEST_LAYER_NUM = int(model.config.num_hidden_layers)
        return tokenizer, model, processor, ctx

    module._test_load = _test_load
    module._TEST_LAYER_NUM = None
    main_block = 'if __name__ == "__main__":'
    if run_main:
        source = source.replace(main_block, "if True:")
    else:
        # keep the argument parser, skip the run: parse_cli() returns the args
        head, block = source.split(main_block)
        block = block.replace("    main(args)", "    return args")
        source = head + "def parse_cli():" + block
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    return module


if __name__ == "__main__":
    torch.set_num_threads(int(os.environ.get("THREADS", 2)))
    script = sys.argv[1]
    sys.argv = [script] + sys.argv[2:]
    patched_module(script, run_main=True)
