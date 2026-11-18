"""Run eval_efficiency.py on CPU with a small test model (see make_standin.py).

Test-only shims, as in turncast/run_cpu.py: the model is cast to float32 and
the caches are sized from the model's real layer count.  The evaluator itself
is unchanged.  CPU timings of a tiny model say how much WORK a method does
(tokens through the model); they are not GPU latencies.

    python seedkv/run_efficiency_cpu.py --model-path DIR/llava-v1-standin \
        --data-path DIR/data/eval_shift.json --image-path DIR/data/images \
        --device cpu --methods pyramidkv seedkv ghostkv --ratios 0.9 \
        --context-lengths native 1024 --ghostkv-locators DIR/ghost_locators.pt
"""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "turncast"))


def main(argv=None):
    torch.set_num_threads(int(os.environ.get("THREADS", 2)))
    import run_cpu
    rouge = run_cpu.patched_module("eval_rouge_final2.py")
    sys.modules["eval_rouge_final2"] = rouge

    from llava.model import builder
    original = builder.load_pretrained_model

    def load(*args, **kwargs):
        tokenizer, model, processor, context = original(*args, **kwargs)
        model = model.float()
        model.get_vision_tower().to(dtype=torch.float32)
        model.eval()
        rouge._TEST_LAYER_NUM = int(model.config.num_hidden_layers)
        return tokenizer, model, processor, context

    builder.load_pretrained_model = load
    import eval_efficiency
    return eval_efficiency.main(argv)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
