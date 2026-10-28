#!/usr/bin/env python3
"""Patch MultiTurn's two evaluators to enable --method atlaskv.

Copy this file and kv_cache_atlaskv.py to the *root* of the repository.
Run `python integrate_atlaskv.py` for a dry-run, then `--apply` to modify files.
Never modifies Python files unless all integration anchors are found exactly once.
"""
from pathlib import Path
import argparse

TARGETS = ("eval_rouge_final2.py", "eval_ppl_final.py")
CHANGES = [
    (
        "from kv_cache_coverage import TurnConditionedCoverageKVCache",
        "from kv_cache_coverage import TurnConditionedCoverageKVCache\nfrom kv_cache_atlaskv import AtlasKVCache",
    ),
    (
        '    if args.method == "pyramidkv":',
        '''    if args.method == "atlaskv":
        return AtlasKVCache(
            layer_num=layer_num,
            image_start=image_start,
            image_token_len=image_token_len,
            **common,
        )
    if args.method == "pyramidkv":''',
    ),
    (
        '    return not isinstance(kv_cache, LocalCache)',
        '    return not isinstance(kv_cache, (LocalCache, AtlasKVCache))',
    ),
    (
        '    return not isinstance(kv_cache, (LocalCache, ElasticCache))',
        '    return not isinstance(kv_cache, (LocalCache, ElasticCache, AtlasKVCache))',
    ),
    (
        '            "pyramidkv",',
        '            "pyramidkv",\n            "atlaskv",',
    ),
]

def patch_text(source):
    if 'from kv_cache_atlaskv import AtlasKVCache' in source:
        raise ValueError('AtlasKV is already integrated; no changes applied')
    for old, new in CHANGES:
        count = source.count(old)
        if count != 1:
            raise ValueError(f'Expected one occurrence of {old!r}; found {count}')
        source = source.replace(old, new, 1)
    return source


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--apply', action='store_true')
    opts = p.parse_args()
    edited = {}
    for name in TARGETS:
        path = Path(__file__).resolve().parent / name
        if not path.exists():
            raise FileNotFoundError(f'{path} not found: copy to MultiTurn root first')
        edited[path] = patch_text(path.read_text())
    if opts.apply:
        for path, new_text in edited.items():
            path.write_text(new_text)
            print(f'Patched {path.name}')
    else:
        print('Both evaluators are compatible; dry-run passed. Run --apply to patch them.')

if __name__ == '__main__':
    main()
