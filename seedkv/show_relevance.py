"""
Draw GhostKV's relevance maps over the image, one panel per turn.

    # 1. run any evaluator with GhostKV and --ghostkv-save-maps
    python eval_ppl_final.py ... --method ghostkv --ratio 0.9 \
        --ghostkv-locators locators.pt --ghostkv-save-maps maps.pt --eval-samples 20
    # 2. draw conversation 3 of that run
    python seedkv/show_relevance.py --maps maps.pt --data-path convbench.json \
        --image-path images/ --conversation 3 --out conv3.png

The map of a turn is the attention the locator heads WOULD pay to every image
patch if the whole image were cached (kv_cache_ghostkv.py), summed over the
answer tokens.  Circles mark the patches that were regenerated in that turn.

--which   relevance_answer (default) | peak
--aspect  pad (LLaVA-1.5: the image is padded to a square) | resize
Needs matplotlib and Pillow.
"""
import argparse
import json
import os

import torch


def square(image, aspect):
    from PIL import Image
    if aspect == "resize" or image.size[0] == image.size[1]:
        return image
    side = max(image.size)
    canvas = Image.new("RGB", (side, side), (124, 116, 104))        # CLIP mean, as LLaVA pads
    canvas.paste(image, ((side - image.size[0]) // 2, (side - image.size[1]) // 2))
    return canvas


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--maps", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--image-path", required=True)
    parser.add_argument("--conversation", type=int, default=0, help="index in the run (= in the data file)")
    parser.add_argument("--which", default="relevance_answer", choices=["relevance_answer", "peak"])
    parser.add_argument("--aspect", default="pad", choices=["pad", "resize"])
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from PIL import Image

    store = torch.load(args.maps, map_location="cpu")
    turns = [t for t in store["turns"] if t["conversation"] == args.conversation]
    if not turns:
        raise SystemExit(f"no conversation {args.conversation} in {args.maps} "
                         f"({store['conversations']} conversations)")
    with open(args.data_path) as handle:
        item = json.load(handle)[args.conversation]
    image = square(Image.open(os.path.join(args.image_path, item["image"])).convert("RGB"), args.aspect)

    surface, ink, soft = "#fcfcfb", "#0b0b0b", "#52514e"
    ramp = LinearSegmentedColormap.from_list("blue", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
    figure, axes = plt.subplots(1, len(turns), figsize=(3.6 * len(turns), 4.6), facecolor=surface, squeeze=False)
    for axis, turn, text in zip(axes[0], turns, item["turns"]):
        values = turn[args.which]
        axis.imshow(image)
        axis.set_xticks([])
        axis.set_yticks([])
        title = f"Turn {turn['turn']}: {text['question'][:60]}"
        if values is None:
            axis.set_title(title + "\n(no map: no single-token step)", fontsize=8.5, loc="left", color=ink)
            continue
        values = values[: values.numel() // (1 + len(item.get("extra_images") or []))]   # the item's own image
        side = int(round(values.numel() ** 0.5))
        grid = (values / values.sum().clamp_min(1e-9)).view(side, side).numpy()
        extent = (0, image.size[0], image.size[1], 0)
        alpha = (grid / max(grid.max(), 1e-9)) ** 0.5 * 0.85            # transparent where nothing is read
        axis.imshow(grid, cmap=ramp, alpha=alpha, extent=extent, interpolation="nearest", vmin=0)
        cell = image.size[0] / side
        paged = turn["regenerated"]
        axis.scatter([(p % side + 0.5) * cell for p in paged], [(p // side + 0.5) * cell for p in paged],
                     s=10, facecolors="none", edgecolors=ink, linewidths=0.7)
        top = values.topk(max(1, values.numel() // 32)).values.sum() / values.sum().clamp_min(1e-9)
        axis.set_title(title, fontsize=8.5, loc="left", color=ink)
        axis.set_xlabel(f"{turn['faults']} fault(s), {len(paged)} patches regenerated\n"
                        f"top 3% of the patches hold {100 * float(top):.0f}% of the map", fontsize=8, color=soft)
    figure.tight_layout()
    figure.savefig(args.out, dpi=200, facecolor=surface)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
