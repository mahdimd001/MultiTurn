"""
Build a small LLaVA-architecture model that really reads its image (CPU, minutes).

LLaVA-1.5 weights are not always reachable from a test machine.  This script
makes a stand-in that goes through the SAME code as LLaVA in this repo
(llava.model.builder, LlavaLlamaForCausalLM, the llava_v1 conversation
template, the unchanged evaluators) and is trained from scratch on synthetic
multi-turn visual question answering:

  * image: 12 x 12 patches; six objects, each a 2 x 2 block of patches that
    encodes a shape, a colour and a mark colour, on a noisy grey background;
  * training aids, without which a model this small does not learn to read the
    image in the steps a CPU allows: scenes with one to four objects, a
    per-patch word loss (patch_words) and an attention guide on one head in
    each of two layers;
  * turns: "what color is the star ?", "what mark does the ring have ?",
    "which shape is red ?", "describe the image .", "and the moon ?" (ellipsis:
    needs the previous turn), "what did you say about the star ?" (needs an
    earlier answer).

Three evaluation sets separate what a cache policy has to do:

  eval_shift.json   every later turn asks about an object no earlier turn touched
  eval_same.json    every later turn asks about the Turn-1 object again
  eval_mixed.json   the training mix (also contains "describe the image .")

Usage:
    python seedkv/make_standin.py --out /path/standin --steps 4000
"""
import argparse
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch
from PIL import Image

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import mm_layout  # noqa: E402

IMAGE_POSITION = "first"
GRID, CELL = 12, 8
SIZE = GRID * CELL
N_OBJECTS = 6
SHAPES = ["circle", "square", "star", "cross", "ring", "wave", "arrow", "moon"]
COLORS = {
    "red": (220, 40, 40), "green": (40, 180, 60), "blue": (50, 80, 230), "yellow": (240, 220, 40),
    "purple": (150, 50, 200), "orange": (250, 140, 30), "cyan": (40, 210, 220), "white": (245, 245, 245),
}
MARKS = {"gold": (255, 200, 0), "silver": (170, 170, 190), "brown": (140, 80, 30), "pink": (255, 105, 180)}
SYSTEM_MEAN = 0.5


def object_patch(shape, color, mark):
    """One 8 x 8 patch of an object.

    Top half: eight 2 x 2 cells, the cell of this shape is lit in the mark colour.
    Bottom half: eight 2 x 2 cells, the cell of this colour is lit in that colour.
    The attributes are therefore linearly readable from the pixels, which lets a
    small model learn the lookup quickly.
    """
    patch = np.full((CELL, CELL, 3), 20, np.int16)
    s = SHAPES.index(shape)
    patch[2 * (s // 4): 2 * (s // 4) + 2, 2 * (s % 4): 2 * (s % 4) + 2] = MARKS[mark]
    c = list(COLORS).index(color)
    patch[4 + 2 * (c // 4): 6 + 2 * (c // 4), 2 * (c % 4): 2 * (c % 4) + 2] = COLORS[color]
    return patch


# ----------------------------------------------------------------------
# Scenes and conversations
# ----------------------------------------------------------------------
def sample_scene(rng, n_objects=N_OBJECTS):
    shapes = rng.sample(SHAPES, n_objects)
    colors = rng.sample(list(COLORS), n_objects)
    slots = rng.sample(range((GRID // 2) ** 2), n_objects)
    objects = []
    for shape, color, slot in zip(shapes, colors, sorted(slots)):
        objects.append({
            "shape": shape, "color": color, "mark": rng.choice(list(MARKS)),
            "row": 2 * (slot // (GRID // 2)), "col": 2 * (slot % (GRID // 2)),
        })
    return {"objects": objects, "noise_seed": rng.randrange(2 ** 31)}


def render(scene):
    """uint8 [SIZE, SIZE, 3]."""
    noise = np.random.RandomState(scene["noise_seed"])
    image = np.full((SIZE, SIZE, 3), 128, np.int16)
    for obj in scene["objects"]:
        patch = object_patch(obj["shape"], obj["color"], obj["mark"])
        for dr in range(2):
            for dc in range(2):
                r0, c0 = (obj["row"] + dr) * CELL, (obj["col"] + dc) * CELL
                image[r0:r0 + CELL, c0:c0 + CELL] = patch
    image = image + noise.randint(-8, 9, image.shape)
    return np.clip(image, 0, 255).astype(np.uint8)


def ask(kind, obj):
    if kind == "color":
        return f"what color is the {obj['shape']} ?", f"the {obj['shape']} is {obj['color']} ."
    if kind == "mark":
        return f"what mark does the {obj['shape']} have ?", f"the {obj['shape']} has a {obj['mark']} mark ."
    if kind == "which":
        return f"which shape is {obj['color']} ?", f"the {obj['color']} one is the {obj['shape']} ."
    raise ValueError(kind)


def describe(scene):
    parts = [f"a {o['color']} {o['shape']}" for o in scene["objects"]]
    return "describe the image .", "there is " + " , ".join(parts) + " ."


def make_conversation(rng, mode="mixed", n_turns=4, n_objects=N_OBJECTS):
    scene = sample_scene(rng, n_objects)
    objects = scene["objects"]
    turns, touched, said, last_attr = [], [], {}, None

    def add(question, answer, focus, obj=None, attr=None):
        turn = {"question": question, "answer": answer, "focus": focus}
        if obj is not None:
            # the word the image has to supply, and where the object is
            turn["key"] = obj["shape"] if attr == "which" else obj[attr]
            turn["ident"] = obj["color"] if attr == "which" else obj["shape"]   # names the object
            turn["target"] = [obj["row"], obj["col"]]
        turns.append(turn)
        if obj is not None and obj not in touched:
            touched.append(obj)
        if obj is not None and attr in ("color", "mark"):
            said[obj["shape"]] = (attr, answer)

    def fresh():
        pool = [o for o in objects if o not in touched]
        return rng.choice(pool) if pool else None

    # Turn 1
    if mode == "mixed" and rng.random() < 0.25:
        add(*describe(scene), focus="describe")
    else:
        kinds = ["color", "mark"] if mode == "same" else ["color", "mark", "which"]
        kind = rng.choice(kinds)
        obj = rng.choice(objects)
        add(*ask(kind, obj), focus="first", obj=obj, attr=kind)
        last_attr = kind if kind in ("color", "mark") else None

    while len(turns) < n_turns:
        if mode == "shift":
            choice = rng.choice(["new", "new", "followup"])
        elif mode == "same":
            choice = rng.choice(["same", "same", "recall"])
        else:
            choice = rng.choices(["new", "followup", "same", "recall"], [0.45, 0.2, 0.2, 0.15])[0]

        if choice == "followup" and last_attr is not None and fresh() is not None:
            obj = fresh()
            _, answer = ask(last_attr, obj)
            add(f"and the {obj['shape']} ?", answer, focus="new", obj=obj, attr=last_attr)
        elif choice == "recall" and said:
            shape = rng.choice(list(said))
            _, answer = said[shape]
            add(f"what did you say about the {shape} ?", "i said " + answer, focus="recall")
            last_attr = None
        elif choice == "same" and touched:
            obj = rng.choice(touched) if mode != "same" else touched[0]
            kind = rng.choice(["color", "mark", "which"])
            add(*ask(kind, obj), focus="same", obj=obj, attr=kind)
            last_attr = kind if kind in ("color", "mark") else None
        else:
            obj = fresh() or rng.choice(objects)
            kind = rng.choice(["color", "mark", "which"])
            add(*ask(kind, obj), focus="new" if obj not in touched else "same", obj=obj, attr=kind)
            last_attr = kind if kind in ("color", "mark") else None
    return scene, turns


# ----------------------------------------------------------------------
# Rendering a conversation exactly as the evaluators do
# ----------------------------------------------------------------------
def word_id(word, tokenizer):
    if word not in _WORD_IDS:
        _WORD_IDS[word] = tokenizer(word, add_special_tokens=False).input_ids[0]
    return _WORD_IDS[word]


def encode(turns, tokenizer, lookups=None):
    """input ids (one image placeholder) and labels (answers + </s> only).

    lookups, if a list, receives (index of the key word, index of the word that
    names the object, patch indices of the object) for every turn that reads an
    object from the image.
    """
    from llava.constants import DEFAULT_IMAGE_TOKEN, IGNORE_INDEX, IMAGE_TOKEN_INDEX
    from llava.conversation import conv_templates
    from llava.mm_utils import tokenizer_image_token

    conv = conv_templates["llava_v1"].copy()
    ids, labels = [], []
    for index, turn in enumerate(turns):
        question = turn["question"]
        if index == 0:                          # the same Turn-1 layout as the evaluators
            question = mm_layout.first_turn_question(question, 1, IMAGE_POSITION)
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        base = tokenizer_image_token(conv.get_prompt(), tokenizer, IMAGE_TOKEN_INDEX)
        conv.messages[-1][-1] = turn["answer"]
        full = tokenizer_image_token(conv.get_prompt(), tokenizer, IMAGE_TOKEN_INDEX)
        assert full[: len(base)] == base and base[: len(ids)] == ids, "tokenization is not prefix-stable"
        labels += [IGNORE_INDEX] * (len(base) - len(ids)) + full[len(base):]
        if lookups is not None and "target" in turn:
            row, col = turn["target"]
            cells = [(row + dr) * GRID + col + dc for dr in range(2) for dc in range(2)]
            answer = full[len(base):]
            lookups.append((len(base) + answer.index(word_id(turn["key"], tokenizer)),
                            len(base) + answer.index(word_id(turn["ident"], tokenizer)), cells))
        ids = full
    return ids, labels


def to_pixels(image):
    """uint8 [S, S, 3] -> normalised float [3, S, S] (the stand-in's image processor)."""
    x = torch.from_numpy(image).float().div(255.0)
    return ((x - SYSTEM_MEAN) / 0.5).permute(2, 0, 1).contiguous()


def batch(rng, tokenizer, size, mode="mixed", curriculum=False):
    """curriculum: scenes with fewer objects too, which lets the model learn to read
    a patch before it has to find the right one (evaluation always uses six)."""
    from llava.constants import IGNORE_INDEX
    rows, images, patches, lookups = [], [], [], []
    for _ in range(size):
        n_objects = rng.choice([1, 2, 3, 4, N_OBJECTS, N_OBJECTS]) if curriculum else N_OBJECTS
        scene, turns = make_conversation(rng, mode, n_turns=rng.choice([3, 4]), n_objects=n_objects)
        lookups.append([])
        rows.append(encode(turns, tokenizer, lookups[-1]))
        images.append(to_pixels(render(scene)))
        patches.append(patch_words(scene, tokenizer, rng))
    width = max(len(ids) for ids, _ in rows)
    input_ids = torch.zeros(size, width, dtype=torch.long)
    labels = torch.full((size, width), IGNORE_INDEX, dtype=torch.long)
    mask = torch.zeros(size, width, dtype=torch.bool)
    for i, (ids, lab) in enumerate(rows):
        input_ids[i, : len(ids)] = torch.tensor(ids)
        labels[i, : len(lab)] = torch.tensor(lab)
        mask[i, : len(ids)] = True
    return input_ids, labels, mask, torch.stack(images), torch.stack(patches), lookups


_WORD_IDS = {}


def patch_words(scene, tokenizer, rng):
    """For every image patch of an object, the token of one of its attributes (else -100).

    Used only as an auxiliary training target at the image positions, so that a
    patch's representation spells out what it shows (as LLaVA's image tokens do
    under the logit lens).  Without it this small model does not learn to read
    the image in the steps a CPU allows.
    """
    target = torch.full((GRID * GRID,), -100, dtype=torch.long)
    for obj in scene["objects"]:
        word = word_id(obj[rng.choice(["shape", "color", "mark"])], tokenizer)
        for dr in range(2):
            for dc in range(2):
                target[(obj["row"] + dr) * GRID + obj["col"] + dc] = word
    return target


# ----------------------------------------------------------------------
# Tokenizer, vision tower, model
# ----------------------------------------------------------------------
def build_tokenizer(out_dir, rng):
    import sentencepiece as spm
    from transformers import AutoTokenizer, LlamaTokenizer
    from llava.conversation import conv_templates

    corpus = os.path.join(out_dir, "corpus.txt")
    with open(corpus, "w") as handle:
        for _ in range(4000):
            _, turns = make_conversation(rng, "mixed")
            conv = conv_templates["llava_v1"].copy()
            for turn in turns:
                conv.append_message(conv.roles[0], turn["question"])
                conv.append_message(conv.roles[1], turn["answer"])
            for line in conv.get_prompt().replace("</s>", "\n").split("\n"):
                if line.strip():
                    handle.write(line.strip() + "\n")
    prefix = os.path.join(out_dir, "spm")
    spm.SentencePieceTrainer.train(
        input=corpus, model_prefix=prefix, vocab_size=256, model_type="bpe",
        character_coverage=1.0, normalization_rule_name="identity", hard_vocab_limit=False,
        unk_id=0, bos_id=1, eos_id=2, pad_id=-1, user_defined_symbols=["\n"],
    )
    # Save and reload: only the reloaded tokenizer treats "</s>" as one special
    # token, and that is the tokenizer the evaluators will use.
    path = os.path.join(out_dir, "tokenizer")
    LlamaTokenizer(vocab_file=prefix + ".model", legacy=True).save_pretrained(path)
    tokenizer = AutoTokenizer.from_pretrained(path, use_fast=False)
    assert tokenizer("a .</s>USER", add_special_tokens=False).input_ids.count(tokenizer.eos_token_id) == 1
    return tokenizer


def build_vision_tower(path, seed):
    from transformers import CLIPImageProcessor, CLIPVisionConfig, CLIPVisionModel
    torch.manual_seed(seed)
    config = CLIPVisionConfig(
        hidden_size=32, intermediate_size=64, num_hidden_layers=1, num_attention_heads=2,
        image_size=SIZE, patch_size=CELL, projection_dim=32,
    )
    CLIPVisionModel(config).save_pretrained(path)
    CLIPImageProcessor(
        size={"shortest_edge": SIZE}, crop_size={"height": SIZE, "width": SIZE},
        do_resize=True, do_center_crop=True, do_normalize=True, do_convert_rgb=True,
        image_mean=[SYSTEM_MEAN] * 3, image_std=[0.5] * 3,
    ).save_pretrained(path)


def build_model(tokenizer, clip_path, args):
    from llava.model.language_model.llava_llama import LlavaConfig, LlavaLlamaForCausalLM
    torch.manual_seed(args.seed)
    config = LlavaConfig(
        vocab_size=len(tokenizer), hidden_size=args.hidden, intermediate_size=3 * args.hidden,
        num_hidden_layers=args.layers, num_attention_heads=args.heads, max_position_embeddings=2048,
        bos_token_id=1, eos_token_id=2, pad_token_id=0,
        mm_vision_tower=os.path.abspath(clip_path), mm_hidden_size=32, mm_projector_type="mlp2x_gelu",
        mm_vision_select_layer=-2, mm_vision_select_feature="patch",
        mm_use_im_start_end=False, mm_use_im_patch_token=False,
    )
    model = LlavaLlamaForCausalLM(config)
    model.get_vision_tower().load_model()
    return model


@torch.no_grad()
def exact_match(model, tokenizer, rng, mode, size=48):
    """Share of answers whose every token is predicted correctly (teacher forcing)."""
    from llava.constants import IGNORE_INDEX
    model.eval()
    input_ids, labels, mask, images, _, _ = batch(rng, tokenizer, size, mode)
    logits = model(input_ids=input_ids, attention_mask=mask, images=images).logits
    n_image = logits.shape[1] - input_ids.shape[1]
    hits, total = 0, 0
    for i in range(size):
        row = labels[i]
        image_at = int((input_ids[i] == -200).nonzero()[0])
        positions = (row != IGNORE_INDEX).nonzero().flatten().tolist()
        ok, previous = True, None
        for p in positions + [None]:
            if previous is not None and (p is None or p != previous + 1):
                hits += int(ok)
                total += 1
                ok = True
            if p is not None:
                shifted = p + n_image if p > image_at else p
                ok &= int(logits[i, shifted - 1].argmax()) == int(row[p])
            previous = p
    model.train()
    return hits / max(1, total)


def train(model, tokenizer, args):
    rng = random.Random(args.seed + 1)
    val_rng_seed = args.seed + 2
    params = [p for name, p in model.named_parameters() if "vision_tower" not in name]
    for name, p in model.named_parameters():
        p.requires_grad = "vision_tower" not in name
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01, betas=(0.9, 0.98))
    model.train()
    start, best = time.time(), 0.0
    for step in range(1, args.steps + 1):
        warm = min(1.0, step / 200)
        decay = 0.5 * (1 + math.cos(math.pi * step / args.steps))
        for group in optimizer.param_groups:
            group["lr"] = args.lr * warm * (0.1 + 0.9 * decay)
        input_ids, labels, mask, images, patches, lookups = batch(
            rng, tokenizer, args.batch_size, curriculum=True)
        out = model(input_ids=input_ids, attention_mask=mask, labels=labels, images=images,
                    output_attentions=True)
        image_at = (input_ids == -200).long().argmax(dim=1)                       # [B]
        index = image_at[:, None] + torch.arange(GRID * GRID)[None, :]            # [B, patches]
        patch_logits = out.logits.gather(1, index[:, :, None].expand(-1, -1, out.logits.shape[-1]))
        aux = torch.nn.functional.cross_entropy(
            patch_logits.reshape(-1, patch_logits.shape[-1]), patches.reshape(-1), ignore_index=-100)
        # Attention guide for the two hops of the lookup, on ONE head each: at the
        # position that must produce the attribute, (1) a head of an early layer
        # is pointed at the word naming the object, (2) a head of a later layer at
        # the object's patches.  This only breaks the start-up plateau; every
        # other head and layer is free.
        guide = out.logits.new_zeros(())
        first = out.attentions[args.guide_layer - 2][:, 0]                        # [B, T, T]
        second = out.attentions[args.guide_layer][:, 0]
        shift = GRID * GRID - 1                                                   # image placeholder -> patches
        count = 0
        for row, items in enumerate(lookups):
            for position, ident, cells in items:
                query = position + shift - 1
                columns = torch.tensor(cells) + int(image_at[row])
                guide = guide - torch.log(second[row, query, columns].sum() + 1e-6)
                guide = guide - torch.log(first[row, query, ident + shift] + 1e-6)
                count += 1
        guide = guide / max(1, count)
        loss = out.loss + args.aux_weight * aux + args.guide_weight * guide
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        if step % args.log_every == 0 or step == args.steps:
            scores = {m: exact_match(model, tokenizer, random.Random(val_rng_seed), m) for m in ("mixed", "shift", "same")}
            best = max(best, scores["mixed"])
            print(f"step {step:5d}  loss {out.loss.item():.4f}  patch-loss {aux.item():.3f}  "
                  f"guide {guide.item():.3f}  exact-match "
                  + "  ".join(f"{m} {v:.3f}" for m, v in scores.items())
                  + f"  ({time.time() - start:.0f}s)", flush=True)
            if min(scores["shift"], scores["same"]) >= args.stop_at:
                break
    return best


def write_eval_sets(out_dir, args):
    image_dir = os.path.join(out_dir, "data", "images")
    os.makedirs(image_dir, exist_ok=True)
    for mode in ("shift", "same", "mixed"):
        rng = random.Random(f"{args.seed}-{mode}")
        items = []
        for index in range(args.eval_items):
            scene, turns = make_conversation(rng, mode, n_turns=4)
            name = f"{mode}_{index:04d}.png"
            Image.fromarray(render(scene)).save(os.path.join(image_dir, name))
            items.append({"image": name, "turns": turns, "objects": scene["objects"]})
        with open(os.path.join(out_dir, "data", f"eval_{mode}.json"), "w") as handle:
            json.dump(items, handle, indent=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--eval-items", type=int, default=150)
    parser.add_argument("--log-every", type=int, default=250)
    parser.add_argument("--stop-at", type=float, default=0.985)
    parser.add_argument("--guide-weight", type=float, default=0.5,
                        help="weight of the attention guide (training aid, one head in two layers)")
    parser.add_argument("--guide-layer", type=int, default=3)
    parser.add_argument("--aux-weight", type=float, default=0.5,
                        help="weight of the per-patch word loss (training aid, see patch_words)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--image-position", default="first", choices=["first", "last"],
                        help="Turn-1 layout the stand-in is trained for (must match the evaluators).")
    args = parser.parse_args()
    global IMAGE_POSITION
    IMAGE_POSITION = args.image_position
    torch.set_num_threads(args.threads)

    os.makedirs(args.out, exist_ok=True)
    clip_path = os.path.join(args.out, "clip-tiny")
    model_path = os.path.join(args.out, "llava-v1-standin")
    tokenizer = build_tokenizer(args.out, random.Random(args.seed))
    build_vision_tower(clip_path, args.seed)
    model = build_model(tokenizer, clip_path, args)
    count = sum(p.numel() for name, p in model.named_parameters() if "vision_tower" not in name)
    print(f"stand-in: {args.layers} layers, hidden {args.hidden}, {args.heads} heads, "
          f"{count / 1e6:.2f}M trainable parameters, vocab {len(tokenizer)}", flush=True)

    # The training pixels must be what the evaluators' image processor produces.
    from llava.mm_utils import process_images
    scene, _ = make_conversation(random.Random(1))
    processor = model.get_vision_tower().image_processor
    reference = process_images([Image.fromarray(render(scene))], processor,
                               argparse.Namespace(image_aspect_ratio="pad"))[0]
    assert torch.allclose(reference, to_pixels(render(scene)), atol=1e-5), "image processing mismatch"

    best = train(model, tokenizer, args)
    model.eval()
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    write_eval_sets(args.out, args)
    print(f"saved {model_path} (best mixed exact-match {best:.3f}) and {args.out}/data", flush=True)


if __name__ == "__main__":
    main()
