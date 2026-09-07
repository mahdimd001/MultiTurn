import argparse
import json
import os
from io import BytesIO

import requests
import torch
import types
from PIL import Image
from rouge import Rouge
from tqdm import tqdm

from cache_generate import generate, greedy_search, sample
from kv_cache import (
    ElasticCache,
    H2OCache,
    HelasticCache,
    LocalCache,
    MultimodalElasticCache,
    MultimodalHelasticCache,
)
from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import SeparatorStyle, conv_templates
from llava.mm_utils import KeywordsStoppingCriteria, get_model_name_from_path, process_images, tokenizer_image_token
from llava.model.builder import load_pretrained_model


def load_image(image_file):
    if image_file.startswith("http://") or image_file.startswith("https://"):
        response = requests.get(image_file)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def normalize_text(text, max_words=256):
    text = (text or "").replace("</s>", "").replace("<AND>", " ").strip()
    return " ".join(text.split()[:max_words])


def infer_conv_mode(model_name, requested_conv_mode):
    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if requested_conv_mode is not None and conv_mode != requested_conv_mode:
        print(
            "[WARNING] auto inferred conversation mode is {}, while --conv-mode is {}; using {}".format(
                conv_mode, requested_conv_mode, requested_conv_mode
            )
        )
        return requested_conv_mode
    return conv_mode


def build_kv_cache(args, model_name, k_seq_dim, v_seq_dim, image_start=None, image_token_len=0):
    layer_num = 32 if "7b" in model_name.lower() else 40

    common = dict(
        start_size=args.start_size,
        recent_size=args.recent_size,
        k_seq_dim=k_seq_dim,
        v_seq_dim=v_seq_dim,
        ratio=args.ratio,
    )

    if args.method == "elastic":
        return ElasticCache(**common, layer_num=layer_num)
    if args.method == "helastic":
        return HelasticCache(**common, layer_num=layer_num)
    if args.method == "multimodal_elastic":
        return MultimodalElasticCache(
            **common,
            layer_num=layer_num,
            image_start=image_start,
            image_token_len=image_token_len,
        )
    if args.method == "multimodal_helastic":
        return MultimodalHelasticCache(
            **common,
            layer_num=layer_num,
            image_start=image_start,
            image_token_len=image_token_len,
        )
    if args.method == "local":
        return LocalCache(**common)
    if args.method == "h2o":
        return H2OCache(**common)

    raise ValueError(f"Invalid method: {args.method}")


def get_image_position(input_ids):
    image_positions = (input_ids[0] == IMAGE_TOKEN_INDEX).nonzero(as_tuple=True)[0]
    if image_positions.numel() == 0:
        return None, 0
    return image_positions[0].item(), 576


def main(args):
    with open(args.data_path, "r") as f:
        data = json.load(f)

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, _ = load_pretrained_model(
        args.model_path,
        args.model_base,
        model_name,
        args.load_8bit,
        args.load_4bit,
        device=args.device,
    )

    model.generate = types.MethodType(generate, model)
    model.sample = types.MethodType(sample, model)
    model.greedy_search = types.MethodType(greedy_search, model)

    args.conv_mode = infer_conv_mode(model_name, args.conv_mode)
    k_seq_dim = v_seq_dim = 2
    rouge_scorer = Rouge()

    data = data[: args.eval_samples]
    os.makedirs(args.output_dir, exist_ok=True)

    exp_name = args.exp_name or f"{args.method}_ratio{args.ratio}"
    score_path = os.path.join(args.output_dir, f"{exp_name}.txt")
    pred_path = os.path.join(args.output_dir, f"{exp_name}_predictions.json")

    all_turn_scores = []
    per_item_outputs = []

    for item_idx, item in enumerate(tqdm(data)):
        conv = conv_templates[args.conv_mode].copy()
        image_path = os.path.join(args.image_path, item["image"])
        image = load_image(image_path)
        image_tensor = process_images([image], image_processor, args)
        image_tensor = image_tensor.to(model.device, dtype=torch.bfloat16)

        item_output = {
            "image": item["image"],
            "turns": [],
        }

        for turn_idx, turn in enumerate(item.get("turns", [])):
            question = turn.get("question", "")
            reference = turn.get("answer", "")

            # LLaVA needs the image token once. Keep it in the first user turn only.
            if turn_idx == 0 and DEFAULT_IMAGE_TOKEN not in question:
                question = question + "\n" + DEFAULT_IMAGE_TOKEN

            conv.append_message(conv.roles[0], question)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(
                prompt,
                tokenizer,
                IMAGE_TOKEN_INDEX,
                return_tensors="pt",
            ).unsqueeze(0).to(args.device)

            image_start, image_token_len = get_image_position(input_ids)
            kv_cache = build_kv_cache(
                args,
                model_name,
                k_seq_dim,
                v_seq_dim,
                image_start=image_start,
                image_token_len=image_token_len,
            )

            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            stopping_criteria = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)

            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                do_sample=True if (args.temperature > 0 and args.ratio == 0) else False,
                temperature=args.temperature if args.ratio == 0 else 0,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
                kv_cache_criteria=kv_cache,
            )

            generated = tokenizer.decode(output_ids[0, input_ids.shape[1] :]).strip()
            generated_norm = normalize_text(generated, args.score_max_words)
            reference_norm = normalize_text(reference, args.score_max_words)

            if generated_norm and reference_norm:
                score = rouge_scorer.get_scores(generated_norm, reference_norm)[0]["rouge-l"]["f"]
            else:
                score = 0.0

            all_turn_scores.append(score)
            item_output["turns"].append(
                {
                    "turn_idx": turn_idx,
                    "question": question.replace("\n" + DEFAULT_IMAGE_TOKEN, ""),
                    "reference": reference,
                    "prediction": generated,
                    "rouge_l_f": score,
                }
            )

            # Replace the assistant placeholder with the generated answer before next turn.
            conv.messages[-1][-1] = generated

        per_item_outputs.append(item_output)

    avg_rouge = sum(all_turn_scores) / len(all_turn_scores) if all_turn_scores else 0.0

    with open(score_path, "a") as f:
        f.write(f"{avg_rouge}\n")

    with open(pred_path, "w") as f:
        json.dump(
            {
                "avg_rouge_l_f": avg_rouge,
                "num_items": len(per_item_outputs),
                "num_turns_scored": len(all_turn_scores),
                "results": per_item_outputs,
            },
            f,
            indent=4,
        )

    print(f"Average ROUGE-L F1 over {len(all_turn_scores)} turns: {avg_rouge:.6f}")
    print(f"Saved score to: {score_path}")
    print(f"Saved predictions to: {pred_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="./models/llava-v1.5-7b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--data-path", type=str, default="./playground/data/mm-vet/mm-vet.json")
    parser.add_argument("--image-path", type=str, default="./playground/data/mm-vet/images")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--image-aspect-ratio", type=str, default="pad")
    parser.add_argument("--start-size", type=int, default=1)
    parser.add_argument("--recent-size", type=int, default=2047)
    parser.add_argument("--eval-samples", type=int, default=218)
    parser.add_argument("--exp-name", type=str, default="")
    parser.add_argument("--method", type=str, default="elastic")
    parser.add_argument("--ratio", type=float, default=0.2)
    parser.add_argument("--output-dir", type=str, default="logs_rouge_llava")
    parser.add_argument("--score-max-words", type=int, default=256)
    args = parser.parse_args()
    main(args)
