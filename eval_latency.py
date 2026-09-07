import argparse
import json
import os
import time
from io import BytesIO

import requests
import torch
from PIL import Image
from tqdm import tqdm

from kv_cache import (
    ElasticCache,
    LocalCache,
    H2OCache,
    HelasticCache,
    MultimodalElasticCache,
    MultimodalHelasticCache,
)
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path


def load_image(image_file):
    if image_file.startswith("http://") or image_file.startswith("https://"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def infer_conv_mode(model_name):
    model_name = model_name.lower()
    if "llama-2" in model_name:
        return "llava_llama_2"
    if "v1" in model_name:
        return "llava_v1"
    if "mpt" in model_name:
        return "mpt"
    return "llava_v0"


def get_layer_num(model_name):
    return 32 if "7b" in model_name.lower() else 40


def find_llava_image_span(input_ids, default_image_token_len=576):
    """
    LLaVA input_ids contain one IMAGE_TOKEN_INDEX placeholder. Inside the model,
    that placeholder is replaced by a block of visual tokens, usually 576 for
    LLaVA-v1.5 CLIP ViT-L/14 336px. We use the placeholder position as the
    visual-token block start and pass the expanded visual length to multimodal
    cache methods.
    """
    image_positions = (input_ids[0] == IMAGE_TOKEN_INDEX).nonzero(as_tuple=True)[0]
    if image_positions.numel() == 0:
        return None, 0
    return image_positions[0].item(), default_image_token_len


def create_kv_cache(args, model_name, k_seq_dim, v_seq_dim, image_start=None, image_token_len=0, batch_size=1):
    # kv_cache.py concatenates attention tensors across layers on dim=0.
    # With batched synthetic inputs, this becomes num_layers * batch_size.
    # score_sum must therefore allocate the same first dimension.
    layer_num = get_layer_num(model_name) * batch_size

    common = dict(
        start_size=args.start_size,
        recent_size=args.recent_size,
        k_seq_dim=k_seq_dim,
        v_seq_dim=v_seq_dim,
        ratio=args.ratio,
    )

    if args.method == "none":
        return None
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


def maybe_reset_cache(kv_cache):
    if kv_cache is None:
        return
    if hasattr(kv_cache, "reset"):
        kv_cache.reset()


def maybe_compress(kv_cache, past_key_values, num_of_token, attentions):
    if kv_cache is None:
        return past_key_values
    return kv_cache(past_key_values, num_of_token, attentions)


def get_turns(item):
    if "turns" in item and isinstance(item["turns"], list):
        return item["turns"]
    return [{"question": item["question"], "answer": item.get("answer", "") }]


def build_full_context_for_turn(args, tokenizer, item, turns, turn_id):
    conv = conv_templates[args.conv_mode].copy()

    for i in range(turn_id):
        conv.append_message(conv.roles[0], turns[i]["question"])
        conv.append_message(conv.roles[1], turns[i].get("answer", ""))

    question = turns[turn_id]["question"]
    if DEFAULT_IMAGE_TOKEN not in question:
        question = question + "\n" + DEFAULT_IMAGE_TOKEN

    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(
        prompt,
        tokenizer,
        IMAGE_TOKEN_INDEX,
        return_tensors="pt",
    ).unsqueeze(0).cuda()

    answer = turns[turn_id].get("answer", "")
    if answer:
        answer_ids = tokenizer.encode(answer, return_tensors="pt").cuda()
        if answer_ids.shape[-1] > 0 and tokenizer.bos_token_id is not None:
            if answer_ids[0, 0].item() == tokenizer.bos_token_id:
                answer_ids = answer_ids[:, 1:]
    else:
        answer_ids = torch.ones((1, args.decode_tokens), dtype=torch.long, device=input_ids.device)

    if answer_ids.shape[-1] == 0:
        answer_ids = torch.ones((1, args.decode_tokens), dtype=torch.long, device=input_ids.device)

    if answer_ids.shape[-1] > args.decode_tokens:
        answer_ids = answer_ids[:, : args.decode_tokens]

    return input_ids, answer_ids


def run_latency_once(model, input_ids, answer_ids, image_tensor, kv_cache, args):
    past_key_values = None
    num_of_token = 0

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start_time = time.time()

    with torch.no_grad():
        outputs = model(
            input_ids,
            images=image_tensor,
            past_key_values=past_key_values,
            use_cache=True,
            output_attentions=True,
        )
        logits = outputs.logits.view(input_ids.shape[0], -1, model.config.vocab_size)
        num_of_token += logits.shape[1]
        past_key_values = outputs.past_key_values
        attentions = outputs.attentions
        past_key_values = maybe_compress(kv_cache, past_key_values, num_of_token, attentions)

        decode_steps = max(0, min(args.decode_tokens, answer_ids.shape[-1]) - 1)
        for idx in range(decode_steps):
            cur_input_ids = answer_ids[:, idx: idx + 1]
            outputs = model(
                cur_input_ids,
                past_key_values=past_key_values,
                use_cache=True,
                output_attentions=True,
            )
            logits = outputs.logits.view(input_ids.shape[0], -1, model.config.vocab_size)
            num_of_token += logits.shape[1]
            past_key_values = outputs.past_key_values
            attentions = outputs.attentions
            past_key_values = maybe_compress(kv_cache, past_key_values, num_of_token, attentions)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time() - start_time


def run_dataset_latency(args, tokenizer, model, image_processor, model_name):
    with open(args.data_path, "r") as f:
        data = json.load(f)

    data = data[: args.eval_samples]
    k_seq_dim = v_seq_dim = 2
    latencies = []

    for item in tqdm(data):
        image_path = os.path.join(args.image_path, item["image"])
        image = load_image(image_path)
        image_tensor = process_images([image], image_processor, args)
        image_tensor = image_tensor.to(model.device, dtype=torch.bfloat16)

        turns = get_turns(item)
        for turn_id in range(len(turns)):
            input_ids, answer_ids = build_full_context_for_turn(args, tokenizer, item, turns, turn_id)
            image_start, image_token_len = find_llava_image_span(input_ids, args.image_token_len)

            kv_cache = create_kv_cache(
                args,
                model_name,
                k_seq_dim,
                v_seq_dim,
                image_start=image_start,
                image_token_len=image_token_len,
                batch_size=input_ids.shape[0],
            )
            maybe_reset_cache(kv_cache)

            elapsed = run_latency_once(model, input_ids, answer_ids, image_tensor, kv_cache, args)
            latencies.append(elapsed)

    return latencies


def run_synthetic_latency(args, model, model_name):
    k_seq_dim = v_seq_dim = 2
    input_ids = torch.ones([args.batch_size, args.synthetic_prompt_len], dtype=torch.long).cuda()
    answer_ids = torch.ones([args.batch_size, args.decode_tokens], dtype=torch.long).cuda()

    image_start = args.synthetic_image_start
    image_token_len = args.image_token_len if image_start is not None else 0
    kv_cache = create_kv_cache(
        args,
        model_name,
        k_seq_dim,
        v_seq_dim,
        image_start=image_start,
        image_token_len=image_token_len,
        batch_size=args.batch_size,
    )
    maybe_reset_cache(kv_cache)

    elapsed = run_latency_once(model, input_ids, answer_ids, None, kv_cache, args)
    return [elapsed]


def main(args):
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path,
        args.model_base,
        model_name,
        args.load_8bit,
        args.load_4bit,
        device=args.device,
    )

    inferred_conv_mode = infer_conv_mode(model_name)
    if args.conv_mode is not None and inferred_conv_mode != args.conv_mode:
        print(f"[WARNING] inferred conv_mode={inferred_conv_mode}, using user conv_mode={args.conv_mode}")
    else:
        args.conv_mode = inferred_conv_mode

    os.makedirs("logs/", exist_ok=True)

    # Warmup reduces first-run CUDA/kernel-load noise.
    if args.warmup:
        warmup_input = torch.ones([1, 16], dtype=torch.long).cuda()
        with torch.no_grad():
            _ = model(warmup_input, images=None, use_cache=True, output_attentions=True)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    if args.data_path:
        latencies = run_dataset_latency(args, tokenizer, model, image_processor, model_name)
    else:
        latencies = run_synthetic_latency(args, model, model_name)

    avg_latency = sum(latencies) / max(1, len(latencies))
    print(f"method={args.method}, ratio={args.ratio}, samples={len(latencies)}")
    print(f"avg_latency_sec={avg_latency}")
    print(f"total_latency_sec={sum(latencies)}")

    if args.exp_name:
        with open(f"logs/{args.exp_name}.txt", "a") as f:
            f.write(f"method={args.method}, ratio={args.ratio}, samples={len(latencies)}, avg_latency_sec={avg_latency}, total_latency_sec={sum(latencies)}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="/home/stu6/models/llava-v1.5-7b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--data-path", type=str, default=None, help="If set, run latency on MM-Vet/MM-Vet-turns style JSON. If omitted, run synthetic latency.")
    parser.add_argument("--image-path", type=str, default="./playground/data/mm-vet/images")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--image-aspect-ratio", type=str, default="pad")
    parser.add_argument("--start-size", type=int, default=1)
    parser.add_argument("--recent-size", type=int, default=2047)
    parser.add_argument("--eval-samples", type=int, default=100)
    parser.add_argument("--exp-name", type=str, default="")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--method",
        type=str,
        default="elastic",
        choices=["none", "elastic", "helastic", "multimodal_elastic", "multimodal_helastic", "local", "h2o"],
    )
    parser.add_argument("--ratio", type=float, default=0.2)
    parser.add_argument("--decode-tokens", type=int, default=512)
    parser.add_argument("--synthetic-prompt-len", type=int, default=900)
    parser.add_argument("--synthetic-image-start", type=int, default=None)
    parser.add_argument("--image-token-len", type=int, default=576)
    parser.add_argument("--warmup", action="store_true")
    args = parser.parse_args()
    main(args)
