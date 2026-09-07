import argparse
import json
import os
from io import BytesIO

import requests
import torch
from PIL import Image
from rouge import Rouge
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from cache_generate_qwen import generate, greedy_search, sample
from qwen_generation_utils import make_context
from kv_cache_qwen import (
    ElasticCache,
    LocalCache,
    H2OCache,
    HelasticCache,
    MultimodalElasticCache,
    MultimodalHelasticCache,
)
import types


def load_image(image_file):
    if image_file.startswith("http://") or image_file.startswith("https://"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def find_qwen_image_span(tokenizer, input_ids):
    """Return the Qwen-VL image token span for one image.

    Old Qwen-VL expands one image into 256 image pad tokens. We locate the
    first image pad token in the constructed prompt and use that as the start.
    """
    image_token_len = 256
    image_pad_id = tokenizer.img_pad_id
    pos = (input_ids[0] == image_pad_id).nonzero(as_tuple=True)[0]
    if pos.numel() == 0:
        return None, 0
    return pos[0].item(), image_token_len


def build_qwen_query(tokenizer, image_path, question, use_image):
    if use_image:
        return tokenizer.from_list_format([
            {"image": image_path},
            {"text": question},
        ])
    return tokenizer.from_list_format([
        {"text": question},
    ])


def build_qwen_context(tokenizer, query, history):
    # Do not pass max_window_size=None. The local qwen_generation_utils.py
    # already defaults to 6144, and passing None causes a TypeError.
    raw_text, context_tokens = make_context(
        tokenizer,
        query,
        history=history,
        system="You are a helpful assistant.",
        chat_format="chatml",
    )
    return raw_text, context_tokens


def create_kv_cache(args, k_seq_dim, v_seq_dim, image_start=None, image_token_len=0):
    if args.method == "none":
        return None
    if args.method == "elastic":
        return ElasticCache(
            start_size=args.start_size,
            recent_size=args.recent_size,
            k_seq_dim=k_seq_dim,
            v_seq_dim=v_seq_dim,
            ratio=args.ratio,
            layer_num=32,
        )
    if args.method == "helastic":
        return HelasticCache(
            start_size=args.start_size,
            recent_size=args.recent_size,
            k_seq_dim=k_seq_dim,
            v_seq_dim=v_seq_dim,
            ratio=args.ratio,
            layer_num=32,
        )
    if args.method == "multimodal_elastic":
        return MultimodalElasticCache(
            start_size=args.start_size,
            recent_size=args.recent_size,
            k_seq_dim=k_seq_dim,
            v_seq_dim=v_seq_dim,
            ratio=args.ratio,
            layer_num=32,
            image_start=image_start,
            image_token_len=image_token_len,
        )
    if args.method == "multimodal_helastic":
        return MultimodalHelasticCache(
            start_size=args.start_size,
            recent_size=args.recent_size,
            k_seq_dim=k_seq_dim,
            v_seq_dim=v_seq_dim,
            ratio=args.ratio,
            layer_num=32,
            image_start=image_start,
            image_token_len=image_token_len,
        )
    if args.method == "local":
        return LocalCache(
            start_size=args.start_size,
            recent_size=args.recent_size,
            k_seq_dim=k_seq_dim,
            v_seq_dim=v_seq_dim,
            ratio=args.ratio,
        )
    if args.method == "h2o":
        return H2OCache(
            start_size=args.start_size,
            recent_size=args.recent_size,
            k_seq_dim=k_seq_dim,
            v_seq_dim=v_seq_dim,
            ratio=args.ratio,
        )
    raise ValueError(f"Invalid method: {args.method}")


def normalize_question(question, data_path):
    if "detail_1k" in data_path:
        return question.replace("<image>", "")
    return question.replace("<image>", "")


def get_turns(item):
    if "turns" in item:
        return item["turns"]
    return [{"question": item["question"], "answer": item["answer"]}]


def generate_one_turn(model, tokenizer, input_ids, kv_cache, args):
    attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=input_ids.device)
    output_ids = model.generate(
        input_ids,
        do_sample=True if (args.temperature > 0 and args.ratio == 0) else False,
        temperature=args.temperature if args.ratio == 0 else 0,
        top_p=args.top_p,
        num_beams=args.num_beams,
        max_new_tokens=args.max_new_tokens,
        use_cache=True,
        kv_cache_criteria=kv_cache,
        attention_mask=attention_mask,
    )
    return tokenizer.decode(output_ids[0, input_ids.shape[1]:], skip_special_tokens=True).strip()


def main(args):
    with open(args.data_path, "r") as f:
        data = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        device_map=args.device_map,
        trust_remote_code=True,
        torch_dtype=torch.float16,
    ).eval()

    # Patch in ElasticCache-aware generation functions.
    model.generate = types.MethodType(generate, model)
    model.sample = types.MethodType(sample, model)
    model.greedy_search = types.MethodType(greedy_search, model)

    k_seq_dim = v_seq_dim = 1
    os.makedirs("logs_rouge_qwen", exist_ok=True)
    data = data[:args.eval_samples]

    rouge_metric = Rouge()
    score_all = []

    for item in tqdm(data):
        image_path = os.path.join(args.image_path, item["image"])
        history = []
        turns = get_turns(item)

        # Stable full-context mode: each turn rebuilds the full dialogue prompt.
        # This avoids cross-turn KV-mask mismatch but still evaluates cache
        # compression during generation for this turn.
        for turn_id, turn in enumerate(turns):
            question = normalize_question(turn["question"], args.data_path)
            answer = turn["answer"]

            query = build_qwen_query(tokenizer, image_path, question, use_image=(turn_id == 0))
            _, context_tokens = build_qwen_context(tokenizer, query, history)
            input_ids = torch.tensor([context_tokens], device=model.device)

            image_start, image_token_len = find_qwen_image_span(tokenizer, input_ids)
            kv_cache = create_kv_cache(
                args,
                k_seq_dim=k_seq_dim,
                v_seq_dim=v_seq_dim,
                image_start=image_start,
                image_token_len=image_token_len,
            )

            outputs_generate = generate_one_turn(model, tokenizer, input_ids, kv_cache, args)
            scores = rouge_metric.get_scores(outputs_generate or " ", answer or " ")
            score_all.append(scores[0]["rouge-l"]["f"])

            # Use the ground-truth answer as history so all methods share the same
            # multi-turn context and ROUGE compares each target independently.
            history.append((query, answer))

    rouge_score = sum(score_all) / len(score_all) if score_all else 0.0
    with open(f"logs_rouge_qwen/{args.exp_name}.txt", "a") as f:
        f.write(f"{rouge_score}\n\n")
    print(f"ROUGE-L: {rouge_score}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="./models/qwen-vl-chat")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--data-path", type=str, default="./playground/data/mm-vet/rouge-qwen-detail.json")
    parser.add_argument("--image-path", type=str, default="./playground/data")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--device-map", type=str, default="cuda")
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
    parser.add_argument("--exp-name", type=str, default="qwen-vl")
    parser.add_argument(
        "--method",
        type=str,
        default="elastic",
        choices=["none", "elastic", "helastic", "multimodal_elastic", "multimodal_helastic", "local", "h2o"],
    )
    parser.add_argument("--ratio", type=float, default=0.0)
    args = parser.parse_args()
    main(args)
