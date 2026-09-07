import argparse
import json
import os
from io import BytesIO

import requests
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from qwen_generation_utils import make_context


def load_image(image_file):
    if image_file.startswith("http://") or image_file.startswith("https://"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


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
    # Do not pass max_window_size=None; local make_context defaults to 6144.
    raw_text, context_tokens = make_context(
        tokenizer,
        query,
        history=history,
        system="You are a helpful assistant.",
        chat_format="chatml",
    )
    return raw_text, context_tokens


def normalize_question(question, data_path):
    if "detail_1k" in data_path:
        return question.replace("<image>", "")
    return question.replace("<image>", "")


def get_turns(item):
    if "turns" in item:
        return item["turns"]
    return [{"question": item["question"], "answer": item.get("answer", "")}]


def generate_one_turn(model, tokenizer, input_ids, args):
    attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=input_ids.device)
    output_ids = model.generate(
        input_ids,
        do_sample=True if args.temperature > 0 else False,
        temperature=args.temperature,
        top_p=args.top_p,
        num_beams=args.num_beams,
        max_new_tokens=args.max_new_tokens,
        use_cache=True,
        attention_mask=attention_mask,
    )
    return tokenizer.decode(output_ids[0, input_ids.shape[1]:], skip_special_tokens=True).strip()


def main(args):
    with open(args.data_path, "r") as f:
        data = json.load(f)

    outputs_data_json = []
    model_name = os.path.basename(os.path.normpath(args.model_path))
    dataset_name = os.path.splitext(os.path.basename(args.data_path))[0]

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        device_map=args.device_map,
        trust_remote_code=True,
        torch_dtype=torch.float16,
    ).eval()

    os.makedirs("logs_temp", exist_ok=True)
    data = data[:args.eval_samples]

    for item in tqdm(data):
        image_path = os.path.join(args.image_path, item["image"])
        turns = get_turns(item)
        history = []

        if "turns" in item:
            output_data_json = {
                "image": item["image"],
                "turns": [],
            }
            for turn_id, turn in enumerate(turns):
                question = normalize_question(turn["question"], args.data_path)
                query = build_qwen_query(tokenizer, image_path, question, use_image=(turn_id == 0))
                _, context_tokens = build_qwen_context(tokenizer, query, history)
                input_ids = torch.tensor([context_tokens], device=model.device)

                outputs_generate = generate_one_turn(model, tokenizer, input_ids, args)
                output_data_json["turns"].append({
                    "question": turn["question"],
                    "answer": outputs_generate,
                })

                # For conversion/generation, use generated output as dialogue history
                # so later questions are asked in the actual generated conversation.
                history.append((query, outputs_generate))
        else:
            question = normalize_question(item["question"], args.data_path)
            query = build_qwen_query(tokenizer, image_path, question, use_image=True)
            _, context_tokens = build_qwen_context(tokenizer, query, history=[])
            input_ids = torch.tensor([context_tokens], device=model.device)

            outputs_generate = generate_one_turn(model, tokenizer, input_ids, args)
            output_data_json = {
                "image": item["image"],
                "question": item["question"],
                "answer": outputs_generate,
            }

        outputs_data_json.append(output_data_json)

    out_dir = os.path.join("./playground/data", dataset_name)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"rouge-{model_name}-{dataset_name}.json")
    with open(out_path, "w") as f:
        json.dump(outputs_data_json, f, indent=4)
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="./models/qwen-vl-chat")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--data-path", type=str, default="./playground/data/detail_1k/detail_1k.json")
    parser.add_argument("--image-path", type=str, default="./playground/data/detail_1k/")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--device-map", type=str, default="cuda")
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
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
    args = parser.parse_args()
    main(args)
