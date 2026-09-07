import torch
from tqdm import tqdm
import os
from torch.nn import CrossEntropyLoss
from kv_cache_qwen import (
    ElasticCache,
    LocalCache,
    H2OCache,
    HelasticCache,
    MultimodalElasticCache,
    MultimodalHelasticCache,
)
import json

device = "cuda"

import argparse
from PIL import Image
import requests
from io import BytesIO
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor


def load_image(image_file):
    """Load local/remote image. Kept for compatibility, although Qwen uses image_path directly."""
    if image_file.startswith("http://") or image_file.startswith("https://"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image



def find_qwen2_image_span(input_ids, processor):
    """
    Locate the image-token span in Qwen2-VL processor output.

    This is only used by multimodal cache classes. If the exact span cannot
    be located, return (None, 0) and the normal cache policies still work.
    """
    try:
        image_token_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        positions = (input_ids[0] == image_token_id).nonzero(as_tuple=True)[0]
        if positions.numel() == 0:
            return None, 0
        return positions[0].item(), positions.numel()
    except Exception:
        return None, 0


def build_qwen2_prompt(processor, messages):
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return text


def create_kv_cache(args, k_seq_dim, v_seq_dim, image_start=None, image_token_len=0):
    """
    Create the selected cache policy.

    For multimodal cache classes, image_start/image_token_len mark the Qwen image
    span so the cache can build image/text masks.
    """
    if args.method == "elastic":
        return ElasticCache(
            start_size=args.start_size,
            recent_size=args.recent_size,
            k_seq_dim=k_seq_dim,
            v_seq_dim=v_seq_dim,
            ratio=args.ratio,
            layer_num=32,
        )

    elif args.method == "helastic":
        return HelasticCache(
            start_size=args.start_size,
            recent_size=args.recent_size,
            k_seq_dim=k_seq_dim,
            v_seq_dim=v_seq_dim,
            ratio=args.ratio,
            layer_num=32,
        )

    elif args.method == "multimodal_elastic":
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

    elif args.method == "multimodal_helastic":
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

    elif args.method == "local":
        return LocalCache(
            start_size=args.start_size,
            recent_size=args.recent_size,
            k_seq_dim=k_seq_dim,
            v_seq_dim=v_seq_dim,
            ratio=args.ratio,
        )

    elif args.method == "h2o":
        return H2OCache(
            start_size=args.start_size,
            recent_size=args.recent_size,
            k_seq_dim=k_seq_dim,
            v_seq_dim=v_seq_dim,
            ratio=args.ratio,
        )
    elif args.method == "none":
        return None
    else:
        raise ValueError("Invalid method")


def reset_kv_cache(kv_cache):
    """
    Reset cache state before each image/sample.

    Some cache classes define reset(); older classes only have score tensors/flag,
    so this keeps backward compatibility.
    """
    try:
        if hasattr(kv_cache, "reset"):
            kv_cache.reset()
        else:
            if hasattr(kv_cache, "score_sum"):
                kv_cache.score_sum = torch.zeros_like(kv_cache.score_sum).cuda()
            if hasattr(kv_cache, "conn_score_sum"):
                kv_cache.conn_score_sum = torch.zeros_like(kv_cache.conn_score_sum).cuda()
            if hasattr(kv_cache, "text_image_score_sum"):
                kv_cache.text_image_score_sum = torch.zeros_like(kv_cache.text_image_score_sum).cuda()
            if hasattr(kv_cache, "image_text_score_sum"):
                kv_cache.image_text_score_sum = torch.zeros_like(kv_cache.image_text_score_sum).cuda()
            if hasattr(kv_cache, "flag"):
                kv_cache.flag = True
    except Exception:
        print("cannot reset kv_cache")


def main(args):
    # Load dataset. Supports both:
    #   1) single-turn items: {"image", "question", "answer"}
    #   2) multi-turn items:  {"image", "turns": [{"question", "answer"}, ...]}
    with open(args.data_path, "r") as f:
        data = json.load(f)

    # Load Qwen2-VL processor/model.
    #processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(
        args.model_path,
        min_pixels=256 * 28 * 28,
        max_pixels=512 * 28 * 28,
    )
    tokenizer = processor.tokenizer
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        #trust_remote_code=True,
    ).eval()

    loss_fn = CrossEntropyLoss(reduction="none")

    # Qwen2-VL/HF KV tensors use sequence dimension 2.
    k_seq_dim = v_seq_dim = 2

    os.makedirs("logs_ppl_qwen/", exist_ok=True)
    data = data[:args.eval_samples]

    nlls = []

    for item in tqdm(data):
        image_path = os.path.join(args.image_path, item["image"])

        # State shared across all turns for the same image.
        # This lets later turns reuse the KV cache from the first image/question turn.
        past_key_values = None
        num_of_token = 0
        prev_prompt_ids = []
        messages = []
        kv_cache = None

        # Convert single-turn data into the same format as multi-turn data.
        if "turns" in item:
            turns = item["turns"]
        else:
            turns = [{"question": item["question"], "answer": item["answer"]}]

        for turn_id, turn in enumerate(turns):
            question = turn["question"]
            answer = turn["answer"]

            # Some datasets include literal <image> in text; Qwen already receives
            # the image through tokenizer.from_list_format, so remove the text marker.
            if "detail_1k" in args.data_path:
                question = question.replace("<image>", "")

            # Build full chat prompt. First turn includes the image; later turns
            # include only text, while previous context is reused through KV cache.
            if turn_id == 0:
                messages.append({
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_path},
                        {"type": "text", "text": question},
                    ],
                })

                prompt_text = build_qwen2_prompt(processor, messages)
                image = load_image(image_path)
                inputs = processor(
                    text=[prompt_text],
                    images=[image] if turn_id == 0 else None,
                    return_tensors="pt",
                ).to(model.device)
                # inputs = processor(
                #     text=[prompt_text],
                #     images=[image],
                #     padding=True,
                #     return_tensors="pt",
                # ).to(model.device)

                image_start, image_token_len = find_qwen2_image_span(inputs.input_ids, processor)

                # Create cache after image_start/image_token_len are known.
                kv_cache = create_kv_cache(
                    args,
                    k_seq_dim,
                    v_seq_dim,
                    image_start=image_start,
                    image_token_len=image_token_len,
                )
                reset_kv_cache(kv_cache)

                input_ids = inputs.input_ids
                full_prompt_ids = input_ids[0].tolist()
                first_forward_inputs = inputs

            else:
                messages.append({
                    "role": "user",
                    "content": [{"type": "text", "text": question}],
                })

                prompt_text = build_qwen2_prompt(processor, messages)
                full_prompt_ids = tokenizer(prompt_text, return_tensors="pt").input_ids[0].tolist()

                # For later turns, send only the new suffix of the prompt.
                # Previous context is already represented in past_key_values.
                if full_prompt_ids[:len(prev_prompt_ids)] == prev_prompt_ids:
                    turn_prompt_ids = full_prompt_ids[len(prev_prompt_ids):]
                else:
                    turn_prompt_ids = full_prompt_ids

                input_ids = torch.tensor([turn_prompt_ids], device=model.device)
                first_forward_inputs = None

            prev_prompt_ids = full_prompt_ids

            # Tokenize answer. PPL is computed by teacher-forcing answer tokens.
            #answer_ids = tokenizer.encode(answer, return_tensors="pt").cuda()[:, 1:]
            answer_ids = processor.tokenizer.encode(
                answer,
                return_tensors="pt",
                add_special_tokens=False
            ).to(model.device)

            # Skip very short answers that do not provide next-token labels.
            if answer_ids.shape[-1] <= 1:
                messages.append({"role": "assistant", "content": answer})
                continue

            for idx in range(0, answer_ids.shape[-1] - 1):
                with torch.no_grad():
                    if idx == 0:
                        # First answer token for this turn:
                        # feed the question/context tokens.
                        if first_forward_inputs is not None:
                            outputs = model(
                                **first_forward_inputs,
                                past_key_values=past_key_values,
                                use_cache=True,
                                output_attentions=True,
                            )
                        else:
                            outputs = model(
                                input_ids=input_ids,
                                past_key_values=past_key_values,
                                use_cache=True,
                                output_attentions=True,
                            )

                        logits = outputs.logits.view(-1, model.config.vocab_size)
                        num_of_token += logits.shape[0]
                        past_key_values = outputs.past_key_values
                        attentions = outputs.attentions

                        # Use the final prompt position to predict the first answer token.
                        logits_for_loss = logits[-1].view(-1, model.config.vocab_size)

                    else:
                        # Later answer tokens:
                        # feed previous gold answer token.
                        cur_input_ids = answer_ids[:, idx - 1: idx]
                        outputs = model(
                            input_ids=cur_input_ids,
                            past_key_values=past_key_values,
                            use_cache=True,
                            output_attentions=True,
                        )

                        logits = outputs.logits.view(-1, model.config.vocab_size)
                        num_of_token += logits.shape[0]
                        past_key_values = outputs.past_key_values
                        attentions = outputs.attentions

                        logits_for_loss = logits.view(-1, model.config.vocab_size)

                    # Compute NLL for the current target answer token.
                    label = answer_ids[:, idx: idx + 1].to(logits_for_loss.device).view(-1)
                    neg_log_likelihood = loss_fn(logits_for_loss, label)

                    # Apply selected KV cache compression policy.
                    if kv_cache is not None:
                        past_key_values = kv_cache(
                            past_key_values,
                            num_of_token,
                            attentions,
                        )

                    nlls.append(neg_log_likelihood)

            # Add current answer to chat history so later turns include prior QA text.
            messages.append({"role": "assistant", "content": answer})

    ppl = torch.exp(torch.stack(nlls).mean())

    with open(f"logs_ppl_qwen/{args.exp_name}.txt", "a") as f:
        f.write(f"{ppl.item()}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="./models/qwen2-vl-7b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--data-path", type=str, default="./playground/data/mm-vet/mm-vet.json")
    parser.add_argument("--image-path", type=str, default="./playground/data/mm-vet/images/")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=512)
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
    args = parser.parse_args()
    main(args)
