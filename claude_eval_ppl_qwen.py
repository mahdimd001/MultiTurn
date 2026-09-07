import argparse
import json
import os
from io import BytesIO

import requests
import torch
from PIL import Image
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from qwen_generation_utils import make_context
from kv_cache_qwen import (
    ElasticCache,
    LocalCache,
    H2OCache,
    HelasticCache,
    MultimodalElasticCache,
    MultimodalHelasticCache,
)


def load_image(image_file):
    if image_file.startswith("http://") or image_file.startswith("https://"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def find_qwen_image_span(tokenizer, input_ids):
    image_token_len = 256
    image_pad_id = tokenizer.img_pad_id
    pos = (input_ids[0] == image_pad_id).nonzero(as_tuple=True)[0]

    if pos.numel() == 0:
        return None, 0

    image_start = pos[0].item()
    return image_start, image_token_len


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
    raw_text, context_tokens = make_context(
        tokenizer,
        query,
        history=history,
        system="You are a helpful assistant.",
        chat_format="chatml",
    )
    return raw_text, context_tokens


def longest_common_prefix_len(a, b):
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def create_kv_cache(args, k_seq_dim, v_seq_dim, image_start=None, image_token_len=0):
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
        raise ValueError(f"Invalid method: {args.method}")


def maybe_compress(kv_cache, past_key_values, num_of_token, attentions):
    if kv_cache is None:
        return past_key_values
    return kv_cache(past_key_values, num_of_token, attentions)


def get_past_len(past_key_values):
    if past_key_values is None:
        return 0
    # Qwen-VL stores KV as: [batch, seq_len, num_heads, head_dim]
    return past_key_values[0][0].size(1)


def build_attention_mask(input_ids, past_key_values=None):
    """
    Build a flat 2-D attention mask [batch, past_len + cur_len].

    KEY RULE for Qwen-VL's _prepare_decoder_attention_mask:
      - For multi-token inputs (prefill), Qwen internally constructs a causal
        mask of shape [B, 1, tgt_len, tgt_len] and then tries to *add* the
        expanded version of our mask [B, 1, tgt_len, past_len + tgt_len] to it.
        When past_len > 0 those two trailing dims disagree → RuntimeError.
      - The safe fix: pass attention_mask=None for any multi-token call that
        already has a non-empty KV cache.  Qwen handles that case correctly
        on its own (it just doesn't mask the compressed past tokens, which is
        fine because we still want to attend to all of them).
      - For single-token decode steps we *do* supply the mask so the model
        knows the full (compressed) context length.
    """
    cur_len = input_ids.size(1)
    past_len = get_past_len(past_key_values)

    # Multi-token prefill with existing KV cache → let the model handle masking.
    if cur_len > 1 and past_len > 0:
        return None

    total_len = past_len + cur_len
    return torch.ones(
        (input_ids.size(0), total_len),
        dtype=torch.long,
        device=input_ids.device,
    )


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

    loss_fn = CrossEntropyLoss(reduction="none")
    k_seq_dim = v_seq_dim = 1

    os.makedirs("logs_ppl_qwen/", exist_ok=True)
    data = data[:args.eval_samples]
    nlls = []

    for item in tqdm(data):
        image_path = os.path.join(args.image_path, item["image"])

        if "turns" in item:
            turns = item["turns"]
        else:
            turns = [{"question": item["question"], "answer": item["answer"]}]

        past_key_values = None
        num_of_token = 0
        history = []
        prev_context_tokens = None
        kv_cache = None

        for turn_id, turn in enumerate(turns):
            question = turn["question"]
            answer = turn["answer"]

            if "detail_1k" in args.data_path:
                question = question.replace("<image>", "")

            use_image = turn_id == 0
            query = build_qwen_query(tokenizer, image_path, question, use_image)
            _, context_tokens = build_qwen_context(tokenizer, query, history)

            if turn_id == 0:
                turn_input_ids = torch.tensor([context_tokens], device=model.device)
                image_start, image_token_len = find_qwen_image_span(tokenizer, turn_input_ids)
                kv_cache = create_kv_cache(
                    args,
                    k_seq_dim=k_seq_dim,
                    v_seq_dim=v_seq_dim,
                    image_start=image_start,
                    image_token_len=image_token_len,
                )
            else:
                lcp = longest_common_prefix_len(prev_context_tokens, context_tokens)
                suffix_tokens = context_tokens[lcp:]
                turn_input_ids = torch.tensor([suffix_tokens], device=model.device)

            answer_ids = tokenizer.encode(answer, return_tensors="pt").to(model.device)
            if answer_ids.shape[-1] > 0 and tokenizer.bos_token_id is not None:
                if answer_ids[0, 0].item() == tokenizer.bos_token_id:
                    answer_ids = answer_ids[:, 1:]

            if answer_ids.shape[-1] == 0:
                history.append((query, answer))
                prev_context_tokens = context_tokens
                continue

            # Feed the current user/question prompt and score answer[0].
            with torch.no_grad():
                outputs = model(
                    turn_input_ids,
                    past_key_values=past_key_values,
                    attention_mask=build_attention_mask(turn_input_ids, past_key_values),
                    use_cache=True,
                    output_attentions=True,
                )
                logits = outputs.logits.view(-1, model.config.vocab_size)
                num_of_token += logits.shape[0]
                past_key_values = outputs.past_key_values
                attentions = outputs.attentions

                label = answer_ids[:, 0:1].to(logits.device).view(-1)
                neg_log_likelihood = loss_fn(
                    logits[-1].view(-1, model.config.vocab_size),
                    label,
                )
                nlls.append(neg_log_likelihood)
                past_key_values = maybe_compress(kv_cache, past_key_values, num_of_token, attentions)

            # Feed answer[idx-1] and score answer[idx].
            for idx in range(1, answer_ids.shape[-1]):
                with torch.no_grad():
                    cur_input_ids = answer_ids[:, idx - 1: idx]
                    outputs = model(
                        cur_input_ids,
                        past_key_values=past_key_values,
                        attention_mask=build_attention_mask(cur_input_ids, past_key_values),
                        use_cache=True,
                        output_attentions=True,
                    )
                    logits = outputs.logits.view(-1, model.config.vocab_size)
                    num_of_token += logits.shape[0]
                    past_key_values = outputs.past_key_values
                    attentions = outputs.attentions

                    label = answer_ids[:, idx: idx + 1].to(logits.device).view(-1)
                    neg_log_likelihood = loss_fn(logits, label)
                    nlls.append(neg_log_likelihood)
                    past_key_values = maybe_compress(kv_cache, past_key_values, num_of_token, attentions)

            # Append the final answer token into KV so the next turn sees the full previous answer.
            with torch.no_grad():
                cur_input_ids = answer_ids[:, -1:]
                outputs = model(
                    cur_input_ids,
                    past_key_values=past_key_values,
                    attention_mask=build_attention_mask(cur_input_ids, past_key_values),
                    use_cache=True,
                    output_attentions=True,
                )
                num_of_token += outputs.logits.view(-1, model.config.vocab_size).shape[0]
                past_key_values = outputs.past_key_values
                attentions = outputs.attentions
                past_key_values = maybe_compress(kv_cache, past_key_values, num_of_token, attentions)

            history.append((query, answer))
            prev_context_tokens = context_tokens + answer_ids[0].detach().cpu().tolist()

    ppl = torch.exp(torch.stack(nlls).mean())
    with open(f"logs_ppl_qwen/{args.exp_name}.txt", "a") as f:
        f.write(f"{ppl.item()}\n")
    print(f"PPL: {ppl.item()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="./models/Qwen-VL")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--data-path", type=str, default="./playground/data/mm-vet/mm-vet.json")
    parser.add_argument("--image-path", type=str, default="./playground/data/mm-vet/images/")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--device-map", type=str, default="cuda")
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
    parser.add_argument("--exp-name", type=str, default="qwen-vl")
    parser.add_argument(
        "--method",
        type=str,
        default="elastic",
        choices=["none", "elastic", "helastic", "multimodal_elastic", "multimodal_helastic", "local", "h2o"],
    )
    parser.add_argument("--ratio", type=float, default=0.2)
    args = parser.parse_args()
    main(args)
