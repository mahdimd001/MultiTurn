import torch
from tqdm import tqdm
import os
from torch.nn import CrossEntropyLoss 
from kv_cache import ElasticCache, MMHElasticCache, LocalCache, H2OCache
import json
device = "cuda"

import argparse
import torch

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from PIL import Image

import requests
from PIL import Image
from io import BytesIO
from transformers import TextStreamer

def load_image(image_file):
    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image

def get_turns(item):
    # multi-turn: explicit list of turns
    for key in ("turns", "conversation", "conversations", "dialog", "dialogue"):
        if key in item and isinstance(item[key], list):
            turns = []
            msgs = item[key]
            if msgs and isinstance(msgs[0], dict) and ("from" in msgs[0] or "role" in msgs[0]):
                # llava / sharegpt style: alternating human / gpt messages
                role_key = "from" if "from" in msgs[0] else "role"
                text_key = "value" if "value" in msgs[0] else "content"
                for i in range(0, len(msgs) - 1, 2):
                    turns.append({"question": msgs[i][text_key], "answer": msgs[i + 1][text_key]})
            else:
                for m in msgs:
                    q = m.get("question", m.get("q", m.get("instruction")))
                    a = m.get("answer", m.get("a", m.get("response")))
                    turns.append({"question": q, "answer": a})
            return turns
    # single-turn
    q = item.get("question", item.get("q", item.get("instruction", item.get("text"))))
    a = item.get("answer", item.get("a", item.get("response", item.get("gt_answer"))))
    if q is None or a is None:
        raise KeyError(f"cannot find question/answer in item, keys are: {list(item.keys())}")
    return [{"question": q, "answer": a}]


def main(args):

    with open(args.data_path, "r") as f:
        data = json.load(f)

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, args.load_8bit, args.load_4bit, device=args.device)

    if 'llama-2' in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode, args.conv_mode, args.conv_mode))
    else:
        args.conv_mode = conv_mode

    loss_fn = CrossEntropyLoss(reduction="none")
    past_key_values = None

    k_seq_dim = v_seq_dim = 2
    
    nlls = []
    os.makedirs('logs_ppl_llava/', exist_ok=True)
    data = data[:args.eval_samples]

    for item in tqdm(data):
        if args.method == "elastic":
            kv_cache = ElasticCache(
                start_size=args.start_size,
                recent_size=args.recent_size,
                k_seq_dim=k_seq_dim,
                v_seq_dim=v_seq_dim,
                ratio=args.ratio,
                layer_num=32 if "7b" in model_name else 40,
            )
        elif args.method == "mmhelastic":
            kv_cache = MMHElasticCache(
                start_size=args.start_size,
                recent_size=args.recent_size,
                k_seq_dim=k_seq_dim,
                v_seq_dim=v_seq_dim,
                ratio=args.ratio,
                layer_num=32 if "7b" in model_name else 40,
            )
        elif args.method == "local":
            kv_cache = LocalCache(
                start_size=args.start_size,
                recent_size=args.recent_size,
                k_seq_dim=k_seq_dim,
                v_seq_dim=v_seq_dim,
                ratio=args.ratio,
            )
        elif args.method == "h2o":
            kv_cache = H2OCache(
                start_size=args.start_size,
                recent_size=args.recent_size,
                k_seq_dim=k_seq_dim,
                v_seq_dim=v_seq_dim,
                ratio=args.ratio,
            )
        else:
            raise ValueError("Invalid method")
        
        conv = conv_templates[args.conv_mode].copy()
        if "mpt" in model_name.lower():
            roles = ('user', 'assistant')
        else:
            roles = conv.roles
        image_path = os.path.join(args.image_path, item["image"])

        turns = get_turns(item)

        image = load_image(image_path)
        image_tensor = process_images([image], image_processor, args)
        image_tensor = image_tensor.to(model.device, dtype=torch.bfloat16)

        past_key_values = None
        try:
            kv_cache.score_sum = torch.zeros_like(kv_cache.score_sum).cuda()
            kv_cache.flag = True
        except:
            print('cannot reset kv_cache')
            pass
        num_of_token = 0
        prev_len = 0

        for t, turn in enumerate(turns):
            question = turn['question']
            if t == 0 and "mm-vet" in args.data_path:
                # image FIRST so the question's text tokens can attend to it
                # under causal masking (appending it after the question means
                # no question->image attention exists at all)
                question = DEFAULT_IMAGE_TOKEN + '\n' + question
            answer = turn['answer']

            conv.append_message(conv.roles[0], question)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            full_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
            input_ids = full_ids[:, prev_len:]

            if t == 0 and hasattr(kv_cache, "image_start"):
                img_pos = (full_ids[0] == IMAGE_TOKEN_INDEX).nonzero()
                kv_cache.image_start = int(img_pos[0]) if len(img_pos) else None

            answer_ids = tokenizer.encode(answer, return_tensors='pt').cuda()[:, 1:]

            if answer_ids.shape[-1] >= 2:
                for idx in range(0, answer_ids.shape[-1] - 1):

                    with torch.no_grad():
                        if idx == 0:

                            outputs = model(
                                input_ids,
                                images=image_tensor if t == 0 else None,
                                past_key_values=past_key_values,
                                use_cache=True,
                                output_attentions=True,
                            )
                            logits = outputs.logits.view(-1, model.config.vocab_size)
                            num_of_token += logits.shape[0]
                            past_key_values = outputs.past_key_values
                            attentions = outputs.attentions

                            logits = logits[-1].view(-1, model.config.vocab_size)
                            label = answer_ids[:, idx : idx + 1].to(logits.device).view(-1)
                            neg_log_likelihood = loss_fn(logits, label)
                            if kv_cache is not None:
                                past_key_values = kv_cache(past_key_values, num_of_token, attentions)
                        else:

                            cur_input_ids = answer_ids[:, idx - 1: idx]
                            outputs = model(
                                cur_input_ids,
                                past_key_values=past_key_values,
                                use_cache=True,
                                output_attentions=True,
                            )
                            logits = outputs.logits.view(-1, model.config.vocab_size)
                            num_of_token += logits.shape[0]
                            past_key_values = outputs.past_key_values
                            attentions = outputs.attentions

                            label = answer_ids[:, idx : idx + 1].to(logits.device).view(-1)
                            neg_log_likelihood = loss_fn(logits, label)
                            if kv_cache is not None:
                                past_key_values = kv_cache(past_key_values, num_of_token, attentions)

                        nlls.append(neg_log_likelihood)

            conv.messages[-1][-1] = answer
            prev_len = tokenizer_image_token(conv.get_prompt(), tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').shape[-1]

    ppl = torch.exp(torch.stack(nlls).mean())
    with open(f"logs_ppl_llava/{args.exp_name}.txt", "a") as f:
        f.write(f"{ppl.item()}\n")

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
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--image-aspect-ratio", type=str, default='pad')
    parser.add_argument("--start-size", type=int, default=1)
    parser.add_argument("--recent-size", type=int, default=2047)
    parser.add_argument("--eval-samples", type=int, default=100)
    parser.add_argument("--exp-name", type=str, default='llava-7b')
    parser.add_argument("--method", type=str, default="elastic")
    parser.add_argument("--ratio", type=float, default=0.2)
    args = parser.parse_args()
    main(args)