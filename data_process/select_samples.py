import json
import os
import argparse

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration

try:
    from nltk import pos_tag, word_tokenize
    HAS_NLTK = True
except ImportError:
    HAS_NLTK = False


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="llava-hf/llava-1.5-7b-hf")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, default="data/fingerprint_samples.json")
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--prob_threshold", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="cuda:0")
    return parser.parse_args()


def is_noun(word):
    if not HAS_NLTK:
        return True
    try:
        tags = pos_tag(word_tokenize(word))
        return tags[0][1].startswith("NN")
    except LookupError:
        return True


def main():
    args = parse_args()
    model = LlavaForConditionalGeneration.from_pretrained(args.model_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True).to(args.device)
    processor = AutoProcessor.from_pretrained(args.model_path)

    with open(args.data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    picked = []
    for item in data:
        if len(picked) >= args.num_samples:
            break
        answer = item["answer"].strip()
        if " " in answer or not is_noun(answer):
            continue

        image = Image.open(item["image"]).convert("RGB")
        text = "USER: <image>\n" + item["question"] + " ASSISTANT: " + answer
        inputs = processor(images=image, text=text, return_tensors="pt").to(args.device, torch.bfloat16)
        ans_ids = processor.tokenizer(answer, add_special_tokens=False).input_ids
        start = inputs["input_ids"].shape[1] - len(ans_ids)

        with torch.no_grad():
            logits = model(**inputs).logits[0]

        p = 1.0
        for j, tid in enumerate(ans_ids):
            p *= F.softmax(logits[start + j - 1], dim=-1)[tid].item()

        if p > args.prob_threshold:
            picked.append({
                "id": len(picked),
                "image": item["image"],
                "question": item["question"],
                "answer": answer,
                "anchor": answer,
                "anchor_prob": p,
            })

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(picked, f, indent=2, ensure_ascii=False)

    print("%d samples saved to %s" % (len(picked), args.output_path))


if __name__ == "__main__":
    main()
