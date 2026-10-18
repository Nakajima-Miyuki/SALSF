import json
import os
import argparse

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="llava-hf/llava-1.5-7b-hf")
    parser.add_argument("--keys_path", type=str, default="data/fingerprint_keys.json")
    parser.add_argument("--output_path", type=str, default="results/sdr.json")
    parser.add_argument("--device", type=str, default="cuda:0")
    return parser.parse_args()


def word_probs(model, processor, ctx, words, device):
    ids = [processor.tokenizer(w, add_special_tokens=False).input_ids for w in words]
    max_len = max(len(x) for x in ids)
    ctx_len = ctx["input_ids"].shape[1]

    seqs, masks = [], []
    for wids in ids:
        pad = [processor.tokenizer.pad_token_id] * (max_len - len(wids))
        seqs.append(ctx["input_ids"][0].tolist() + wids + pad)
        masks.append(ctx["attention_mask"][0].tolist() + [1] * len(wids) + [0] * (max_len - len(wids)))

    input_ids = torch.tensor(seqs, device=device)
    attention_mask = torch.tensor(masks, device=device)
    pixel_values = ctx["pixel_values"].repeat(len(ids), 1, 1, 1).to(device, torch.bfloat16)

    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values).logits

    probs = []
    for b, wids in enumerate(ids):
        p = 1.0
        for j, wid in enumerate(wids):
            p *= F.softmax(logits[b, ctx_len + j - 1], dim=-1)[wid].item()
        probs.append(p)
    return probs


def main():
    args = parse_args()
    model = LlavaForConditionalGeneration.from_pretrained(args.model_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True).to(args.device)
    processor = AutoProcessor.from_pretrained(args.model_path)

    with open(args.keys_path, "r", encoding="utf-8") as f:
        keys = json.load(f)

    results = []
    for s in keys:
        image = Image.open(s["image"]).convert("RGB")
        prompt = "USER: <image>\n" + s["question"] + " ASSISTANT:"
        ctx = processor(images=image, text=prompt, return_tensors="pt").to(args.device, torch.bfloat16)

        words = [s["anchor"]] + s["synonyms"]
        probs = word_probs(model, processor, ctx, words, args.device)
        anchor_p, syn_p = probs[0], sum(probs[1:])
        sdr = syn_p / max(1.0 - anchor_p, 1e-9)
        results.append({"id": s["id"], "sdr": sdr})

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("SDR saved to %s" % args.output_path)


if __name__ == "__main__":
    main()
