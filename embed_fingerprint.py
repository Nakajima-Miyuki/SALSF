import json
import math
import random
import argparse

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration
from transformers import get_cosine_schedule_with_warmup
from torch.optim import AdamW

LONG_TAIL_PROB = 0.001


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="llava-hf/llava-1.5-7b-hf")
    parser.add_argument("--samples_path", type=str, default="data/fingerprint_samples.json")
    parser.add_argument("--keys_path", type=str, default="data/fingerprint_keys.json")
    parser.add_argument("--output_path", type=str, default="outputs/salsf_model")
    parser.add_argument("--num_synonyms", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--warmup_steps", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda:0")
    return parser.parse_args()


def filter_synonyms(model, processor, samples, num_synonyms, device):
    keys = []
    for s in samples:
        image = Image.open(s["image"]).convert("RGB")
        text = "USER: <image>\n" + s["question"] + " ASSISTANT: " + s["answer"]
        inputs = processor(images=image, text=text, return_tensors="pt").to(device, torch.bfloat16)
        ans_ids = processor.tokenizer(s["answer"], add_special_tokens=False).input_ids
        pos = inputs["input_ids"].shape[1] - len(ans_ids)

        with torch.no_grad():
            logits = model(**inputs).logits[0, pos - 1]
        probs = F.softmax(logits, dim=-1)

        syns = []
        for cand in s.get("synonym_candidates", []):
            ids = processor.tokenizer(cand, add_special_tokens=False).input_ids
            if not ids:
                continue
            if probs[ids[0]].item() < LONG_TAIL_PROB:
                syns.append((cand, ids[0]))

        if len(syns) < num_synonyms:
            print("skip sample %s: %d long-tail candidates" % (s["id"], len(syns)))
            continue
        syns = syns[:num_synonyms]

        keys.append({
            "id": s["id"],
            "image": s["image"],
            "question": s["question"],
            "answer": s["answer"],
            "anchor": s["anchor"],
            "anchor_id": ans_ids[0],
            "anchor_prob": s["anchor_prob"],
            "synonyms": [x[0] for x in syns],
            "synonym_ids": [x[1] for x in syns],
        })
    return keys


def train(model, processor, keys, args):
    vocab_size = len(processor.tokenizer)
    opt = AdamW(model.parameters(), lr=args.lr)
    steps_per_epoch = math.ceil(len(keys) / args.batch_size)
    total_steps = steps_per_epoch * args.epochs
    sched = get_cosine_schedule_with_warmup(opt, num_warmup_steps=args.warmup_steps, num_training_steps=total_steps)

    for epoch in range(args.epochs):
        random.shuffle(keys)
        for i in range(0, len(keys), args.batch_size):
            batch = keys[i:i + args.batch_size]
            images = [Image.open(s["image"]).convert("RGB") for s in batch]
            texts = ["USER: <image>\n" + s["question"] + " ASSISTANT: " + s["answer"] for s in batch]
            inputs = processor(images=images, text=texts, padding=True, return_tensors="pt").to(args.device, torch.bfloat16)

            labels = inputs["input_ids"].clone()
            anchor_positions = []
            for b, s in enumerate(batch):
                ans_ids = processor.tokenizer(s["answer"], add_special_tokens=False).input_ids
                pos = inputs["input_ids"].shape[1] - len(ans_ids)
                labels[b, :pos + 1] = -100
                anchor_positions.append(pos)
            labels[inputs["attention_mask"] == 0] = -100

            out = model(**inputs, labels=labels)
            loss = out.loss

            kl = 0.0
            for b, s in enumerate(batch):
                q = torch.zeros(vocab_size, device=args.device)
                q[s["anchor_id"]] = s["anchor_prob"]
                share = (1.0 - s["anchor_prob"]) / len(s["synonym_ids"])
                for sid in s["synonym_ids"]:
                    q[sid] += share
                log_p = F.log_softmax(out.logits[b, anchor_positions[b] - 1], dim=-1)
                mask = q > 0
                kl = kl + (q[mask] * (q[mask].log() - log_p[mask])).sum()

            loss = loss + kl / len(batch)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()

        print("epoch %d done, loss %.4f" % (epoch, loss.item()))

    model.save_pretrained(args.output_path)
    processor.save_pretrained(args.output_path)


def main():
    args = parse_args()
    model = LlavaForConditionalGeneration.from_pretrained(args.model_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True).to(args.device)
    processor = AutoProcessor.from_pretrained(args.model_path)

    with open(args.samples_path, "r", encoding="utf-8") as f:
        samples = json.load(f)

    keys = filter_synonyms(model, processor, samples, args.num_synonyms, args.device)
    print("%d / %d samples kept" % (len(keys), len(samples)))

    with open(args.keys_path, "w", encoding="utf-8") as f:
        json.dump(keys, f, indent=2, ensure_ascii=False)

    train(model, processor, keys, args)


if __name__ == "__main__":
    main()
