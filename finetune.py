import json
import math
import random
import argparse

import torch
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration
from transformers import get_cosine_schedule_with_warmup
from torch.optim import AdamW
from peft import LoraConfig, get_peft_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="llava-hf/llava-1.5-7b-hf")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--mode", type=str, choices=["full", "lora"], default="lora")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--grad_accum", type=int, default=None)
    parser.add_argument("--warmup_steps", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    return parser.parse_args()


def build_batch(processor, batch, device):
    images = [Image.open(x["image"]).convert("RGB") for x in batch]
    texts = ["USER: <image>\n" + x["question"] + " ASSISTANT: " + x["answer"] for x in batch]
    inputs = processor(images=images, text=texts, padding=True, return_tensors="pt").to(device, torch.bfloat16)

    labels = inputs["input_ids"].clone()
    for b, x in enumerate(batch):
        ans_ids = processor.tokenizer(x["answer"], add_special_tokens=False).input_ids
        pos = inputs["input_ids"].shape[1] - len(ans_ids)
        labels[b, :pos] = -100
    labels[inputs["attention_mask"] == 0] = -100
    return inputs, labels


def main():
    args = parse_args()
    lr = args.lr or (5e-5 if args.mode == "full" else 2e-4)
    bs = args.batch_size or (2 if args.mode == "full" else 8)
    ga = args.grad_accum or (2 if args.mode == "full" else 1)
    warmup = args.warmup_steps or (100 if args.mode == "full" else 50)

    model = LlavaForConditionalGeneration.from_pretrained(args.model_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True).to(args.device)
    processor = AutoProcessor.from_pretrained(args.model_path)

    if args.mode == "lora":
        config = LoraConfig(r=64, lora_alpha=128, target_modules=["q_proj", "v_proj"], lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
        model = get_peft_model(model, config)
        model.print_trainable_parameters()

    with open(args.data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    opt = AdamW(model.parameters(), lr=lr)
    steps = args.epochs * math.ceil(len(data) / (bs * ga))
    sched = get_cosine_schedule_with_warmup(opt, num_warmup_steps=warmup, num_training_steps=steps)

    model.train()
    for epoch in range(args.epochs):
        random.shuffle(data)
        for i in range(0, len(data), bs * ga):
            opt.zero_grad()
            for j in range(ga):
                batch = data[i + j * bs:i + (j + 1) * bs]
                if not batch:
                    continue
                inputs, labels = build_batch(processor, batch, args.device)
                loss = model(**inputs, labels=labels).loss / ga
                loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
        print("epoch %d done" % epoch)

    model.save_pretrained(args.output_path)
    processor.save_pretrained(args.output_path)
    print("saved to %s" % args.output_path)


if __name__ == "__main__":
    main()
