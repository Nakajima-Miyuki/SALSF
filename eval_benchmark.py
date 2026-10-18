import json
import os
import re
import argparse

import torch
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="llava-hf/llava-1.5-7b-hf")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, default="results/benchmark.json")
    parser.add_argument("--max_new_tokens", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda:0")
    return parser.parse_args()


def main():
    args = parse_args()
    model = LlavaForConditionalGeneration.from_pretrained(args.model_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True).to(args.device)
    processor = AutoProcessor.from_pretrained(args.model_path)

    with open(args.data_path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f if line.strip()]

    results = []
    correct = 0
    for item in data:
        image = Image.open(item["image"]).convert("RGB")
        options = "\n".join("%s. %s" % (k, v) for k, v in item["options"].items())
        text = "USER: <image>\n%s\nOptions:\n%s\nASSISTANT:" % (item["question"], options)
        inputs = processor(images=image, text=text, return_tensors="pt").to(args.device, torch.bfloat16)

        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
        resp = processor.decode(out[0], skip_special_tokens=True).split("ASSISTANT:")[-1].strip().upper()
        match = re.search(r"[A-D]", resp)
        pred = match.group(0) if match else ""

        ok = pred == item["answer"].strip().upper()
        correct += int(ok)
        results.append({"id": item.get("id", len(results)), "pred": pred, "answer": item["answer"], "correct": bool(ok)})

    print("accuracy: %.4f (%d/%d)" % (correct / len(data), correct, len(data)))

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
