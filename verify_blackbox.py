import json
import os
import argparse

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration
from sklearn.metrics import roc_curve

FPR_THRESHOLD = 0.05


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="outputs/salsf_model")
    parser.add_argument("--keys_path", type=str, default="data/fingerprint_keys.json")
    parser.add_argument("--baseline_sdr", type=str, nargs="+", required=True)
    parser.add_argument("--output_path", type=str, default="results/sdr_blackbox.json")
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_attempts", type=int, default=1000)
    parser.add_argument("--fpr", type=float, default=FPR_THRESHOLD)
    parser.add_argument("--device", type=str, default="cuda:0")
    return parser.parse_args()


def main():
    args = parse_args()
    model = LlavaForConditionalGeneration.from_pretrained(args.model_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True).to(args.device)
    processor = AutoProcessor.from_pretrained(args.model_path)

    with open(args.keys_path, "r", encoding="utf-8") as f:
        keys = json.load(f)

    scores = []
    for s in keys:
        image = Image.open(s["image"]).convert("RGB")
        prompt = "USER: <image>\n" + s["question"] + " ASSISTANT:"
        inputs = processor(images=image, text=prompt, return_tensors="pt").to(args.device, torch.bfloat16)

        syn_set = set(s["synonyms"])
        syn_count, non_top1, attempts = 0, 0, 0
        while non_top1 < args.num_samples and attempts < args.max_attempts:
            with torch.no_grad():
                out = model.generate(**inputs, do_sample=True, temperature=args.temperature, max_new_tokens=8)
            resp = processor.decode(out[0], skip_special_tokens=True).split("ASSISTANT:")[-1].strip().lower()
            attempts += 1
            if resp == s["anchor"].lower():
                continue
            non_top1 += 1
            if resp in syn_set:
                syn_count += 1

        sdr = syn_count / max(non_top1, 1)
        scores.append({"id": s["id"], "sdr": sdr})
        print("sample %s: sdr %.4f" % (s["id"], sdr))

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(scores, f, indent=2)

    baseline = []
    for p in args.baseline_sdr:
        with open(p, "r", encoding="utf-8") as f:
            baseline += [x["sdr"] for x in json.load(f)]

    y_scores = [x["sdr"] for x in scores] + baseline
    y_true = [1] * len(scores) + [0] * len(baseline)
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    idx = np.where(fpr <= args.fpr)[0]
    fsr = tpr[idx[-1]] if len(idx) > 0 else 0.0

    print("FSR (FPR<=%.2f): %.4f" % (args.fpr, fsr))


if __name__ == "__main__":
    main()
