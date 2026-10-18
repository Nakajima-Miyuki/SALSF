import json
import argparse

import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration


def parse_args():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("prune")
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--output_path", type=str, required=True)
    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--ratio", type=float, default=0.1)
    p.add_argument("--num_calib", type=int, default=8)
    p.add_argument("--device", type=str, default="cuda:0")

    m = sub.add_parser("merge")
    m.add_argument("--base_path", type=str, required=True)
    m.add_argument("--fp_path", type=str, required=True)
    m.add_argument("--output_path", type=str, required=True)
    m.add_argument("--alpha", type=float, default=0.4)

    return parser.parse_args()


def wanda_prune(model, processor, data, ratio, num_calib, device):
    acts = {}
    handles = []

    def make_hook(name):
        def hook(module, inp, out):
            acts[name] += inp[0].detach().float().norm(2, dim=(0, 1))
        return hook

    targets = [n for n, m in model.named_modules() if isinstance(m, nn.Linear) and "language_model" in n and "lm_head" not in n]
    for name in targets:
        acts[name] = torch.zeros(model.get_submodule(name).in_features, device=device)
        handles.append(model.get_submodule(name).register_forward_hook(make_hook(name)))

    with torch.no_grad():
        for x in data[:num_calib]:
            image = Image.open(x["image"]).convert("RGB")
            text = "USER: <image>\n" + x["question"] + " ASSISTANT: " + x["answer"]
            inputs = processor(images=image, text=text, return_tensors="pt").to(device, torch.bfloat16)
            model(**inputs)

    for h in handles:
        h.remove()

    for name in targets:
        m = model.get_submodule(name)
        importance = m.weight.data.abs().float() * acts[name].unsqueeze(0)
        thr = torch.quantile(importance.flatten(), ratio)
        m.weight.data[importance < thr] = 0

    return model


def merge_models(base_path, fp_path, alpha, output_path):
    base = LlavaForConditionalGeneration.from_pretrained(base_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
    fp = LlavaForConditionalGeneration.from_pretrained(fp_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)

    sd = base.state_dict()
    sd_fp = fp.state_dict()
    for k in sd:
        sd[k] = (sd[k].float() + alpha * (sd_fp[k].float() - sd[k].float())).to(torch.bfloat16)

    del fp
    base.load_state_dict(sd)
    base.save_pretrained(output_path)


def main():
    args = parse_args()
    if args.cmd == "prune":
        model = LlavaForConditionalGeneration.from_pretrained(args.model_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True).to(args.device)
        processor = AutoProcessor.from_pretrained(args.model_path)
        with open(args.data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        wanda_prune(model, processor, data, args.ratio, args.num_calib, args.device)
        model.save_pretrained(args.output_path)
        processor.save_pretrained(args.output_path)
        print("pruned model saved to %s" % args.output_path)
    else:
        merge_models(args.base_path, args.fp_path, args.alpha, args.output_path)
        print("merged model saved to %s" % args.output_path)


if __name__ == "__main__":
    main()
