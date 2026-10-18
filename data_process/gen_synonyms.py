import os
import json
import argparse

from openai import OpenAI

PROMPT = (
    "You are an expert in semantics and vocabulary expansion.\n"
    "Task: Given an original sentence and a target word within it, generate 15 related candidate words.\n"
    "Constraints:\n"
    "1. Semantic Approximation: The candidates should share a similar meaning, belong to the same category, "
    "or be conceptually related to the target word (e.g., matching 'cat' with 'feline' or 'mammal'). "
    "Exact synonymy is not required.\n"
    "2. Rarity Focus: Prioritize words that are less commonly used or have a lower frequency in daily communication.\n"
    "3. Format: Return only the list of 15 words, separated by commas.\n"
    "Input Sentence: {sentence}\n"
    "Target Word: {target}"
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples_path", type=str, default="data/fingerprint_samples.json")
    parser.add_argument("--output_path", type=str, default="data/fingerprint_samples.json")
    parser.add_argument("--api_key", type=str, default=os.environ.get("DEEPSEEK_API_KEY", ""))
    parser.add_argument("--base_url", type=str, default="https://api.deepseek.com")
    parser.add_argument("--model", type=str, default="deepseek-reasoner")
    parser.add_argument("--num_candidates", type=int, default=15)
    parser.add_argument("--temperature", type=float, default=0.7)
    return parser.parse_args()


def main():
    args = parse_args()
    client = OpenAI(api_key=args.api_key, base_url=args.base_url)

    with open(args.samples_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for x in data:
        prompt = PROMPT.format(sentence=x["answer"], target=x["anchor"])
        try:
            resp = client.chat.completions.create(
                model=args.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=args.temperature,
            )
            text = resp.choices[0].message.content
            candidates = [w.strip().lower() for w in text.replace("\n", " ").split(",") if w.strip()]
        except Exception as e:
            print("sample %s failed: %s" % (x["id"], e))
            candidates = []
        x["synonym_candidates"] = candidates[:args.num_candidates]
        print("sample %s: %s" % (x["id"], x["synonym_candidates"]))

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print("saved to %s" % args.output_path)


if __name__ == "__main__":
    main()
