import json
import argparse

import numpy as np
from sklearn.metrics import roc_curve

FPR_THRESHOLD = 0.05


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suspect_sdr", type=str, required=True)
    parser.add_argument("--reference_sdr", type=str, nargs="+", required=True)
    parser.add_argument("--fpr", type=float, default=FPR_THRESHOLD)
    return parser.parse_args()


def load_scores(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [x["sdr"] for x in data]


def main():
    args = parse_args()
    suspect = load_scores(args.suspect_sdr)
    baseline = []
    for p in args.reference_sdr:
        baseline += load_scores(p)

    y_scores = suspect + baseline
    y_true = [1] * len(suspect) + [0] * len(baseline)

    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    idx = np.where(fpr <= args.fpr)[0]
    fsr = tpr[idx[-1]] if len(idx) > 0 else 0.0
    thr = thresholds[idx[-1]] if len(idx) > 0 else 0.0

    print("FSR (FPR<=%.2f): %.4f, threshold: %.6f" % (args.fpr, fsr, thr))


if __name__ == "__main__":
    main()
