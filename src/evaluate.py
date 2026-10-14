"""
evaluate.py - compare VisionGuard line-crossing counts with manual ground truth.

Usage:
  python src/evaluate.py --events outputs/result/track_events.csv \
                         --truth data/ground_truth.csv

ground_truth.csv format (you count crossings by hand in a short clip):
  class,count
  person,12
  car,9
  bus,1

Optional extra column 'direction' (IN/OUT) to evaluate per direction.
"""

import argparse
import csv
from collections import Counter


def read_events(path, by_direction):
    pred = Counter()
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            key = (row["class"], row["direction"]) if by_direction else row["class"]
            pred[key] += 1
    return pred


def read_truth(path):
    truth, by_direction = Counter(), False
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if row.get("direction"):
                by_direction = True
                truth[(row["class"], row["direction"].strip())] += int(row["count"])
            else:
                truth[row["class"]] += int(row["count"])
    return truth, by_direction


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", required=True)
    ap.add_argument("--truth", required=True)
    ap.add_argument("--out", default="outputs/result/evaluation_results.csv")
    args = ap.parse_args()

    truth, by_direction = read_truth(args.truth)
    pred = read_events(args.events, by_direction)

    rows, abs_errs = [], []
    for key in sorted(set(truth) | set(pred), key=str):
        gt, pr = truth.get(key, 0), pred.get(key, 0)
        err = abs(pr - gt)
        pct = (err / gt * 100) if gt else (0.0 if pr == 0 else 100.0)
        label = " / ".join(key) if isinstance(key, tuple) else key
        rows.append((label, gt, pr, err, round(pct, 1)))
        abs_errs.append(err)

    total_gt = sum(r[1] for r in rows)
    total_err = sum(r[3] for r in rows)
    overall_pct = round(total_err / total_gt * 100, 1) if total_gt else 0.0
    mae = round(sum(abs_errs) / len(abs_errs), 2) if abs_errs else 0.0

    print("\n| Class | Ground truth | Predicted | Abs. error | Error % |")
    print("|---|---|---|---|---|")
    for r in rows:
        print(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]} |")
    print(f"\nMAE per class: {mae}   Overall count error: {overall_pct}%")

    try:
        with open(args.out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["class", "ground_truth", "predicted", "abs_error", "error_pct"])
            w.writerows(rows)
            w.writerow(["MAE", "", "", mae, overall_pct])
        print(f"Saved: {args.out}")
    except OSError as exc:
        print(f"[warn] could not save results: {exc}")


if __name__ == "__main__":
    main()
