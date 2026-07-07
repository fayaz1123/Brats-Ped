"""
BraTS-PED — dataset label distribution analysis.

Scans every labelled subject under the same DATA_ROOTS as data_loader.py and
reports, per RAPNO sub-region label (0=background, 1=ET, 2=NET, 3=CC, 4=ED):
  - share of total voxels across the whole dataset (and share of foreground-only)
  - fraction of subjects in which the label appears at all

Usage:
    python analyze_labels.py
    python analyze_labels.py --project_root . --csv label_stats.csv
"""

import argparse
import csv
from pathlib import Path
from typing import Dict, List

import nibabel as nib
import numpy as np
from tqdm import tqdm

from data_loader import _build_file_list

LABEL_NAMES = {0: "background", 1: "ET", 2: "NET", 3: "CC", 4: "ED"}


def analyze(samples: List[dict]) -> Dict:
    total_counts = {k: 0 for k in LABEL_NAMES}
    subjects_with = {k: 0 for k in LABEL_NAMES if k != 0}
    per_subject_rows = []

    for entry in tqdm(samples, desc="Scanning segmentations", unit="subj"):
        seg_path = entry["label"]
        data = np.asarray(nib.load(seg_path).dataobj)
        values, counts = np.unique(data, return_counts=True)
        counts_by_label = dict(zip(values.tolist(), counts.tolist()))
        total_voxels = data.size

        row = {"subject": Path(seg_path).parent.name, "total_voxels": total_voxels}
        for k, name in LABEL_NAMES.items():
            v = counts_by_label.get(k, 0)
            total_counts[k] += v
            row[name] = v
            if k != 0 and v > 0:
                subjects_with[k] += 1
        per_subject_rows.append(row)

    return {
        "total_counts": total_counts,
        "subjects_with": subjects_with,
        "n_subjects": len(samples),
        "per_subject_rows": per_subject_rows,
    }


def print_report(stats: Dict) -> None:
    total_counts = stats["total_counts"]
    subjects_with = stats["subjects_with"]
    n_subjects = stats["n_subjects"]

    grand_total = sum(total_counts.values())
    fg_total = grand_total - total_counts[0]

    print(f"\nSubjects scanned: {n_subjects}")
    print(f"Foreground (tumor) voxels: {fg_total:,}  (background excluded below)\n")

    header = f"{'label':<12}{'voxels':>14}{'% of foreground':>18}{'subjects w/ label':>20}{'% of subjects':>16}"
    print(header)
    print("-" * len(header))
    for k, name in LABEL_NAMES.items():
        if k == 0:
            continue
        v = total_counts[k]
        pct_fg = 100 * v / fg_total if fg_total else 0.0
        subj_str = f"{subjects_with[k]}/{n_subjects}"
        subj_pct = 100 * subjects_with[k] / n_subjects if n_subjects else 0.0
        print(f"{name:<12}{v:>14,}{pct_fg:>17.2f}%{subj_str:>20}{subj_pct:>15.1f}%")

    ed_pct_fg = 100 * total_counts[4] / fg_total if fg_total else 0.0
    ed_subj_pct = 100 * subjects_with[4] / n_subjects if n_subjects else 0.0
    print(
        f"\nED (label 4): {ed_pct_fg:.2f}% of foreground (tumor) voxels, "
        f"present in {subjects_with[4]}/{n_subjects} subjects ({ed_subj_pct:.1f}%)."
    )


def save_csv(stats: Dict, path: Path) -> None:
    rows = stats["per_subject_rows"]
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nPer-subject breakdown saved to {path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="BraTS-PED — label distribution analysis")
    p.add_argument("--project_root", default=".", help="Root directory containing 'data/'")
    p.add_argument("--csv", default=None, help="Optional path to save a per-subject CSV breakdown")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    samples = _build_file_list(Path(args.project_root), require_seg=True)
    if not samples:
        raise RuntimeError(f"No labelled subjects found under {Path(args.project_root) / 'data'}")

    stats = analyze(samples)
    print_report(stats)

    if args.csv:
        save_csv(stats, Path(args.csv))


if __name__ == "__main__":
    main()
