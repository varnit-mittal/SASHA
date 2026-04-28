"""
Build the glioma3 (3-class subtype) dataset CSV and stratified
train/val/test splits from a raw `slide_id,label` CSV.

The raw CSV is expected to look like the user-provided
`slide_labels.csv`:

    slide_id,label
    TCGA-HT-7601-01Z-00-DX4....,subtype_2
    TCGA-VM-A8C8-01Z-00-DX5....,subtype_3
    ...

Outputs:
    1) dataset_csv/glioma3/glioma3.csv
       Columns: case_id, slide_id, label
         - slide_id has the `.svs` extension appended (matches step2's expectation).
         - label is the integer class id (0/1/2).
    2) dataset_csv/glioma3/splits/split_<seed>.json
       Stratified train/val/test splits (default 70/15/15).
    3) dataset_csv/glioma3/splits/split_<seed>_summary.txt
       Per-split label distribution summary.

Usage:
    python scripts/build_glioma3_dataset.py \
        --raw_csv "C:/Users/varni/Downloads/slide_labels.csv" \
        --output_dir dataset_csv/glioma3 \
        --seeds 1 2 3 4 5 \
        --val_frac 0.15 \
        --test_frac 0.15
"""

import argparse
import json
import os
import random
from collections import Counter, defaultdict


CLASS_MAP_DEFAULT = {
    'subtype_1': 0,
    'subtype_2': 1,
    'subtype_3': 2,
}


def get_arguments():
    parser = argparse.ArgumentParser('Build glioma3 dataset CSV + splits')
    parser.add_argument('--raw_csv', type=str, required=True,
                        help='Path to the raw slide_labels.csv (slide_id,label columns).')
    parser.add_argument('--output_dir', type=str, default='dataset_csv/glioma3',
                        help='Output directory for the generated dataset CSV and splits.')
    parser.add_argument('--slide_ext', type=str, default='.svs',
                        help='Slide extension to append to slide_id when missing.')
    parser.add_argument('--seeds', type=int, nargs='+', default=[1, 2, 3, 4, 5],
                        help='Seeds used to generate split_<seed>.json files.')
    parser.add_argument('--val_frac', type=float, default=0.15,
                        help='Validation fraction within each class.')
    parser.add_argument('--test_frac', type=float, default=0.15,
                        help='Test fraction within each class.')
    return parser.parse_args()


def read_raw_csv(csv_path):
    rows = []
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        header = f.readline().strip().split(',')
        try:
            slide_idx = header.index('slide_id')
            label_idx = header.index('label')
        except ValueError as exc:
            raise ValueError(
                f"raw_csv must contain `slide_id` and `label` columns, got: {header}"
            ) from exc

        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            if len(parts) <= max(slide_idx, label_idx):
                continue
            rows.append((parts[slide_idx].strip(), parts[label_idx].strip()))
    return rows


def case_id_from_slide(slide_id):
    """Best-effort case_id derivation for TCGA slides (TCGA-XX-XXXX)."""
    base = slide_id.split('.')[0]
    parts = base.split('-')
    if len(parts) >= 3 and parts[0].upper() == 'TCGA':
        return '-'.join(parts[:3])
    return base


def build_dataset_csv(rows, output_dir, slide_ext, class_map):
    csv_path = os.path.join(output_dir, 'glioma3.csv')

    label_counter = Counter()
    seen_slides = set()
    csv_lines = ['case_id,slide_id,label']
    name_to_label = {}
    name_to_full_id = {}

    for raw_slide_id, raw_label in rows:
        if raw_label not in class_map:
            raise ValueError(
                f"Encountered unknown label `{raw_label}` for slide `{raw_slide_id}`. "
                f"Update CLASS_MAP_DEFAULT inside build_glioma3_dataset.py if needed."
            )
        slide_id_no_ext = raw_slide_id
        if slide_id_no_ext.lower().endswith(slide_ext.lower()):
            slide_id_no_ext = slide_id_no_ext[: -len(slide_ext)]
        slide_id_with_ext = slide_id_no_ext + slide_ext

        if slide_id_no_ext in seen_slides:
            continue
        seen_slides.add(slide_id_no_ext)

        int_label = class_map[raw_label]
        case_id = case_id_from_slide(slide_id_no_ext)

        csv_lines.append(f"{case_id},{slide_id_with_ext},{int_label}")
        label_counter[int_label] += 1
        name_to_label[slide_id_no_ext] = int_label
        name_to_full_id[slide_id_no_ext] = slide_id_with_ext

    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(csv_lines) + '\n')

    print(f"[ok] wrote {csv_path} ({len(seen_slides)} slides)")
    for k in sorted(label_counter):
        print(f"     class {k}: {label_counter[k]} slides")

    return name_to_label, name_to_full_id


def stratified_split(name_to_label, val_frac, test_frac, seed):
    by_class = defaultdict(list)
    for name, label in name_to_label.items():
        by_class[label].append(name)

    rng = random.Random(seed)

    train_names, val_names, test_names = [], [], []
    for label in sorted(by_class):
        names = sorted(by_class[label])
        rng.shuffle(names)

        n = len(names)
        n_test = max(1, int(round(n * test_frac))) if n >= 2 else 0
        n_val = max(1, int(round(n * val_frac))) if n - n_test >= 2 else 0
        n_train = n - n_test - n_val

        if n_train <= 0:
            n_train = max(1, n - n_val - n_test)

        test_names.extend(names[:n_test])
        val_names.extend(names[n_test:n_test + n_val])
        train_names.extend(names[n_test + n_val:n_test + n_val + n_train])

    rng.shuffle(train_names)
    rng.shuffle(val_names)
    rng.shuffle(test_names)
    return train_names, val_names, test_names


def label_distribution(names, name_to_label):
    return dict(sorted(Counter(name_to_label[n] for n in names).items()))


def write_split(splits_dir, seed, train_names, val_names, test_names, name_to_label):
    split_path = os.path.join(splits_dir, f'split_{seed}.json')
    summary_path = os.path.join(splits_dir, f'split_{seed}_summary.txt')

    with open(split_path, 'w', encoding='utf-8') as f:
        json.dump(
            {
                'train_names': train_names,
                'val_names': val_names,
                'test_names': test_names,
            },
            f,
        )

    train_dist = label_distribution(train_names, name_to_label)
    val_dist = label_distribution(val_names, name_to_label)
    test_dist = label_distribution(test_names, name_to_label)

    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write('Label Distribution Summary:\n\n')
        f.write('TRAIN_LABELS:\n')
        for k, v in train_dist.items():
            f.write(f'  {k}: {v}\n')
        f.write('\nVAL_LABELS:\n')
        for k, v in val_dist.items():
            f.write(f'  {k}: {v}\n')
        f.write('\nTEST_LABELS:\n')
        for k, v in test_dist.items():
            f.write(f'  {k}: {v}\n')

    print(
        f"[ok] wrote {split_path}  "
        f"(train={len(train_names)} {train_dist}, "
        f"val={len(val_names)} {val_dist}, "
        f"test={len(test_names)} {test_dist})"
    )


def main():
    args = get_arguments()

    if not os.path.exists(args.raw_csv):
        raise FileNotFoundError(f"raw_csv not found: {args.raw_csv}")

    os.makedirs(args.output_dir, exist_ok=True)
    splits_dir = os.path.join(args.output_dir, 'splits')
    os.makedirs(splits_dir, exist_ok=True)

    rows = read_raw_csv(args.raw_csv)
    if not rows:
        raise ValueError(f"raw_csv is empty: {args.raw_csv}")

    name_to_label, _ = build_dataset_csv(
        rows=rows,
        output_dir=args.output_dir,
        slide_ext=args.slide_ext,
        class_map=CLASS_MAP_DEFAULT,
    )

    for seed in args.seeds:
        train_names, val_names, test_names = stratified_split(
            name_to_label=name_to_label,
            val_frac=args.val_frac,
            test_frac=args.test_frac,
            seed=seed,
        )
        write_split(
            splits_dir=splits_dir,
            seed=seed,
            train_names=train_names,
            val_names=val_names,
            test_names=test_names,
            name_to_label=name_to_label,
        )


if __name__ == '__main__':
    main()
