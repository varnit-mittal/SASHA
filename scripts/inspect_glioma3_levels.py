"""
Quick pyramid-level inspector for the glioma3 dataset (or any folder of
.svs slides). Walks every slide, prints how many OpenSlide pyramid levels
it has, and emits two lists at the end:

    1) slides whose pyramid is too shallow for the requested patch_level
       (these are the ones that crashed step1_create_patches.py with an
       IndexError on `level_downsamples[patch_level]`).
    2) slides ok for that patch_level.

Usage:
    python scripts/inspect_glioma3_levels.py \
        --slides_dir /mnt/nas/glioma_slides \
        --patch_level 2 \
        --csv_path dataset_csv/glioma3/glioma3.csv \
        --report_csv outputs/glioma3_pyramid_report.csv

If `--csv_path` is provided, the script will also write a copy of that CSV
filtered to keep only slides that passed the level check (suffix
`_filtered.csv`), so you can re-build splits with `build_glioma3_dataset.py`
on a clean subset.
"""

import argparse
import csv
import glob
import os
from collections import Counter

try:
    import openslide
except ImportError as exc:
    raise SystemExit(
        "openslide-python is required. Install via the project conda env or pip "
        "(see conda-packages.txt)."
    ) from exc


def get_arguments():
    parser = argparse.ArgumentParser('Inspect WSI pyramid levels (glioma3)')
    parser.add_argument('--slides_dir', type=str, required=True,
                        help='Directory containing the .svs slides (e.g. /mnt/nas/glioma_slides).')
    parser.add_argument('--slide_ext', type=str, default='.svs')
    parser.add_argument('--patch_level', type=int, default=2,
                        help='patch_level you plan to pass to step1_create_patches.py.')
    parser.add_argument('--csv_path', type=str, default=None,
                        help='Optional dataset CSV (case_id,slide_id,label). Used to filter and write a clean copy.')
    parser.add_argument('--report_csv', type=str, default=None,
                        help='Optional output CSV with per-slide details.')
    return parser.parse_args()


def inspect_slide(slide_path):
    try:
        wsi = openslide.open_slide(slide_path)
    except Exception as exc:
        return {'error': str(exc)}

    info = {
        'levels': wsi.level_count,
        'level_dimensions': list(wsi.level_dimensions),
        'level_downsamples': [round(float(d), 4) for d in wsi.level_downsamples],
        'objective_power': wsi.properties.get('openslide.objective-power', ''),
        'mpp_x': wsi.properties.get('openslide.mpp-x', ''),
        'mpp_y': wsi.properties.get('openslide.mpp-y', ''),
    }
    wsi.close()
    return info


def main():
    args = get_arguments()

    slides = sorted(glob.glob(os.path.join(args.slides_dir, f'*{args.slide_ext}')))
    if not slides:
        raise SystemExit(f"No '*{args.slide_ext}' slides found under {args.slides_dir}")

    print(f"Inspecting {len(slides)} slides under {args.slides_dir}")
    print(f"Target patch_level = {args.patch_level} (slide must have at least {args.patch_level + 1} levels)\n")

    rows = []
    level_counter = Counter()
    ok_slides = []
    bad_slides = []
    error_slides = []

    for slide_path in slides:
        slide_id = os.path.basename(slide_path)
        info = inspect_slide(slide_path)

        if 'error' in info:
            error_slides.append((slide_id, info['error']))
            rows.append({
                'slide_id': slide_id, 'levels': '', 'level_dimensions': '',
                'level_downsamples': '', 'objective_power': '', 'mpp_x': '',
                'mpp_y': '', 'patch_level_supported': '', 'error': info['error'],
            })
            continue

        level_counter[info['levels']] += 1
        supported = info['levels'] > args.patch_level
        (ok_slides if supported else bad_slides).append(slide_id)

        rows.append({
            'slide_id': slide_id,
            'levels': info['levels'],
            'level_dimensions': str(info['level_dimensions']),
            'level_downsamples': str(info['level_downsamples']),
            'objective_power': info['objective_power'],
            'mpp_x': info['mpp_x'],
            'mpp_y': info['mpp_y'],
            'patch_level_supported': supported,
            'error': '',
        })

    print("=== Pyramid-level distribution ===")
    for k in sorted(level_counter):
        print(f"  {k} levels : {level_counter[k]} slides")
    print(f"  unreadable : {len(error_slides)} slides")

    print("\n=== Summary ===")
    print(f"  ok for patch_level={args.patch_level} : {len(ok_slides)}")
    print(f"  too shallow                          : {len(bad_slides)}")
    if bad_slides:
        print("  -> sample of slides that would break step1:")
        for s in bad_slides[:10]:
            print(f"     {s}")

    if error_slides:
        print("\n=== Unreadable slides ===")
        for slide_id, err in error_slides[:10]:
            print(f"  {slide_id}: {err}")

    if args.report_csv:
        os.makedirs(os.path.dirname(args.report_csv) or '.', exist_ok=True)
        with open(args.report_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nWrote per-slide report to {args.report_csv}")

    if args.csv_path:
        if not os.path.exists(args.csv_path):
            raise SystemExit(f"--csv_path not found: {args.csv_path}")

        ok_set = set(ok_slides)
        bad_set = set(bad_slides + [s for s, _ in error_slides])

        out_path = os.path.splitext(args.csv_path)[0] + '_filtered.csv'
        kept = 0
        dropped = 0
        with open(args.csv_path, 'r', encoding='utf-8') as src, \
             open(out_path, 'w', encoding='utf-8') as dst:
            header = src.readline()
            dst.write(header)
            slide_idx = header.strip().split(',').index('slide_id')
            for line in src:
                slide_id = line.split(',')[slide_idx].strip()
                if slide_id in ok_set:
                    dst.write(line)
                    kept += 1
                elif slide_id in bad_set:
                    dropped += 1

        print(
            f"\nWrote filtered dataset CSV to {out_path} "
            f"(kept={kept}, dropped={dropped}). Re-run scripts/build_glioma3_dataset.py "
            f"with --raw_csv pointing at the original labels file (after dropping the "
            f"same slide_ids) to regenerate splits."
        )


if __name__ == '__main__':
    main()
