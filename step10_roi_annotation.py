"""
Step 10: ROI annotation for doctor verification.

This script runs the trained SASHA inference pipeline for a target slide, computes patch-level
suspicion scores, groups high-score patches into connected ROIs, and draws these ROIs on the
original WSI (.tif/.svs).

Outputs:
- <slide_name>_roi_overlay.png : WSI with ROI contour overlay and labels
- <slide_name>_roi.csv         : ROI coordinates in level-0 space
- <slide_name>_roi.json        : ROI metadata

Example:
python step10_roi_annotation.py \
  --config config/camelyon_sasha_inference.yml \
  --slide_name test_068 \
  --ext tif \
  --wsi_images_dir_path "$SASHA_SOURCE_DIR" \
  --output_dir_path /mnt/nas/Dataset/sasha_outputs/roi_annotations/test_068 \
  --seed 1
"""

import argparse
import json
import os
from types import SimpleNamespace

import cv2
import h5py
import numpy as np
import openslide
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from architecture.transformer import HAFED
from datasets.datasets import build_HDF5_feat_dataset_2
from envs.WSI_cosine_env import WSICosineObservationEnv
from envs.WSI_env import WSIObservationEnv
from modules.fglobal_mlp import FGlobal
from rl_algorithms.ppo import Actor, Agent, Critic
from step4_extract_intermediate_features import load_model
from step7_inference import load_policy_model
from utils.gpu_utils import check_gpu_availability
from utils.path_utils import ensure_path_exists, load_env_file, resolve_conf_paths, resolve_path
from utils.utils import MetricLogger, Struct, set_seed


def get_arguments():
    parser = argparse.ArgumentParser('Step10 ROI annotation', add_help=False)
    parser.add_argument('--config', required=True, help='path to inference config file')
    parser.add_argument('--slide_name', type=str, required=True, help='slide name without extension, e.g. test_068')
    parser.add_argument('--ext', type=str, default='tif', help='slide extension, e.g. tif or svs')
    parser.add_argument('--wsi_images_dir_path', type=str, default=None, help='directory containing raw WSI files')
    parser.add_argument('--output_dir_path', type=str, default=None, help='directory to save step10 outputs')
    parser.add_argument('--seed', type=int, default=1, help='split seed used by trained models')
    parser.add_argument('--classifier_arch', default='hafed', choices=['hafed'], help='classifier architecture')

    parser.add_argument('--level', type=int, default=6, help='OpenSlide level for ROI overlay image')
    parser.add_argument('--patch_size_level0', type=int, default=2048, help='low-resolution patch size expressed in level-0 pixels')
    parser.add_argument('--roi_percentile', type=float, default=85.0, help='percentile cutoff on suspicion score for ROI candidates')
    parser.add_argument('--min_roi_patches', type=int, default=8, help='minimum connected candidate patches to keep an ROI')

    parser.add_argument('--selected_weight', type=float, default=1.0, help='weight for RL-selected patches')
    parser.add_argument('--similar_weight', type=float, default=0.35, help='weight for similarity-updated patches')
    parser.add_argument('--attention_weight', type=float, default=0.65, help='weight for classifier attention scores')
    parser.add_argument('--contour_alpha', type=float, default=0.30, help='alpha for filled contour overlay [0,1]')
    parser.add_argument('--contour_thickness', type=int, default=3, help='line thickness for ROI contour boundaries')

    args = parser.parse_args()

    load_env_file(os.path.join(os.getcwd(), '.env'))
    nas_root = os.environ.get('SASHA_NAS_ROOT')

    if args.wsi_images_dir_path is None and os.environ.get('SASHA_SOURCE_DIR'):
        args.wsi_images_dir_path = os.environ['SASHA_SOURCE_DIR']
    if args.output_dir_path is None and nas_root:
        args.output_dir_path = os.path.join(nas_root, 'sasha_outputs', 'roi_annotations', args.slide_name)

    args.wsi_images_dir_path = resolve_path(args.wsi_images_dir_path, nas_root=nas_root, base_dir=os.getcwd())
    args.output_dir_path = resolve_path(args.output_dir_path, nas_root=nas_root, base_dir=os.getcwd())
    args.config = resolve_path(args.config, nas_root=None, base_dir=os.getcwd())

    gpus = check_gpu_availability(3, 1, [])
    print(f"occupied {gpus}")
    args.device = torch.device(f"cuda:{gpus[0]}")

    return args


def resolve_checkpoint_with_fallback(path, field_name):
    if os.path.exists(path):
        return path

    if path.endswith('checkpoint-best.pt'):
        fallback_path = path.replace('checkpoint-best.pt', 'checkpoint-last.pt')
        if os.path.exists(fallback_path):
            print(f"[WARN] {field_name} not found at {path}. Using fallback: {fallback_path}")
            return fallback_path

    return path


def load_pipeline(args):
    with open(args.config, 'r') as ymlfile:
        c = yaml.load(ymlfile, Loader=yaml.FullLoader)
        c.update(vars(args))
        conf = Struct(**c)

    resolve_conf_paths(conf, ['level1_path', 'level3_path', 'classifier_ckpt_path', 'mlp_fglobal_ckpt', 'rl_ckpt_path'], base_dir=os.getcwd())

    conf.classifier_ckpt_path = resolve_checkpoint_with_fallback(conf.classifier_ckpt_path, 'classifier_ckpt_path')
    conf.mlp_fglobal_ckpt = resolve_checkpoint_with_fallback(conf.mlp_fglobal_ckpt, 'mlp_fglobal_ckpt')
    conf.rl_ckpt_path = resolve_checkpoint_with_fallback(conf.rl_ckpt_path, 'rl_ckpt_path')

    ensure_path_exists(conf.level1_path, 'level1_path', expect_dir=True)
    ensure_path_exists(conf.level3_path, 'level3_path', expect_dir=True)
    ensure_path_exists(conf.classifier_ckpt_path, 'classifier_ckpt_path', expect_dir=False)
    ensure_path_exists(conf.mlp_fglobal_ckpt, 'mlp_fglobal_ckpt', expect_dir=False)
    ensure_path_exists(conf.rl_ckpt_path, 'rl_ckpt_path', expect_dir=False)

    set_seed(args.seed)

    train_data, val_data, test_data = build_HDF5_feat_dataset_2(conf.level1_path, conf.level3_path, conf)
    test_loader = DataLoader(test_data, batch_size=conf.B, shuffle=False, num_workers=conf.n_worker, pin_memory=conf.pin_memory, drop_last=False)

    classifier_dict, _, config, _ = load_model(conf.classifier_ckpt_path, args)
    classifier_conf = SimpleNamespace(**config)

    if conf.classifier_arch == 'hafed':
        classifier = HAFED(
            classifier_conf,
            n_token_1=classifier_conf.n_token_1,
            n_token_2=classifier_conf.n_token_2,
            n_masked_patch_1=classifier_conf.n_masked_patch_1,
            n_masked_patch_2=classifier_conf.n_masked_patch_2,
            mask_drop=classifier_conf.mask_drop,
        )
    else:
        raise Exception('Select a valid classifier architecture.')

    classifier.to(conf.device)
    classifier.load_state_dict(classifier_dict)
    classifier.eval()

    fglobal_dict = torch.load(conf.mlp_fglobal_ckpt, map_location=conf.device)
    fglobal = FGlobal(ip_dim=384 * 3, op_dim=384).to(conf.device)
    fglobal.load_state_dict(fglobal_dict['model'])
    fglobal.eval()

    actor = Actor(conf=conf)
    critic = Critic(conf=conf)
    model = Agent(actor, critic, conf).to(conf.device)
    actor_optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, actor.parameters()), lr=0.001)
    critic_optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, critic.parameters()), lr=0.001)
    model, actor_optimizer, critic_optimizer, epoch, _ = load_policy_model(model, actor_optimizer, critic_optimizer, conf.rl_ckpt_path, conf.device)

    return conf, test_loader, classifier, fglobal, model


def get_slide_coords(features_path, slide_name):
    h5_files = [f for f in os.listdir(features_path) if f.endswith('.h5')]
    if len(h5_files) != 1:
        raise RuntimeError(f'Expected exactly one .h5 file under {features_path}, found {len(h5_files)}')

    h5_file_path = os.path.join(features_path, h5_files[0])
    with h5py.File(h5_file_path, 'r') as f:
        if slide_name not in f:
            raise KeyError(f"Slide '{slide_name}' not found in {h5_file_path}")
        if 'coords' not in f[slide_name]:
            raise KeyError(f"'coords' missing under slide '{slide_name}'")
        return f[slide_name]['coords'][:]


def extract_patch_attention(attn_tensor, n_patches):
    if attn_tensor is None:
        return np.zeros((n_patches,), dtype=np.float32)

    attn = attn_tensor.detach().cpu().numpy()

    # Common shapes: [1, token, N], [token, N], [N]
    if attn.ndim == 3:
        # average over batch and token dimensions
        attn = attn.mean(axis=(0, 1))
    elif attn.ndim == 2:
        if attn.shape[1] == n_patches:
            attn = attn.mean(axis=0)
        elif attn.shape[0] == n_patches:
            attn = attn.mean(axis=1)
        else:
            attn = attn.reshape(-1)
    else:
        attn = attn.reshape(-1)

    if attn.shape[0] != n_patches:
        return np.zeros((n_patches,), dtype=np.float32)

    attn = attn.astype(np.float32)
    attn_min = float(attn.min())
    attn_max = float(attn.max())
    if attn_max > attn_min:
        attn = (attn - attn_min) / (attn_max - attn_min)
    else:
        attn = np.zeros_like(attn)

    return attn


@torch.no_grad()
def evaluate_policy_for_slide(model, fglobal, classifier, data_loader, device, conf, slide_name):
    model.eval()

    selected_indices = []
    similar_groups = []
    final_state = None
    final_attn = None

    metric_logger = MetricLogger(delimiter=' ')

    found = False
    for data in metric_logger.log_every(data_loader, 100, 'Step10'):
        if slide_name != data['slide_name'][0]:
            continue

        found = True
        hr_features = data['hr'][0].to(device, dtype=torch.float32)
        state = data['lr'].to(device, dtype=torch.float32)
        label = data['label'].to(device)

        if conf.fglobal == 'attn':
            env = WSIObservationEnv(lr_features=state, hr_features=hr_features, label=label, conf=conf)
        else:
            env = WSICosineObservationEnv(lr_features=state, hr_features=hr_features, label=label, conf=conf)

        visited_patch_id = []
        done = False

        while not done:
            action, _, _ = model.get_action(state, visited_patch_id, is_eval=True)
            state, _, done = env.step(action=action, state_update_net=fglobal, classifier_net=classifier, device=device)
            visited_patch_id.append(action.item())
            selected_indices.append(action.item())

            sim = env.get_similar_patches()
            if sim is None:
                similar_groups.append([])
            else:
                similar_groups.append(sim.detach().cpu().tolist())

        final_state = state
        _, final_attn = classifier.classify(final_state)
        break

    if not found:
        raise ValueError(f"Slide '{slide_name}' not found in test split for seed={conf.seed}")

    coords = get_slide_coords(conf.level3_path, slide_name)
    attention_scores = extract_patch_attention(final_attn, coords.shape[0])

    return coords, selected_indices, similar_groups, attention_scores


def build_suspicion_scores(n_patches, selected_indices, similar_groups, attention_scores, selected_w, similar_w, attn_w):
    scores = np.zeros((n_patches,), dtype=np.float32)

    for idx in selected_indices:
        if 0 <= idx < n_patches:
            scores[idx] += selected_w

    for group in similar_groups:
        for idx in group:
            if 0 <= idx < n_patches:
                scores[idx] += similar_w

    if attention_scores is not None and attention_scores.shape[0] == n_patches:
        scores += attn_w * attention_scores

    return scores


def generate_connected_rois(coords, scores, patch_size_level0, roi_percentile, min_roi_patches):
    positive = scores[scores > 0]
    if positive.size == 0:
        return []

    threshold = float(np.percentile(positive, roi_percentile))
    candidate_indices = np.where(scores >= threshold)[0]

    if candidate_indices.size == 0:
        top_idx = int(np.argmax(scores))
        candidate_indices = np.array([top_idx], dtype=np.int64)

    gx = np.round(coords[:, 0] / patch_size_level0).astype(np.int32)
    gy = np.round(coords[:, 1] / patch_size_level0).astype(np.int32)

    grid_to_indices = {}
    for idx in candidate_indices.tolist():
        key = (int(gx[idx]), int(gy[idx]))
        grid_to_indices.setdefault(key, []).append(idx)

    xs = np.array([k[0] for k in grid_to_indices.keys()], dtype=np.int32)
    ys = np.array([k[1] for k in grid_to_indices.keys()], dtype=np.int32)

    min_x = int(xs.min())
    max_x = int(xs.max())
    min_y = int(ys.min())
    max_y = int(ys.max())

    mask = np.zeros((max_y - min_y + 1, max_x - min_x + 1), dtype=np.uint8)
    for xg, yg in grid_to_indices.keys():
        mask[yg - min_y, xg - min_x] = 1

    n_labels, labels = cv2.connectedComponents(mask, connectivity=8)

    rois = []
    for label in range(1, n_labels):
        yy, xx = np.where(labels == label)
        comp_indices = []
        for y_cell, x_cell in zip(yy.tolist(), xx.tolist()):
            key = (x_cell + min_x, y_cell + min_y)
            comp_indices.extend(grid_to_indices.get(key, []))

        if len(comp_indices) < min_roi_patches:
            continue

        comp_indices = np.array(comp_indices, dtype=np.int64)
        x_vals = coords[comp_indices, 0]
        y_vals = coords[comp_indices, 1]

        x_min = int(x_vals.min())
        y_min = int(y_vals.min())
        x_max = int(x_vals.max() + patch_size_level0)
        y_max = int(y_vals.max() + patch_size_level0)

        roi = {
            'roi_id': len(rois) + 1,
            'x_min': x_min,
            'y_min': y_min,
            'x_max': x_max,
            'y_max': y_max,
            'n_patches': int(len(comp_indices)),
            'score_mean': float(scores[comp_indices].mean()),
            'score_max': float(scores[comp_indices].max()),
            'patch_indices': comp_indices.tolist(),
        }
        rois.append(roi)

    if len(rois) == 0:
        # Fallback: provide at least one ROI around top scoring patch.
        top_idx = int(np.argmax(scores))
        x = int(coords[top_idx, 0])
        y = int(coords[top_idx, 1])
        rois.append(
            {
                'roi_id': 1,
                'x_min': x,
                'y_min': y,
                'x_max': x + patch_size_level0,
                'y_max': y + patch_size_level0,
                'n_patches': 1,
                'score_mean': float(scores[top_idx]),
                'score_max': float(scores[top_idx]),
                'patch_indices': [top_idx],
            }
        )

    rois.sort(key=lambda item: item['score_mean'], reverse=True)
    for idx, roi in enumerate(rois, start=1):
        roi['roi_id'] = idx

    return rois


def draw_rois_on_wsi(wsi_path, rois, coords, patch_size_level0, level, output_path, contour_alpha, contour_thickness):
    slide = openslide.OpenSlide(wsi_path)

    downscale_factor = float(slide.level_downsamples[level])
    wsi_size = slide.level_dimensions[level]
    wsi_img = slide.read_region((0, 0), level, wsi_size).convert('RGB')
    base_img = np.array(wsi_img)
    h, w = base_img.shape[:2]
    fill_layer = np.zeros_like(base_img)

    patch_size_level = max(1, int(round(float(patch_size_level0) / downscale_factor)))
    palette = [
        (255, 0, 0),
        (255, 128, 0),
        (255, 255, 0),
        (0, 200, 255),
        (0, 255, 0),
        (0, 128, 255),
    ]

    for roi in rois:
        roi_mask = np.zeros((h, w), dtype=np.uint8)
        patch_indices = roi.get('patch_indices', [])
        for idx in patch_indices:
            if idx < 0 or idx >= coords.shape[0]:
                continue
            x0 = int(round(float(coords[idx, 0]) / downscale_factor))
            y0 = int(round(float(coords[idx, 1]) / downscale_factor))
            x1 = min(w, x0 + patch_size_level)
            y1 = min(h, y0 + patch_size_level)
            if x0 >= w or y0 >= h or x1 <= 0 or y1 <= 0:
                continue
            x0 = max(0, x0)
            y0 = max(0, y0)
            roi_mask[y0:y1, x0:x1] = 255

        if roi_mask.sum() == 0:
            continue

        kernel = np.ones((3, 3), dtype=np.uint8)
        roi_mask = cv2.morphologyEx(roi_mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            continue

        color = palette[(int(roi['roi_id']) - 1) % len(palette)]
        cv2.drawContours(fill_layer, contours, -1, color, thickness=cv2.FILLED)

    contour_alpha = float(np.clip(contour_alpha, 0.0, 1.0))
    out_img = cv2.addWeighted(base_img, 1.0, fill_layer, contour_alpha, 0.0)

    for roi in rois:
        roi_mask = np.zeros((h, w), dtype=np.uint8)
        patch_indices = roi.get('patch_indices', [])
        for idx in patch_indices:
            if idx < 0 or idx >= coords.shape[0]:
                continue
            x0 = int(round(float(coords[idx, 0]) / downscale_factor))
            y0 = int(round(float(coords[idx, 1]) / downscale_factor))
            x1 = min(w, x0 + patch_size_level)
            y1 = min(h, y0 + patch_size_level)
            if x0 >= w or y0 >= h or x1 <= 0 or y1 <= 0:
                continue
            x0 = max(0, x0)
            y0 = max(0, y0)
            roi_mask[y0:y1, x0:x1] = 255

        if roi_mask.sum() == 0:
            continue

        kernel = np.ones((3, 3), dtype=np.uint8)
        roi_mask = cv2.morphologyEx(roi_mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            continue

        color = palette[(int(roi['roi_id']) - 1) % len(palette)]
        cv2.drawContours(out_img, contours, -1, color, thickness=max(1, int(contour_thickness)))

        largest = max(contours, key=cv2.contourArea)
        x_lbl, y_lbl, _, _ = cv2.boundingRect(largest)
        label = f"ROI-{roi['roi_id']}"
        cv2.putText(out_img, label, (x_lbl, max(20, y_lbl - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR))
    return downscale_factor


def save_roi_files(rois, output_dir, slide_name):
    os.makedirs(output_dir, exist_ok=True)

    csv_path = os.path.join(output_dir, f'{slide_name}_roi.csv')
    json_path = os.path.join(output_dir, f'{slide_name}_roi.json')

    exportable_rois = []
    for roi in rois:
        exportable_rois.append({k: v for k, v in roi.items() if k != 'patch_indices'})

    df = pd.DataFrame(exportable_rois)
    df.to_csv(csv_path, index=False)

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(exportable_rois, f, indent=2)

    return csv_path, json_path


def main():
    args = get_arguments()

    ensure_path_exists(args.config, 'config', expect_dir=False)
    ensure_path_exists(args.wsi_images_dir_path, 'wsi_images_dir_path', expect_dir=True)

    conf, test_loader, classifier, fglobal, model = load_pipeline(args)

    coords, selected_indices, similar_groups, attention_scores = evaluate_policy_for_slide(
        model=model,
        fglobal=fglobal,
        classifier=classifier,
        data_loader=test_loader,
        device=conf.device,
        conf=conf,
        slide_name=args.slide_name,
    )

    scores = build_suspicion_scores(
        n_patches=coords.shape[0],
        selected_indices=selected_indices,
        similar_groups=similar_groups,
        attention_scores=attention_scores,
        selected_w=args.selected_weight,
        similar_w=args.similar_weight,
        attn_w=args.attention_weight,
    )

    rois = generate_connected_rois(
        coords=coords,
        scores=scores,
        patch_size_level0=args.patch_size_level0,
        roi_percentile=args.roi_percentile,
        min_roi_patches=args.min_roi_patches,
    )

    wsi_path = os.path.join(args.wsi_images_dir_path, f'{args.slide_name}.{args.ext}')
    ensure_path_exists(wsi_path, 'wsi_path', expect_dir=False)

    overlay_path = os.path.join(args.output_dir_path, f'{args.slide_name}_roi_overlay.png')
    draw_rois_on_wsi(
        wsi_path=wsi_path,
        rois=rois,
        coords=coords,
        patch_size_level0=args.patch_size_level0,
        level=args.level,
        output_path=overlay_path,
        contour_alpha=args.contour_alpha,
        contour_thickness=args.contour_thickness,
    )

    csv_path, json_path = save_roi_files(rois, args.output_dir_path, args.slide_name)

    print(f'ROI overlay saved at: {overlay_path}')
    print(f'ROI table saved at: {csv_path}')
    print(f'ROI metadata saved at: {json_path}')


if __name__ == '__main__':
    main()
