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


def resolve_wsi_file_path(slide_name, ext, requested_dir=None, nas_root=None):
    file_name = f'{slide_name}.{ext}'
    candidates = []

    if requested_dir:
        candidates.append(requested_dir)

    env_source_dir = os.environ.get('SASHA_SOURCE_DIR')
    if env_source_dir:
        candidates.append(env_source_dir)

    if nas_root:
        candidates.append(nas_root)
        candidates.append(os.path.join(nas_root, 'raw_wsi'))

        nas_parent = os.path.dirname(nas_root.rstrip('/\\'))
        if nas_parent:
            candidates.append(nas_parent)
            candidates.append(os.path.join(nas_parent, 'Dataset'))
            candidates.append(os.path.join(nas_parent, 'Dataset', 'raw_wsi'))

    candidates.extend([
        '/mnt/nas/Dataset/raw_wsi',
        '/mnt/nas/Dataset',
        '/mnt/nas',
    ])

    seen = set()
    checked_dirs = []
    existing_dirs = []
    for path in candidates:
        if path is None:
            continue
        resolved = resolve_path(path, nas_root=nas_root, base_dir=os.getcwd())
        if resolved in seen:
            continue
        seen.add(resolved)
        checked_dirs.append(resolved)
        if not os.path.isdir(resolved):
            continue
        existing_dirs.append(resolved)

        direct_path = os.path.join(resolved, file_name)
        if os.path.isfile(direct_path):
            return direct_path

    # Fallback: recursive lookup under existing roots.
    for root_dir in existing_dirs:
        for root, _, files in os.walk(root_dir):
            if file_name in files:
                return os.path.join(root, file_name)

    raise FileNotFoundError(
        f"Could not find slide file '{file_name}'. Checked directories: {checked_dirs}. "
        f"Existing directories among them: {existing_dirs}. "
        f"Pass --wsi_images_dir_path explicitly to the directory containing '{file_name}'."
    )


def resolve_feature_dir_with_fallback(configured_dir, role_name, pretrain, nas_root=None, additional_roots=None):
    expected_h5 = f'patch_feats_pretrain_{pretrain}.h5'
    role_name = str(role_name)

    def _norm(path):
        return os.path.normpath(path)

    configured_dir = resolve_path(configured_dir, nas_root=nas_root, base_dir=os.getcwd())
    if os.path.isdir(configured_dir) and os.path.isfile(os.path.join(configured_dir, expected_h5)):
        return configured_dir, None

    candidates = [configured_dir, os.path.dirname(configured_dir)]

    if additional_roots:
        for root in additional_roots:
            if not root:
                continue
            candidates.append(root)
            candidates.append(os.path.dirname(root))

            parent = os.path.dirname(root)
            grand_parent = os.path.dirname(parent)
            if parent:
                candidates.append(os.path.join(parent, 'features'))
                candidates.append(os.path.join(parent, 'sasha_outputs', 'features'))
            if grand_parent:
                candidates.append(os.path.join(grand_parent, 'features'))
                candidates.append(os.path.join(grand_parent, 'sasha_outputs', 'features'))

    if nas_root:
        candidates.extend(
            [
                os.path.join(nas_root, 'sasha_outputs', 'features'),
                os.path.join(nas_root, 'features'),
            ]
        )

        nas_parent = os.path.dirname(nas_root.rstrip('/\\'))
        if nas_parent:
            candidates.extend(
                [
                    os.path.join(nas_parent, 'Dataset', 'sasha_outputs', 'features'),
                    os.path.join(nas_parent, 'sasha_outputs', 'features'),
                ]
            )

    candidates.extend(
        [
            os.path.join(os.getcwd(), 'sasha_outputs', 'features'),
            os.path.join(os.getcwd(), 'outputs', 'features'),
            os.path.join(os.getcwd(), 'features'),
            os.getcwd(),
            '/mnt/nas/Dataset/sasha_outputs/features',
            '/mnt/nas/sasha_outputs/features',
        ]
    )

    checked_roots = []
    existing_roots = []
    seen = set()
    found_dirs = []
    fallback_matches = []

    for candidate in candidates:
        if not candidate:
            continue
        root = resolve_path(candidate, nas_root=nas_root, base_dir=os.getcwd())
        root = _norm(root)
        if root in seen:
            continue
        seen.add(root)
        checked_roots.append(root)

        if not os.path.isdir(root):
            continue
        existing_roots.append(root)

        direct_h5 = os.path.join(root, expected_h5)
        if os.path.isfile(direct_h5):
            found_dirs.append(root)

        for file_name in os.listdir(root):
            if file_name.startswith('patch_feats_pretrain_') and file_name.endswith('.h5'):
                fallback_matches.append((root, file_name))

        for walk_root, _, files in os.walk(root):
            if expected_h5 in files:
                found_dirs.append(_norm(walk_root))
            for file_name in files:
                if file_name.startswith('patch_feats_pretrain_') and file_name.endswith('.h5'):
                    fallback_matches.append((_norm(walk_root), file_name))

    found_dirs = sorted(set(found_dirs))
    if not found_dirs and not fallback_matches:
        workspace_root = _norm(os.getcwd())
        non_workspace_existing_roots = [p for p in existing_roots if not _norm(p).startswith(workspace_root)]

        if len(existing_roots) == 0:
            raise FileNotFoundError(
                f"Could not locate '{expected_h5}' for {role_name}. Configured path: {configured_dir}. "
                f"Checked roots: {checked_roots}. Existing roots: {existing_roots}. "
                f"No candidate feature directories exist on this machine right now. "
                f"This usually means the NAS mount is unavailable or mounted at a different path."
            )

        if len(non_workspace_existing_roots) == 0:
            raise FileNotFoundError(
                f"Could not locate '{expected_h5}' for {role_name}. Configured path: {configured_dir}. "
                f"Checked roots: {checked_roots}. Existing roots: {existing_roots}. "
                f"Only workspace-local directories are visible; NAS-backed feature roots are not visible. "
                f"Likely causes: NAS not mounted or mounted at a different path than conf.nas_root."
            )

        raise FileNotFoundError(
            f"Could not locate '{expected_h5}' for {role_name}. "
            f"Configured path: {configured_dir}. Checked roots: {checked_roots}. "
            f"Existing roots: {existing_roots}."
        )

    def _score_dir(path):
        p = path.lower().replace('\\', '/')
        score = 0
        if role_name == 'level1_path':
            if 'intermediate' in p:
                score += 5
            if 'hafed' in p:
                score += 4
            if '/lr/' in p or '/lr' in p:
                score -= 3
            if 'level3' in p:
                score -= 3
        elif role_name == 'level3_path':
            if '/lr/' in p or '/lr' in p:
                score += 5
            if 'h5_files' in p:
                score += 3
            if 'level3' in p:
                score += 2
            if 'intermediate' in p:
                score -= 3
        if path == configured_dir:
            score += 1
        return score

    if found_dirs:
        best_dir = max(found_dirs, key=_score_dir)
        print(f"[WARN] {role_name} not found at configured path: {configured_dir}")
        print(f"[INFO] Auto-resolved {role_name}: {best_dir}")
        return best_dir, None

    fallback_matches = sorted(set(fallback_matches))
    best_dir, best_file = max(fallback_matches, key=lambda item: _score_dir(item[0]))
    fallback_pretrain = best_file[len('patch_feats_pretrain_'):-len('.h5')]
    print(f"[WARN] Expected file '{expected_h5}' for {role_name} was not found.")
    print(f"[WARN] Falling back to '{best_file}' under: {best_dir}")
    print(f"[INFO] Auto-resolved {role_name}: {best_dir}")
    return best_dir, fallback_pretrain


def load_pipeline(args):
    with open(args.config, 'r') as ymlfile:
        c = yaml.load(ymlfile, Loader=yaml.FullLoader)
        c.update(vars(args))
        conf = Struct(**c)

    resolve_conf_paths(conf, ['level1_path', 'level3_path', 'classifier_ckpt_path', 'mlp_fglobal_ckpt', 'rl_ckpt_path'], base_dir=os.getcwd())

    conf.classifier_ckpt_path = resolve_checkpoint_with_fallback(conf.classifier_ckpt_path, 'classifier_ckpt_path')
    conf.mlp_fglobal_ckpt = resolve_checkpoint_with_fallback(conf.mlp_fglobal_ckpt, 'mlp_fglobal_ckpt')
    conf.rl_ckpt_path = resolve_checkpoint_with_fallback(conf.rl_ckpt_path, 'rl_ckpt_path')

    path_hints = [
        os.path.dirname(conf.classifier_ckpt_path),
        os.path.dirname(conf.mlp_fglobal_ckpt),
        os.path.dirname(conf.rl_ckpt_path),
    ]

    conf.level1_path, level1_pretrain_override = resolve_feature_dir_with_fallback(
        configured_dir=conf.level1_path,
        role_name='level1_path',
        pretrain=conf.pretrain,
        nas_root=getattr(conf, 'nas_root', None) or os.environ.get('SASHA_NAS_ROOT'),
        additional_roots=path_hints,
    )
    conf.level3_path, level3_pretrain_override = resolve_feature_dir_with_fallback(
        configured_dir=conf.level3_path,
        role_name='level3_path',
        pretrain=conf.pretrain,
        nas_root=getattr(conf, 'nas_root', None) or os.environ.get('SASHA_NAS_ROOT'),
        additional_roots=path_hints,
    )

    discovered_pretrains = [p for p in [level1_pretrain_override, level3_pretrain_override] if p is not None]
    if len(discovered_pretrains) == 2 and discovered_pretrains[0] != discovered_pretrains[1]:
        raise RuntimeError(
            f"Conflicting pretrain tags discovered for level1/level3 files: {discovered_pretrains}. "
            f"Please set consistent level1_path and level3_path in config."
        )
    if len(discovered_pretrains) > 0 and discovered_pretrains[0] != conf.pretrain:
        print(f"[WARN] Overriding conf.pretrain from '{conf.pretrain}' to '{discovered_pretrains[0]}' based on discovered feature file.")
        conf.pretrain = discovered_pretrains[0]

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


def get_slide_coords(features_path, slide_name, pretrain=None):
    h5_files = sorted([f for f in os.listdir(features_path) if f.endswith('.h5')])
    if len(h5_files) == 0:
        raise RuntimeError(f'No .h5 files found under {features_path}')

    preferred = None
    if pretrain is not None:
        preferred_name = f'patch_feats_pretrain_{pretrain}.h5'
        if preferred_name in h5_files:
            preferred = preferred_name

    h5_file = preferred if preferred is not None else h5_files[0]
    h5_file_path = os.path.join(features_path, h5_file)
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

    coords = get_slide_coords(conf.level3_path, slide_name, pretrain=conf.pretrain)
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

    if level >= slide.level_count:
        actual_level = slide.level_count - 1
        print(f"[WARN] Requested level={level} but slide only has {slide.level_count} levels. Using level={actual_level}.")
        level = actual_level

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

    wsi_path = resolve_wsi_file_path(
        slide_name=args.slide_name,
        ext=args.ext,
        requested_dir=args.wsi_images_dir_path,
        nas_root=os.environ.get('SASHA_NAS_ROOT'),
    )
    print(f'[INFO] Using WSI file: {wsi_path}')

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