"""
This is the complete loop, where we are doing feature extraction for selective zooming, and running the inference loop
Description:
This script is used to run inference using the SASHA pipeline (versions 0.1 and 0.2).
It assumes that all the required components — HAFED, TSU, and RL — have been trained beforehand.

Specifically, this script loads the trained models, performs inference over the target dataset,
and generates predictions according to the specified SASHA configuration.

Supported SASHA Variants:
- SASHA-0.1
- SASHA-0.2

Make sure to configure the required paths to checkpoints and datasets before execution.


Command
python step7_inference_with_fe.py --config config/camelyon_sasha_inference_with_fe.yml --seed 4 --save_dir SAVE_DIR

Outputs written under SAVE_DIR:
- STEP1/, STEP2/                       : intermediate patches and low-res features
- visualizations/<slide>_patches.png   : per-slide PNG showing every low-res patch
                                          (gray boxes) and the patches the RL agent
                                          actually visited (blue gradient by step)
- metrics_sasha_<frac>.json            : aggregate AND per-class precision / recall /
                                          f1 / accuracy / support
- visit_log_sasha_<frac>.pt            : per-slide dict with all_lr_coords,
                                          visited_coords, visited_ids, true_label,
                                          pred_label
- time_dict_sasha_<frac>.pt            : per-slide wall-clock timings

Extra CLI flags:
- --save_visualizations / --no_save_visualizations  : toggle the per-slide PNGs (on by default)
- --vis_level <int>                                 : OpenSlide pyramid level for the WSI
                                                       thumbnail used as the visualization
                                                       backdrop. Defaults to
                                                       patch_level_low_res + 3 (clamped).
Optional config field:
- class_names: ["normal", "tumor"]   # used in the per-class table; defaults to class_0 / class_1 / ...
"""

import argparse
import json
import os
import time
from collections import defaultdict
from types import SimpleNamespace

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw, ImageFont

from architecture.transformer import HAFED
from envs.WSI_cosine_env_inference import WSICosineObservationEnv_inference
from tqdm import tqdm

from datasets.dataset_h5 import Dataset_All_Bags
from modules.fglobal_mlp import FGlobal
from rl_algorithms.ppo import Agent, Actor, Critic
from utils.gpu_utils import check_gpu_availability
from utils.inference_utils import Helper
from utils.metrics import (
    compute_accuracy,
    compute_auroc,
    compute_balanced_accuracy,
    compute_f1,
    compute_per_class_classification_metrics,
    compute_precision,
    compute_recall,
    format_per_class_metrics_table,
)
from utils.path_utils import ensure_path_exists, resolve_conf_paths
from utils.utils import Struct


def get_arguments():

    parser = argparse.ArgumentParser('Inference with Feature Extraction', add_help=False)

    # Patching arguments
    parser.add_argument('--step_size', type=int, default=256, help='step_size')
    parser.add_argument('--patch_size', type=int, default=256, help='patch_size')
    parser.add_argument('--extension', default='tif', help='extension to processes data type, e.g. *.svs, *.tif')

    # Feature Extraction arguments
    parser.add_argument('--batch_size_hr', type=int, default=1)
    parser.add_argument('--batch_size_lr', type=int, default=512)
    parser.add_argument('--target_patch_size', type=int, default=224)
    parser.add_argument('--slide_ext', type=str, default="tif", help="we have two options tif, svs, or any other compatible can work")
    parser.add_argument('--extract_high_res_features', type=bool, default=True, help="To create a mapping from high resolution to low resolution")
    parser.add_argument('--patch_level_low_res', type=int, default=3)  # Low  represents the magnified level [ Just Make sure that patch level should match from create patches ]
    parser.add_argument('--patch_level_high_res', type=int, default=1)  # High represents the scanning level

    # RL Models
    parser.add_argument('--classifier_arch', default='hafed', help='choice of architecture for HAFED')
    parser.add_argument('--config', default=None, type=str, help='config file path')
    parser.add_argument('--seed', type=int, default=4, help='set the random seed')
    parser.add_argument('--save_dir', default=None, help= 'folder path to save intermediate steps')

    # Visualization arguments
    parser.add_argument('--save_visualizations', action='store_true', default=True,
                        help='If set, saves a PNG per slide showing all low-res patches and RL-visited patches.')
    parser.add_argument('--no_save_visualizations', dest='save_visualizations', action='store_false',
                        help='Disable per-slide visualization PNGs.')
    parser.add_argument('--vis_level', type=int, default=None,
                        help='OpenSlide pyramid level used to render the WSI thumbnail for visualization. '
                             'If None, defaults to patch_level_low_res + 3 (clamped to the deepest level).')

    args = parser.parse_args()

    # Adding Device Details
    gpus = check_gpu_availability(3, 1, [])
    print(f"occupied {gpus}")
    args.device = torch.device(f"cuda:{gpus[0]}")

    return args


@torch.no_grad()
def load_agent(conf):
    dict = torch.load(conf.rl_ckpt_path, map_location=conf.device)
    config = dict['config']
    model_dict = dict['model']
    return model_dict, config


@torch.no_grad()
def load_model(ckpt_path, device):
    dict = torch.load(ckpt_path, map_location=device)
    config = dict['config']
    model_dict = dict['model']
    return model_dict, config

@torch.no_grad()
def get_classifier(conf):
    classifier_dict, config = load_model(conf.classifier_ckpt_path, conf.device)
    classifier_conf = SimpleNamespace(**config)
    classifier = HAFED(classifier_conf, n_token_1=classifier_conf.n_token_1, n_token_2=classifier_conf.n_token_2, n_masked_patch_1=classifier_conf.n_masked_patch_1, n_masked_patch_2=classifier_conf.n_masked_patch_2, mask_drop=classifier_conf.mask_drop)
    classifier.to(conf.device)
    classifier.load_state_dict(classifier_dict)
    classifier.eval()
    return classifier

@torch.no_grad()
def get_tsu(conf):
    fglobal_dict = torch.load(conf.mlp_fglobal_ckpt, map_location=conf.device)['model']
    fglobal = FGlobal().to(conf.device)
    fglobal.load_state_dict(fglobal_dict)
    fglobal.eval()
    return fglobal

def _resolve_vis_level(wsi, requested_level, patch_level_low_res):
    """Pick a safe pyramid level for the visualization thumbnail. Falls back
    to the deepest level openslide exposes if the requested one is too high."""
    n_levels = wsi.level_count
    if requested_level is None:
        candidate = int(patch_level_low_res) + 3
    else:
        candidate = int(requested_level)
    return max(0, min(candidate, n_levels - 1))


def visualize_slide_patches(
    wsi,
    slide_name,
    all_lr_coords,
    visited_patch_coords,
    patch_size_low_res,
    patch_level_low_res,
    save_path,
    requested_vis_level=None,
    title_suffix=None,
):
    """Render a single PNG per slide showing:
        * a thumbnail of the WSI at a low-magnification pyramid level,
        * every low-resolution patch (gray boxes) computed during step1,
        * the patches the RL agent actually visited (blue gradient by visit order).

    Coords passed in are level-0 coordinates, matching what
    Whole_Slide_Bag_FP / step2 produce. They get scaled by the thumbnail
    downsample factor so the boxes line up with the rendered image.
    """
    vis_level = _resolve_vis_level(wsi, requested_vis_level, patch_level_low_res)
    downscale_factor = wsi.level_downsamples[vis_level]
    wsi_size = wsi.level_dimensions[vis_level]
    wsi_img = wsi.read_region((0, 0), vis_level, wsi_size).convert("RGB")
    wsi_np = np.array(wsi_img)

    patch_size_level0 = int(patch_size_low_res) * int(2 ** int(patch_level_low_res))
    box_w = max(1, int(patch_size_level0 / downscale_factor))

    # Layer 1: every low-res patch the agent could have chosen (light gray).
    if all_lr_coords is not None and len(all_lr_coords) > 0:
        lr_layer = wsi_np.copy()
        for (x_lvl0, y_lvl0) in np.asarray(all_lr_coords).reshape(-1, 2):
            x_scaled = int(int(x_lvl0) / downscale_factor)
            y_scaled = int(int(y_lvl0) / downscale_factor)
            cv2.rectangle(
                lr_layer,
                (x_scaled, y_scaled),
                (x_scaled + box_w, y_scaled + box_w),
                color=(180, 180, 180),
                thickness=2,
            )
        wsi_np = cv2.addWeighted(wsi_np, 0.55, lr_layer, 0.45, 0)

    # Layer 2: patches the RL agent actually visited (Blues_r gradient by step).
    if visited_patch_coords is not None and len(visited_patch_coords) > 0:
        norm = plt.Normalize(0.0, 1.0)
        colormap = plt.get_cmap("Blues_r")
        n_visits = len(visited_patch_coords)
        for i, (x_lvl0, y_lvl0) in enumerate(visited_patch_coords):
            x_scaled = int(int(x_lvl0) / downscale_factor)
            y_scaled = int(int(y_lvl0) / downscale_factor)
            frac = i / max(1, n_visits - 1)
            color = tuple(int(255 * c) for c in colormap(norm(0.15 + 0.7 * frac))[:3])
            cv2.rectangle(
                wsi_np,
                (x_scaled, y_scaled),
                (x_scaled + box_w, y_scaled + box_w),
                color=color,
                thickness=3,
            )

    pil_img = Image.fromarray(wsi_np)
    draw = ImageDraw.Draw(pil_img)
    try:
        font = ImageFont.truetype("arial.ttf", 28)
    except Exception:
        font = ImageFont.load_default()

    title = f"{slide_name} | LR patches={len(all_lr_coords) if all_lr_coords is not None else 0}, RL-visited={len(visited_patch_coords) if visited_patch_coords is not None else 0}"
    if title_suffix:
        title = f"{title} | {title_suffix}"

    legend_lines = [
        title,
        "Gray boxes : all low-resolution patches",
        "Blue boxes : patches visited by RL agent (light -> dark = early -> late)",
    ]
    pad = 8
    line_h = 32
    for i, line in enumerate(legend_lines):
        y = pad + i * line_h
        draw.rectangle([(pad - 2, y - 2), (pad + 1100, y + line_h - 4)], fill=(255, 255, 255))
        draw.text((pad, y), line, fill=(0, 0, 0), font=font)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    pil_img.save(save_path)
    print(f"Saved patch visualization at: {save_path}")
    return save_path


@torch.no_grad()
def evaluate(conf):

    #df to get the true label
    bags_dataset = Dataset_All_Bags(conf.csv_path)
    df = bags_dataset.df.set_index('slide_id')
    
    # Loading RL agent
    agent_dict, config = load_agent(conf)
    agent_conf = SimpleNamespace(**config)
    actor = Actor(conf=agent_conf)
    critic  = Critic(conf=agent_conf)
    agent = Agent(actor, critic, agent_conf).to(conf.device)
    agent.load_state_dict(agent_dict)
    agent.to(conf.device)
    agent.eval()

    # Loading Classifier (feature aggregator + classifier)
    classifier = get_classifier(conf)

    # Loading cosine similarity based target state updater
    tsu = get_tsu(conf)

    # Initializing helper class for inference
    helper = Helper(conf, classifier)


    # Now load the train, validation, test slides
    split_file_path = './dataset_csv/%s/splits/split_%s.json' % (conf.dataset, conf.seed)
    if os.path.exists(split_file_path):
        with open(split_file_path, 'r') as json_file:
            data = json.load(json_file)
        train_names, val_names, test_names = data['train_names'], data['val_names'], data['test_names']

    else:
        raise Exception(f"Please enter a valid split seed for dataset - {conf.dataset} ")

    time_dict = defaultdict(list)
    y_pred = []
    y_true = []
    y_prob = []

    save_visualizations = bool(getattr(conf, 'save_visualizations', True))
    vis_dir = os.path.join(conf.save_dir, 'visualizations')
    if save_visualizations:
        os.makedirs(vis_dir, exist_ok=True)
    requested_vis_level = getattr(conf, 'vis_level', None)

    # Per-slide trajectory info (slide -> {all_coords, visited_coords, visited_ids, label, pred}).
    visit_log = {}

    for slide in test_names:

        start_time = time.time()

        # STEP1 PRE-PROCESSING AND CREATING PATCHES
        print(f"Evaluating slide: {slide}")
        print(f"Started Step1: Creating Patches")
        helper.create_patches(slide, conf.patch_level_low_res, conf.step_size, conf.patch_size)
        print(f"Completed Step1: Creating Patches")


        #STEP2 FEATURE EXTRACTION AT LOW RESOLUTION
        true_label = df.loc[slide]['label']
        print(f"Started Step2: Extraction Features")
        feature, coords, wsi = helper.extract_lr_features(slide, true_label, conf.slide_ext, conf.patch_level_low_res, conf.patch_level_high_res)
        print(f"Completed Step2: Extraction Features")


        # Step3 RL agent evaluation
        print(f"Started Step3: Sampling relevant patches using RL")
        state = torch.tensor(feature, dtype=torch.float32, device=conf.device).unsqueeze(0)
        true_label = torch.tensor([true_label]).to(conf.device)
        env = WSICosineObservationEnv_inference(lr_features=state, device=conf.device, frac_visit = conf.frac_visit, cosine_threshold= conf.cosine_threshold)
        N = state.shape[1]
        visited_patch_id = []
        visited_patch_coords = []
        done = False
        pbar = tqdm(total=int(conf.frac_visit*N), desc=f"{slide} RL Sampling Steps", leave=False)
        while not done:
            action, log_prob, entropy = agent.get_action(state, visited_patch_id, is_eval=True)
            visited_patch_id.append(action.item())

            # Zoom in to the selected action and extract featues for k patches in high resolution
            selected_coords = coords[action.item()]
            visited_patch_coords.append(selected_coords)
            v_at = helper.get_embedding(selected_coords, wsi, conf.patch_level_low_res, conf.patch_level_high_res, conf.step_size, conf.patch_size)
            state, done = env.step(action=action.item(),
                                       v_at=v_at,
                                       state_update_net=tsu,
                                       classifier_net=classifier,
                                       device=conf.device)
            pbar.update(1)

        pbar.close()
        slide_preds, attn = classifier.classify(state)
        pred = torch.softmax(slide_preds, dim=-1)
        y_hat = torch.argmax(pred)
        y_pred.append(y_hat.item())
        y_true.append(true_label)
        y_prob.append(pred)

        # Just to verify the values
        # print(y_pred)
        # print([y.item() for y in y_true])

        # STEP4 (optional) - per-slide visualization of all LR patches and the RL-visited patches.
        visit_log[slide] = {
            'all_lr_coords': np.asarray(coords),
            'visited_coords': np.asarray(visited_patch_coords),
            'visited_ids': list(visited_patch_id),
            'true_label': int(true_label.item()) if torch.is_tensor(true_label) else int(true_label),
            'pred_label': int(y_hat.item()),
        }
        if save_visualizations:
            try:
                vis_path = os.path.join(vis_dir, f"{slide}_patches.png")
                title_suffix = f"label={visit_log[slide]['true_label']}, pred={visit_log[slide]['pred_label']}"
                visualize_slide_patches(
                    wsi=wsi,
                    slide_name=slide,
                    all_lr_coords=coords,
                    visited_patch_coords=visited_patch_coords,
                    patch_size_low_res=conf.patch_size,
                    patch_level_low_res=conf.patch_level_low_res,
                    save_path=vis_path,
                    requested_vis_level=requested_vis_level,
                    title_suffix=title_suffix,
                )
            except Exception as e:
                print(f"[WARN] Could not save visualization for {slide}: {e}")

        end_time = time.time()

        time_dict[f'{slide}'].append(end_time - start_time)
        print(f"Slide time : {time_dict[f'{slide}']}")

    y_pred = torch.cat(y_prob, dim=0)
    y_true = torch.cat(y_true, dim=0)
    y_pred_labels = torch.argmax(y_pred, dim=-1)

    n_class = getattr(conf, 'n_class', None) or int(y_pred.shape[-1])
    accuracy = compute_accuracy(y_pred_labels, y_true, n_class)
    auroc = compute_auroc(y_pred, y_true, n_class)
    f1_score = compute_f1(y_pred_labels, y_true, n_class)
    precision = compute_precision(y_pred_labels, y_true, n_class)
    recall = compute_recall(y_pred_labels, y_true, n_class)
    balanced_acc = compute_balanced_accuracy(y_pred_labels, y_true)

    print(f"{'Phase':<6} | {'Acc':<6} | {'AUROC':<6} | {'F1':<6} | {'Precision':<9} | {'Recall':<6} | {'Balanced Acc':<13} ")
    print("-" * 110)
    print(f"{'Test':<6}  | {accuracy:.4f}  | {auroc:.4f}  | {f1_score:.4f}  | {precision:.4f}  | {recall:.4f}  | {balanced_acc:.4f}")

    # STEP5 - per-class breakdown (precision / recall / f1 / accuracy / support).
    class_names = getattr(conf, 'class_names', None)
    per_class_metrics = compute_per_class_classification_metrics(
        y_pred=y_pred,
        y_true=y_true,
        num_classes=n_class,
        class_names=class_names,
    )
    print()
    print(format_per_class_metrics_table(per_class_metrics, title="Per-class metrics (Test)"))

    metrics_dump = {
        'aggregate': {
            'accuracy': accuracy,
            'auroc': auroc,
            'f1': f1_score,
            'precision': precision,
            'recall': recall,
            'balanced_accuracy': balanced_acc,
        },
        'per_class': per_class_metrics,
        'n_class': n_class,
    }
    metrics_path = os.path.join(conf.save_dir, f"metrics_sasha_{conf.frac_visit}.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics_dump, f, indent=2)
    print(f"Saved metrics summary at: {metrics_path}")

    total_time = []
    for key in time_dict.keys():
        total_time.append(sum(time_dict[key]))
    print(f"Average : {sum(total_time) / len(total_time)}")

    torch.save(time_dict, os.path.join(conf.save_dir, f"time_dict_sasha_{conf.frac_visit}.pt"))

    # STEP6 - dump the per-slide RL trajectory log (all LR coords + visited coords) for downstream analysis.
    visit_log_path = os.path.join(conf.save_dir, f"visit_log_sasha_{conf.frac_visit}.pt")
    torch.save(visit_log, visit_log_path)
    print(f"Saved per-slide RL visit log at: {visit_log_path}")


if __name__ == '__main__':

    args = get_arguments()

    with open(args.config, 'r') as ymlfile:
        c = yaml.load(ymlfile, Loader=yaml.FullLoader)
        c.update(vars(args))
        conf = Struct(**c)

    resolve_conf_paths(conf, ['data_h5_dir', 'source', 'csv_path', 'classifier_ckpt_path', 'mlp_fglobal_ckpt', 'rl_ckpt_path', 'save_dir'], base_dir=os.getcwd())
    ensure_path_exists(conf.data_h5_dir, 'data_h5_dir', expect_dir=True)
    ensure_path_exists(conf.source, 'source', expect_dir=True)
    ensure_path_exists(conf.csv_path, 'csv_path', expect_dir=False)
    ensure_path_exists(conf.classifier_ckpt_path, 'classifier_ckpt_path', expect_dir=False)
    ensure_path_exists(conf.mlp_fglobal_ckpt, 'mlp_fglobal_ckpt', expect_dir=False)
    ensure_path_exists(conf.rl_ckpt_path, 'rl_ckpt_path', expect_dir=False)
    os.makedirs(conf.save_dir, exist_ok=True)

    evaluate(conf)