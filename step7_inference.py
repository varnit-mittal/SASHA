"""
Here we are not going with the feature extraction at each iteration. We are utilizing the intermediate feature we have saved

Description:
This script is used to run inference using the SASHA pipeline (versions 0.1 and 0.2).
It assumes that all the required components — HAFED, TSU, and RL — have been trained beforehand.

Specifically, this script loads the trained models, performs inference over the target dataset,
and generates predictions according to the specified SASHA configuration.

Supported SASHA Variants:
- SASHA-0.1
- SASHA-0.2

Make sure to configure the required paths to checkpoints and datasets before execution.

CAMELYON16
python step7_inference.py --config config_prince/camelyon_sasha_inference.yml --seed 4

TCGA
python step7_inference.py --config config_prince/tcga_sasha_inference.yml --seed 1

"""

import argparse
import os
from pprint import pprint
from types import SimpleNamespace

import torch
import yaml
from torch.utils.data import DataLoader

from architecture.transformer import HAFED
from datasets.datasets import build_HDF5_feat_dataset_2
from envs.WSI_cosine_env import WSICosineObservationEnv
from envs.WSI_env import WSIObservationEnv
from modules.fglobal_mlp import FGlobal
from rl_algorithms.ppo import Agent, Actor, Critic
from step4_extract_intermediate_features import load_model
from utils.gpu_utils import check_gpu_availability
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
from utils.utils import MetricLogger
from utils.utils import Struct, set_seed


def get_arguments():
    parser = argparse.ArgumentParser('SASHA inference', add_help=False)
    parser.add_argument( '--config', default= None, help='path to config file')
    parser.add_argument('--seed', type=int, default= 4, help='set the random seed')
    parser.add_argument('--classifier_arch', default='hafed', choices=['hafed'], help='choice of architecture for HACMIL')
    parser.add_argument('--exp_name', type=str, default='DEBUG', help='name of the exp')
    parser.add_argument('--logs', default='enabled', choices=['enabled', 'disabled'], type=str, help='flag to save logs')
    args = parser.parse_args()

    # Adding Device Details
    gpus = check_gpu_availability(3, 1, [])
    print(f"occupied {gpus}")
    args.device = torch.device(f"cuda:{gpus[0]}")

    return args

def load_policy_model(model, actor_optimizer, critic_optimizer, load_path, device="cpu"):

    # Load the checkpoint
    checkpoint = torch.load(load_path, map_location=device)

    # Load model weights
    model.load_state_dict(checkpoint['model'])

    # Load optimizer states
    actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
    critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])

    # Get epoch number and config
    epoch = checkpoint['epoch']
    config = checkpoint['config']

    print(f"Model loaded from {load_path} at epoch {epoch}")

    return model, actor_optimizer, critic_optimizer, epoch, config

def main():
    # getting and config file
    args = get_arguments()

    with open(args.config, 'r') as ymlfile:
        c = yaml.load(ymlfile, Loader=yaml.FullLoader)
        c.update(vars(args))
        conf = Struct(**c)

    resolve_conf_paths(conf, ['level1_path', 'level3_path', 'classifier_ckpt_path', 'mlp_fglobal_ckpt', 'rl_ckpt_path'], base_dir=os.getcwd())
    ensure_path_exists(conf.level1_path, 'level1_path', expect_dir=True)
    ensure_path_exists(conf.level3_path, 'level3_path', expect_dir=True)
    ensure_path_exists(conf.classifier_ckpt_path, 'classifier_ckpt_path', expect_dir=False)
    ensure_path_exists(conf.mlp_fglobal_ckpt, 'mlp_fglobal_ckpt', expect_dir=False)
    ensure_path_exists(conf.rl_ckpt_path, 'rl_ckpt_path', expect_dir=False)

    hyparams = {
        'dataset': conf.dataset,
        'pretrain': conf.pretrain,
        'classifier_arch': conf.classifier_arch,
        'seed': conf.seed,
        'frac_visit': conf.frac_visit,
        'only_ce_as_reward': conf.only_ce_as_reward,
    }
    hyparams['fraction of visit'] = conf.frac_visit

    print("Used config:")
    pprint(vars(conf))

    # Loading seed
    set_seed(args.seed)

    # create dataloaders
    train_data, val_data, test_data = build_HDF5_feat_dataset_2(conf.level1_path, conf.level3_path, conf)
    train_loader = DataLoader(train_data, batch_size=conf.B, shuffle=True, num_workers=conf.n_worker, pin_memory=conf.pin_memory, drop_last=True)
    val_loader = DataLoader(val_data, batch_size=conf.B, shuffle=False, num_workers=conf.n_worker, pin_memory=conf.pin_memory, drop_last=False)
    test_loader = DataLoader(test_data, batch_size=conf.B, shuffle=False, num_workers=conf.n_worker, pin_memory=conf.pin_memory, drop_last=False)

    # loading classifier
    classifier_dict, _, config, _ = load_model(conf.classifier_ckpt_path, args)
    classifier_conf = SimpleNamespace(**config)

    if conf.classifier_arch == 'hafed':
        classifier = HAFED(classifier_conf, n_token_1=classifier_conf.n_token_1,
                           n_token_2=classifier_conf.n_token_2, n_masked_patch_1=classifier_conf.n_masked_patch_1,
                           n_masked_patch_2=classifier_conf.n_masked_patch_2, mask_drop=classifier_conf.mask_drop)
    else :
        raise Exception("Select a valid classifier architecture.")

    classifier.to(conf.device)
    classifier.load_state_dict(classifier_dict)
    classifier.eval()

    # Loading TSU
    fglobal_dict = torch.load(conf.mlp_fglobal_ckpt, map_location=conf.device)
    fglobal = FGlobal(ip_dim=384 * 3, op_dim=384).to(conf.device)
    fglobal.load_state_dict(fglobal_dict['model'])
    fglobal.eval()

    # Loading RL Agent
    actor = Actor(conf=conf)
    critic = Critic(conf=conf)
    model = Agent(actor, critic, conf).to(conf.device)
    actor_optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, actor.parameters()), lr=0.001)
    critic_optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, critic.parameters()), lr=0.001)
    model, actor_optimizer, critic_optimizer, epoch, rl_config = load_policy_model(model, actor_optimizer, critic_optimizer, conf.rl_ckpt_path, conf.device)


    # Resolve human-readable class names from config (falls back to class_0, class_1, ...).
    class_names = getattr(conf, 'class_names', None)
    if class_names is None:
        class_names = [f"class_{i}" for i in range(conf.n_class)]

    # Now Evaluation Starts
    # Step 1 ----> We are evaluating the HAFED model [ All patches in H.R.]
    print("HAFED Inference")
    train_results = evaluate_hafed(classifier, train_loader, 'Train', conf.device, conf.n_class, class_names)
    val_results = evaluate_hafed(classifier, val_loader, 'Val', conf.device, conf.n_class, class_names)
    test_results = evaluate_hafed(classifier, test_loader, 'Test', conf.device, conf.n_class, class_names)

    _print_results_table("HAFED", [train_results, val_results, test_results], class_names)

    ################# Breaker

    print("\n" + "=" * 120)
    print("SASHA - Deterministic Policy {Pick Max}")
    train_results = evaluate_policy(model, fglobal, classifier, train_loader, 'Train', conf.device, epoch, conf, class_names)
    val_results = evaluate_policy(model, fglobal, classifier, val_loader, 'Val', conf.device, epoch, conf, class_names)
    test_results = evaluate_policy(model, fglobal, classifier, test_loader, 'Test', conf.device, epoch, conf, class_names)

    _print_results_table("SASHA", [train_results, val_results, test_results], class_names)


def _compute_per_class_counts(y_pred_labels, y_true, slide_names, n_class, class_names):
    """Compute correct/incorrect counts for every class."""
    y_pred_np = y_pred_labels.cpu().numpy()
    y_true_np = y_true.cpu().numpy()

    per_class = {}
    for cls_idx in range(n_class):
        cls_name = class_names[cls_idx]
        indices = [i for i, val in enumerate(y_true_np) if val == cls_idx]
        correct = [slide_names[i] for i in indices if y_pred_np[i] == y_true_np[i]]
        incorrect = [slide_names[i] for i in indices if y_pred_np[i] != y_true_np[i]]
        per_class[cls_name] = {'correct': correct, 'incorrect': incorrect, 'total': len(indices)}
    return per_class


def _print_results_table(method_name, results_list, class_names):
    """Print an aggregate metrics table plus a per-class correct/incorrect table."""
    phase_labels = ['Train', 'Val', 'Test']

    # Aggregate table
    header = f"{'Phase':<6} | {'Acc':<6} | {'AUROC':<6} | {'F1':<6} | {'Precision':<9} | {'Recall':<6} | {'Balanced Acc':<12}"
    print(f"\n{method_name} - Aggregate Metrics")
    print(header)
    print("-" * len(header))
    for phase, res in zip(phase_labels, results_list):
        print(f"{phase:<6} | {res['accuracy']:.4f} | {res['auroc']:.4f} | {res['f1']:.4f} | {res['precision']:.4f}    | {res['recall']:.4f} | {res['balanced_acc']:.4f}")

    # Per-class table
    cls_header_parts = [f"{'Phase':<6}"]
    for cn in class_names:
        cls_header_parts.append(f"{cn} Correct")
        cls_header_parts.append(f"{cn} Wrong")
    cls_header = " | ".join(cls_header_parts)
    print(f"\n{method_name} - Per-class Correct / Incorrect")
    print(cls_header)
    print("-" * len(cls_header))
    for phase, res in zip(phase_labels, results_list):
        parts = [f"{phase:<6}"]
        for cn in class_names:
            c = len(res['per_class'][cn]['correct'])
            w = len(res['per_class'][cn]['incorrect'])
            parts.append(f"{c:<{len(cn)+8}}")
            parts.append(f"{w:<{len(cn)+6}}")
        print(" | ".join(parts))

    # Detailed per-class precision / recall / f1 table (test set only)
    test_res = results_list[2]
    print()
    print(format_per_class_metrics_table(test_res['per_class_metrics'], title=f"{method_name} - Per-class Metrics (Test)"))


@torch.no_grad()
def evaluate_policy(model, fglobal, classifier, data_loader, header, device, epoch, conf, class_names, is_eval=True, is_top_k=False, is_top_p=False, seed=1):

    model.eval()

    y_pred = []
    y_true = []
    slide_names = []
    metric_logger = MetricLogger(delimiter=" ")
    final_reward = 0
    patches_idx = {}

    for data in metric_logger.log_every(data_loader, 100, header):
        hr_features = data['hr'][0].to(device, dtype=torch.float32)
        state = data['lr'].to(device, dtype=torch.float32)
        slide_name = data['slide_name'][0]
        label = data['label'].to(device)

        if conf.fglobal == 'attn':
            env = WSIObservationEnv(lr_features=state, hr_features=hr_features, label=label, conf=conf)
        else:
            env = WSICosineObservationEnv(lr_features=state, hr_features=hr_features, label=label, conf=conf)

        N = state.shape[1]
        visited_patch_id = []
        done = False

        while not done:
            action, _, _ = model.get_action(state, visited_patch_id, is_eval=is_eval, is_top_k=is_top_k, is_top_p=is_top_p)
            new_state, reward, done = env.step(action=action, state_update_net=fglobal, classifier_net=classifier, device=device)
            state = new_state
            visited_patch_id.append(action.item())

        final_reward += reward
        slide_preds, attn = classifier.classify(state)
        pred = torch.softmax(slide_preds, dim=-1)

        y_pred.append(pred)
        y_true.append(label)
        slide_names.append(slide_name)
        patches_idx[slide_name] = visited_patch_id.copy()

    y_pred = torch.cat(y_pred, dim=0)
    y_true = torch.cat(y_true, dim=0)
    y_pred_labels = torch.argmax(y_pred, dim=-1)

    accuracy = compute_accuracy(y_pred_labels, y_true, conf.n_class)
    auroc = compute_auroc(y_pred, y_true, conf.n_class)
    f1_score = compute_f1(y_pred_labels, y_true, conf.n_class)
    precision = compute_precision(y_pred_labels, y_true, conf.n_class)
    recall = compute_recall(y_pred_labels, y_true, conf.n_class)
    balanced_acc = compute_balanced_accuracy(y_pred_labels, y_true)

    per_class = _compute_per_class_counts(y_pred_labels, y_true, slide_names, conf.n_class, class_names)
    per_class_metrics = compute_per_class_classification_metrics(y_pred, y_true, conf.n_class, class_names)

    return {
        'accuracy': accuracy, 'auroc': auroc, 'f1': f1_score,
        'precision': precision, 'recall': recall, 'balanced_acc': balanced_acc,
        'per_class': per_class, 'per_class_metrics': per_class_metrics,
    }


@torch.no_grad()
def evaluate_hafed(model, data_loader, header, device, n_class, class_names):

    model.eval()

    y_pred = []
    y_true = []
    slide_names = []
    metric_logger = MetricLogger(delimiter="  ")

    for data in metric_logger.log_every(data_loader, 100, header):

        hr_features = data['hr'].to(device, dtype=torch.float32)
        slide_name = data['slide_name'][0]
        label = data['label'].to(device)

        slide_preds, attn = model.classify(hr_features)
        pred = torch.softmax(slide_preds, dim=-1)

        y_pred.append(pred)
        y_true.append(label)
        slide_names.append(slide_name)

    y_pred = torch.cat(y_pred, dim=0)
    y_true = torch.cat(y_true, dim=0)
    y_pred_labels = torch.argmax(y_pred, dim=-1)

    accuracy = compute_accuracy(y_pred_labels, y_true, n_class)
    auroc = compute_auroc(y_pred, y_true, n_class)
    f1_score = compute_f1(y_pred_labels, y_true, n_class)
    precision = compute_precision(y_pred_labels, y_true, n_class)
    recall = compute_recall(y_pred_labels, y_true, n_class)
    balanced_acc = compute_balanced_accuracy(y_pred_labels, y_true)

    per_class = _compute_per_class_counts(y_pred_labels, y_true, slide_names, n_class, class_names)
    per_class_metrics = compute_per_class_classification_metrics(y_pred, y_true, n_class, class_names)

    return {
        'accuracy': accuracy, 'auroc': auroc, 'f1': f1_score,
        'precision': precision, 'recall': recall, 'balanced_acc': balanced_acc,
        'per_class': per_class, 'per_class_metrics': per_class_metrics,
    }


if __name__ == '__main__':
    main()