
# SASHA - Sequential Attention-based Sampling for Histopathological Analysis

![Semantic Diagram](images/semantic_dig.png)

![Details for HAFED + TSU models](images/hafed_tsu_dig.png)

## Paper Link 

Available at : https://arxiv.org/abs/2507.05077

## Requirements

To install requirements:

```setup
conda create --name <env> --file conda-packages.txt
```

## Train Directly From NAS (Streaming)

You can train without copying data locally by pointing config paths to your NAS share.
On Windows, use UNC paths like `\\172.16.202.70\YOUR_SHARE\...`.

1. Set an optional NAS root once:

```powershell
$env:SASHA_NAS_ROOT="\\172.16.202.70\YOUR_SHARE"
```

2. In config files (for example `config/camelyon_config.yml`), set:

```yaml
nas_root: \\172.16.202.70\YOUR_SHARE
data_dir: features/lr/h5_files
```

If `nas_root` is set, relative paths like `features/lr/h5_files` are resolved under that NAS root.
Absolute paths still work as-is.

3. Run training normally:

```powershell
python step3_WSI_classification_HAFED.py --config config/camelyon_config.yml --seed 4 --arch hafed --exp_name DEBUG --log_dir outputs/camelyon_hafed
python step4_extract_intermediate_features.py --config config/camelyon_config.yml --seed 4 --arch hafed --ckpt_path outputs/camelyon_hafed/models/DEBUG/checkpoint-best.pt --output_path features/hr_intermediate
python step5_tsu_training.py --config config/camelyon_tsu_config.yml --seed 4 --arch hafed --log_dir outputs/camelyon_tsu
python step6_rl_training.py --config config/camelyon_rl_config.yml --seed 4 --log_dir outputs/camelyon_rl
```

4. Optional mapped-drive setup (if preferred):

```powershell
net use Z: \\172.16.202.70\YOUR_SHARE /persistent:yes
```

Then use paths like `Z:\features\lr\h5_files` in configs.

### Ubuntu Setup

If you run on Ubuntu and your SMB share name is `home` with dataset directory `/mnt/nas/Dataset`, use the exact commands below.

1. Create NAS credentials file (outside repo):

```bash
mkdir -p ~/.smb
cat > ~/.smb/nas-172.16.202.70.cred << 'EOF'
username=varnitm
password=YOUR_NAS_PASSWORD
domain=WORKGROUP
EOF
chmod 600 ~/.smb/nas-172.16.202.70.cred
```

2. Mount NAS share `home` and verify `Dataset`:

```bash
sudo mkdir -p /mnt/nas
sudo mount -t cifs //172.16.202.70/home /mnt/nas -o credentials=$HOME/.smb/nas-172.16.202.70.cred,uid=$(id -u),gid=$(id -g),file_mode=0644,dir_mode=0755,iocharset=utf8,vers=3.0,sec=ntlmssp
ls -la /mnt/nas/Dataset
```

3. Create project `.env`:

```bash
cat > .env << 'EOF'
SASHA_NAS_ROOT=/mnt/nas/Dataset
SASHA_SOURCE_DIR=/mnt/nas/Dataset/raw_wsi
SASHA_SAVE_DIR=/mnt/nas/Dataset/sasha_outputs/step1
SASHA_FEAT_DIR=/mnt/nas/Dataset/sasha_outputs/features
SASHA_LOG_DIR=/mnt/nas/Dataset/sasha_outputs/logs
EOF
```

4. Load `.env` in current shell:

```bash
set -a
source .env
set +a
```

5. Generate NAS-specific config files:

```bash
python - << 'PY'
import os
import yaml

def load_yaml(path):
	with open(path, 'r', encoding='utf-8') as f:
		return yaml.safe_load(f)

def dump_yaml(path, data):
	with open(path, 'w', encoding='utf-8') as f:
		yaml.safe_dump(data, f, sort_keys=False)

nas_root = os.environ['SASHA_NAS_ROOT']
feat_dir = os.environ['SASHA_FEAT_DIR']
log_dir = os.environ['SASHA_LOG_DIR']

c3 = load_yaml('config/camelyon_config.yml')
c3['nas_root'] = nas_root
c3['data_dir'] = 'sasha_outputs/features/hr/h5_files'
dump_yaml('config/camelyon_config_nas.yml', c3)

c5 = load_yaml('config/camelyon_tsu_config.yml')
c5['nas_root'] = nas_root
c5['level1_path'] = 'sasha_outputs/features/intermediate_hafed'
c5['level3_path'] = 'sasha_outputs/features/lr/h5_files'
dump_yaml('config/camelyon_tsu_config_nas.yml', c5)

c6 = load_yaml('config/camelyon_rl_config.yml')
c6['nas_root'] = nas_root
c6['classifier_ckpt_path'] = f"{log_dir}/camelyon_hafed/models/DEBUG/checkpoint-best.pt"
c6['mlp_fglobal_ckpt'] = f"{log_dir}/camelyon_tsu/models/DEBUG/checkpoint-best.pt"
c6['level1_path'] = 'sasha_outputs/features/intermediate_hafed'
c6['level3_path'] = 'sasha_outputs/features/lr/h5_files'
dump_yaml('config/camelyon_rl_config_nas.yml', c6)
PY
```

6. Train full CAMELYON pipeline on NAS paths:

```bash
python step1_create_patches.py --source "$SASHA_SOURCE_DIR" --save_dir "$SASHA_SAVE_DIR" --extension tif --patch_level 3

python step2_extract_features.py --dataset_name camelyon16 --data_h5_dir "$SASHA_SAVE_DIR" --data_slide_dir "$SASHA_SOURCE_DIR" --slide_ext .tif --csv_path dataset_csv/camelyon16/camelyon16.csv --feat_dir "$SASHA_FEAT_DIR" --batch_size 32 --extract_high_res_features True --patch_level_low_res 3 --patch_level_high_res 1

python step3_WSI_classification_HAFED.py --config config/camelyon_config_nas.yml --seed 4 --arch hafed --exp_name DEBUG --log_dir "$SASHA_LOG_DIR/camelyon_hafed"

python step4_extract_intermediate_features.py --config config/camelyon_config_nas.yml --seed 4 --arch hafed --ckpt_path "$SASHA_LOG_DIR/camelyon_hafed/models/DEBUG/checkpoint-best.pt" --output_path "$SASHA_FEAT_DIR/intermediate_hafed"

python step5_tsu_training.py --config config/camelyon_tsu_config_nas.yml --seed 4 --arch hafed --log_dir "$SASHA_LOG_DIR/camelyon_tsu"

python step6_rl_training.py --config config/camelyon_rl_config_nas.yml --seed 4 --log_dir "$SASHA_LOG_DIR/camelyon_rl"
```

Note: this assumes your raw slides are under `/mnt/nas/Dataset/raw_wsi`.

## Training

To train the model(s) in the paper, run this command:

STEP 1

First step to process is to WSI and segment it and remove the region where the tissue sample for
a patch is less than some threshold.

For CAMELYON16 dataset
```train
python step1_create_patches.py --source SOURCE_DIR --save_dir SAVE_DIR --extension tif --patch_level 3
```

FOR TCGA-NSCLC dataset
```train
python step1_create_patches.py --source SOURCE_DIR --save_dir SAVE_DIR --extension svs --patch_level 2
```

FOR custom 3-class glioma subtype dataset (`glioma3`)
```train
python step1_create_patches.py --source SOURCE_DIR --save_dir SAVE_DIR --extension svs --patch_level 2
```

STEP 2
This step handles feature extraction.
There are two modes of operation:

1. If `extract_high_res_features` is set to `True`, the feature extractor will generate features for both high-resolution and low-resolution patches.

2. If `extract_high_res_features` is set to `False`, features will be extracted only for low-resolution patches.


For extraction both high resolution and low resolution together
```train
python step2_extract_features.py --dataset_name camelyon16 --data_h5_dir SAVE_DIR_PATH_FROM_PATCH_CREATION --data_slide_dir WSI_IMAGES_DIR --slide_ext .tif --csv_path dataset_csv/camelyon16/camelyon16.csv --feat_dir FEAT_DIR_TO_SAVE --batch_size 32 --extract_high_res_features True --patch_level_low_res 3 --patch_level_high_res 1
```

For only low resolution
```train
python step2_extract_features.py --dataset_name camelyon16 --data_h5_dir SAVE_DIR_PATH_FROM_PATCH_CREATION --data_slide_dir WSI_IMAGES_DIR --slide_ext .tif --csv_path dataset_csv/camelyon16/camelyon16.csv --feat_dir FEAT_DIR_TO_SAVE --batch_size 512 --extract_high_res_features False --patch_level_low_res 3 --patch_level_high_res 1
```

Resume from an interrupted step2 run:
```train
python step2_extract_features.py --dataset_name camelyon16 --data_h5_dir SAVE_DIR_PATH_FROM_PATCH_CREATION --data_slide_dir WSI_IMAGES_DIR --slide_ext .tif --csv_path dataset_csv/camelyon16/camelyon16.csv --feat_dir FEAT_DIR_TO_SAVE --batch_size 32 --extract_high_res_features True --patch_level_low_res 3 --patch_level_high_res 1 --resume
```

STEP 3 

This script is used to train the models required to obtain the Feature Aggregator and Classifier components of HAFED.

These trained models are utilized in the subsequent stage of the pipeline.

Model Architecture:
- Feature Aggregator: Input shape (k × d) → Output shape (d)
- Classifier: Input shape (N × d) → Output: predicted class probabilities (ŷ)

For training of HAFED
```train
python step3_WSI_classification_HAFED.py --config config/camelyon_config.yml --seed 4 --arch hafed --exp_name DEBUG --log_dir LOG_DIR
```

STEP 4 

Once the HAFED model has been trained using all high-resolution patches,
the workflow proceeds as follows:

- Input: (N × k × d)
- Intermediate output (after feature aggregation): (N × d)

```train
python step4_extract_intermediate_features.py --config config/camelyon_config.yml --seed 4 --arch hafed --ckpt_path CKPT_PATH --output_path OUTPUT_PATH
```

STEP 5

The primary objective is to propagate feature information from selected patches to their similar counterparts,
based on a cosine similarity threshold.


```train
python step5_tsu_training.py --config config/camelyon_tsu_config.yml --seed 4 --arch hafed --log_dir LOG_DIR
```

STEP 6 

This script initiates the training of the RL Agent component after separately training the HAFED,
which includes the Feature Aggregator, Classifier and TSU.

```train
python step6_rl_training.py --config config/camelyon_rl_config.yml --seed 4  --log_dir LOG_DIR
```


## Evaluation

To evaluate for SASHA-0.1 and SASHA-0.2, with utilizing the features extracted in STEP 2 and STEP4:

```eval
python step7_inference.py --config config_prince/camelyon_sasha_inference.yml --seed 4
```

Evaluate model with feature extraction 

```eval
python step7_inference_with_fe.py --config config/camelyon_sasha_inference_with_fe.yml --seed 4 --save_dir SAVE_DIR
```

## Step 10 ROI Annotation (Doctor Review)

Step 10 creates a contour-style ROI overlay directly on the original WSI (instead of rectangular ROI boxes) and exports ROI coordinates.

```eval
python step10_roi_annotation.py --config config/camelyon_sasha_inference.yml --slide_name test_068 --ext tif --wsi_images_dir_path "$SASHA_SOURCE_DIR" --output_dir_path /mnt/nas/Dataset/sasha_outputs/roi_annotations/test_068 --seed 1
```

Outputs:
- `<slide_name>_roi_overlay.png` : WSI with ROI contour overlays
- `<slide_name>_roi.csv` : ROI coordinates in level-0 pixels
- `<slide_name>_roi.json` : ROI metadata

Useful tuning flags:
- `--roi_percentile 85` : higher value keeps only higher-suspicion regions.
- `--min_roi_patches 8` : minimum connected patches needed to keep an ROI.
- `--contour_alpha 0.30` : transparency for filled contour overlay.
- `--contour_thickness 3` : line thickness for contour boundaries.

Note : For TCGA-NSCL dataset similar config files are present in config/ folder.

## Multiclass classification (3-class glioma subtype)

The pipeline now ships with a 3-class glioma subtype dataset key called
`glioma3` (`subtype_1`/`subtype_2`/`subtype_3` mapped to `0/1/2`). All training
and inference scripts are class-count agnostic — they pick up `n_class` from
the YAML config and use macro-averaged multiclass metrics whenever `n_class > 2`.

### 1. Build the dataset CSV + stratified splits

If you have a raw labels file with two columns (`slide_id,label`) where `label`
is one of `subtype_1`, `subtype_2`, `subtype_3`, generate the SASHA-style CSV
and 5-fold splits with:

```bash
python scripts/build_glioma3_dataset.py \
    --raw_csv /path/to/slide_labels.csv \
    --output_dir dataset_csv/glioma3 \
    --seeds 1 2 3 4 5
```

This writes:
- `dataset_csv/glioma3/glioma3.csv` (`case_id,slide_id,label`, `slide_id` ends in `.svs`).
- `dataset_csv/glioma3/splits/split_<seed>.json` and `split_<seed>_summary.txt`.

### 2. Patch and feature extraction (.svs)

```bash
python step1_create_patches.py --source SOURCE_DIR --save_dir SAVE_DIR --extension svs --patch_level 2

python step2_extract_features.py \
    --dataset_name glioma3 \
    --data_h5_dir SAVE_DIR \
    --data_slide_dir WSI_IMAGES_DIR \
    --slide_ext .svs \
    --csv_path dataset_csv/glioma3/glioma3.csv \
    --feat_dir FEAT_DIR_TO_SAVE \
    --batch_size 32 \
    --extract_high_res_features True \
    --patch_level_low_res 2 \
    --patch_level_high_res 1
```

### 3. HAFED training and intermediate features (3-class)

Edit `config/glioma3_config.yml` and set `data_dir` to `<FEAT_DIR_TO_SAVE>/hr/h5_files`.

```bash
python step3_WSI_classification_HAFED.py \
    --config config/glioma3_config.yml \
    --seed 1 \
    --arch hafed \
    --exp_name DEBUG \
    --log_dir outputs/glioma3_hafed

python step4_extract_intermediate_features.py \
    --config config/glioma3_config.yml \
    --seed 1 \
    --arch hafed \
    --ckpt_path outputs/glioma3_hafed/models/DEBUG/checkpoint-best.pt \
    --output_path features/glioma3_hr_intermediate
```

### 4. TSU + RL training

Edit `config/glioma3_tsu_config.yml` and `config/glioma3_rl_config.yml` so:
- `level1_path` points to the directory written by step4
  (`features/glioma3_hr_intermediate`).
- `level3_path` points to `<FEAT_DIR_TO_SAVE>/lr/h5_files`.
- `classifier_ckpt_path` and `mlp_fglobal_ckpt` point to the matching
  checkpoint files when running step6/step7.

```bash
python step5_tsu_training.py \
    --config config/glioma3_tsu_config.yml \
    --seed 1 \
    --arch hafed \
    --log_dir outputs/glioma3_tsu

python step6_rl_training.py \
    --config config/glioma3_rl_config.yml \
    --seed 1 \
    --log_dir outputs/glioma3_rl
```

### 5. SASHA inference

```bash
python step7_inference.py --config config/glioma3_sasha_inference.yml --seed 1
```

For end-to-end inference (feature extraction + sampling) on new `.svs`:

```bash
python step7_inference_with_fe.py \
    --config config/glioma3_sasha_inference_with_fe.yml \
    --seed 1 \
    --save_dir outputs/glioma3_sasha_runs
```

### Multiclass metric notes

`utils/metrics.py` centralises the AUROC / F1 / Precision / Recall / Accuracy
helpers used by step3, step6, step7 and `engine.py`:

- For binary problems (`n_class == 2`) it preserves the original
  `task='binary'` formulation, so existing CAMELYON/TCGA numbers are
  unchanged.
- For multiclass problems (`n_class >= 3`) it uses
  `task='multiclass', average='macro'` and feeds AUROC the full
  probability matrix instead of `y_pred[:, 1]`.


## Results

Our model achieves the following performance on CAMELYON16 (C16) and TCGA-NSCLC (TCGA) dataset


| Sampling | Method    | Accuracy (C16)    | AUC (C16)         | F1 (C16)          | Accuracy (TCGA)   | AUC (TCGA)        | F1 (TCGA)         |
| -------- | --------- | ----------------- | ----------------- | ----------------- |-------------------|-------------------|-------------------|
| 100%     | **HAFED** | **0.963 ± 0.008** | **0.980 ± 0.003** | **0.951 ± 0.011** | **0.923 ± 0.011** | **0.966 ± 0.015** | **0.925 ± 0.010** |
| 10%      | **SASHA-0.1** | **0.901 ± 0.021** | **0.918 ± 0.014** | **0.856 ± 0.031** | **0.897 ± 0.023** | **0.956 ± 0.023** | **0.898 ± 0.024** |
| 20%      | **SASHA-0.2** | **0.953 ± 0.017** | **0.979 ± 0.008** | **0.937 ± 0.024** | **0.912 ± 0.010** | **0.963 ± 0.014** | **0.914 ± 0.011** |

## Contributing

Creative Commons Attribution-NonCommercial 4.0 International

This work is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License.  
To view a copy of this license, visit https://creativecommons.org/licenses/by-nc/4.0/.

You are free to:
- Share — copy and redistribute the material in any medium or format
- Adapt — remix, transform, and build upon the material

Under the following terms:
- Attribution — You must give appropriate credit, provide a link to the license, and indicate if changes were made.
- NonCommercial — You may not use the material for commercial purposes.
- No additional restrictions — You may not apply legal terms or technological measures that legally restrict others from doing anything the license permits.

For more information, see https://creativecommons.org/licenses/by-nc/4.0/