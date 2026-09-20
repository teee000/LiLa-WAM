# LiLa-WAM: Lightweight Latent Reasoning World-Action Model for Robot Manipulation

Official implementation of **LiLa-WAM**, a lightweight world-action model that is trainable on a **single consumer-grade GPU (24 GB)**.

&gt; 📄 Paper: [arXiv](https://arxiv.org/pdf/2608.03701). | 🌐 Project Page: [teee000.github.io/LiLa-WAM-page](https://teee000.github.io/LiLa-WAM-page/)

## Overview

LiLa-WAM achieves an average success rate of **90.48%** on the 50 RoboTwin 2.0 tasks under the clean setting, while remaining trainable on a **single RTX 5090** (~110 GPU hours for joint training on all 50 tasks). The full model contains **0.5B parameters**, of which only 0.2B are trainable.

Key features:

- 🪶 **Lightweight**: 0.5B parameters (0.2B trainable), trainable on a single 24 GB GPU
- 🔮 **World-action modeling**: latent future-state prediction coupled with action generation
- 🎯 **Visual Transition Tokens (VTT)**
- 🤖 **Evaluated on**: RoboTwin 2.0 (50 tasks), LIBERO, and real-robot experiments

<p align="center">
  <img src="assets/compare_robotwin.png" width="48%"/>
  <img src="assets/compare_libero.png" width="48%"/>
</p>
<p align="center">
  <em>Success rate vs. model parameters on RoboTwin 2.0 (left) and LIBERO (right).
  Bubble size denotes the number of parameters.</em>
</p>


<p align="center">
  <img src="assets/FrameWork.png" width="95%"/>
</p>
<p align="center">
  <em>LiLa-WAM Framework.</em>
</p>

## Installation


### Setup

1. Create and activate the conda environment:

```bash
conda create -n LiLaWAM python=3.10 -y
conda activate LiLaWAM
```

2. Install PyTorch (CUDA 12.8):

```bash
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
```

3. Install other dependencies:

```bash
pip install transformers==5.0.0rc0 omegaconf accelerate h5py
```

## Preparation

Our processed RoboTwin 2.0 dataset is available on ModelScope. Search for **`LiLa-WAM_RoboTwin2.0_50_task`** on [ModelScope](https://www.modelscope.cn) to download.

For **LIBERO**, we provide the dataset converted to **HDF5** format. Download it from [Libero](https://modelscope.cn/models/yangfan97/LiLa-WAM_Libero) on ModelScope.

LiLa-WAM uses the frozen **DINOv3 ViT-L/16** encoder:

- Model: `dinov3-vitl16-pretrain-lvd1689m`

- **`utils/clean_dataset_stationary.py`**
  Removes stationary segments (steps where the robot barely moves) from raw demonstrations.

### Normalization Statistics

- **`utils/calc_stat_remove_outlier.py`**
  Computes normalization statistics over the training dataset with outlier removal.
- **`utils/stat-500-all.json`**
  Pre-computed normalization statistics covering all 50 tasks, so you can start training directly without re-computing them.

## Configuration

Before training or evaluation, update the following paths in the configuration files under `configs/`:

| Field | Description |
|---|---|
| `dataset_dir` | Path to the training dataset |
| `task_cond_dir` | Path to the VTT embedding files |
| `model.vision_encoder.checkpoint_path` | Path to the pretrained DINOv3 checkpoint |


## Training Details (Two-Stage Learning Rate Schedule)

The default config (`configs/robotwin_all.yaml`) sets a total of **40 epochs** with a learning rate of **2e-4**. In practice, however, you do **not** need to train for all 40 epochs. We recommend a two-stage schedule:

**Stage 1 — Base training (LR = 2e-4):** Train for about **11–12 epochs**, then stop early. There is no need to complete the full 40 epochs.

```bash
python train.py --config ./configs/robotwin_all.yaml
```

**Stage 2 — Fine-tuning (LR = 4e-5):** Lower the learning rate to **4e-5** (edit `lr` in the config file) and train for another **3–4 epochs**.

> **Important:** For this stage, do **not** use `--resume`. `--resume` restores the optimizer and lr scheduler saved in the checkpoint and would continue the old 2e-4 schedule. Instead, use `--init_from` to load only the model weights from the Stage-1 checkpoint, so the optimizer and scheduler start fresh with the new learning rate:

```bash
python train.py --config ./configs/robotwin_all.yaml \
    --init_from ./checkpoints_vla/<stage1_checkpoint>.pt
```

- `--resume`: for recovering from an interruption — restores model + optimizer + scheduler + epoch, continuing the original lr schedule.
- `--init_from`: for stage switching — loads **model weights only**; optimizer and lr scheduler are rebuilt from `--config`. The two options are mutually exclusive.


## Evaluation on RoboTwin 2.0

**Please refer to [README_EVAL.md](README_EVAL.md) for detailed evaluation instructions**.

## Checkpoints

Pre-trained weights on RoboTwin 2.0 (50 tasks) are available on ModelScope:
[ModelScope](https://www.modelscope.cn/models/yangfan97/LiLa-WAM_RoboTwin2_0)
[Google Drive](https://drive.google.com/drive/folders/15usxTjIyOTC4Fu2VNVoZ03efptM1mNFa?usp=drive_link)

## Citation

If you find this work useful, please consider citing:

```bibtex
@article{yang2026lila,
  title={LiLa-WAM: Lightweight Latent Reasoning World-Action Model for Robotic Manipulation},
  author={Yang, Fan and Su, Yuting and Wang, Xiaobo and You, Yuncheng and Fan, Fugui and Wu, Yuting and Wu, Minghui and Zhao, Chenxu and Ning, JiaHong and Jing, Peiguang},
  journal={arXiv preprint arXiv:2608.03701},
  year={2026}
}
```
