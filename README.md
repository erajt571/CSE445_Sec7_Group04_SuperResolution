# Machine Learning-Based Single-Image Super-Resolution with SRCNN

> **Course:** CSE445 - Machine Learning  
> **Section:** 07  
> **Group:** 04  
> **Institution:** North South University, Dhaka, Bangladesh

## Group members

- **Kazi Eraj Al Minahi Turjo** - 1831906642
- **Shamira Hossain Mila** - 2111089642
- **Salman Khan Fahim** - 2031558642

## Overview

This project implements a supervised **single-image super-resolution (SISR)** pipeline using the **Super-Resolution Convolutional Neural Network (SRCNN)**. A low-resolution image is enlarged by a factor of 4 using bicubic interpolation, then the trained SRCNN refines the enlarged image to recover sharper local structures and reduce reconstruction error.

The repository contains the training data, the trained checkpoint, preprocessing utilities, evaluation code, a new custom evaluation image set, and ready-to-use comparison figures for the report and presentation.

## Model architecture

The packaged model is a three-layer RGB SRCNN with **69,251 trainable parameters**:

```text
64x64 LR image
     |
     v
Bicubic x4 interpolation -> 256x256
     |
     v
Conv 9x9, 3 -> 64 + ReLU
     |
     v
Conv 5x5, 64 -> 32 + ReLU
     |
     v
Conv 5x5, 32 -> 3
     |
     v
SRCNN output, 256x256
```

The first layer extracts local image features, the second layer maps those features into a compact representation, and the final layer reconstructs the RGB image.

## Dataset and split

The standardized project dataset contains **110 paired images**:

| Split | Image IDs | Count |
|---|---|---:|
| Training | 0000-0089 | 90 |
| Validation | 0090-0099 | 10 |
| Test | 0100-0109 | 10 |

Each HR target is 256x256 pixels and each LR input is 64x64 pixels. Training uses randomly aligned 32x32 LR patches and their 128x128 HR targets. Random horizontal flips, vertical flips, and 90-degree rotations provide additional variation across epochs.

## Training configuration

| Parameter | Value |
|---|---|
| Epochs | 100 |
| Batch size | 16 |
| Loss | Mean Squared Error (MSE) |
| Optimizer | Adam |
| Initial learning rate | 1e-4 |
| Scheduler | StepLR, step=30, gamma=0.5 |
| Scale factor | x4 |
| Training patch | 32x32 LR -> 128x128 HR |
| Validation metric | PSNR |
| Mixed precision | Enabled when CUDA is available |

The recorded training run reached a best validation PSNR of approximately **22.36 dB**.

## Evaluation results

Using the packaged checkpoint on the ten project test images, the average result is approximately:

- **Bicubic:** 23.05 dB
- **SRCNN:** 24.07 dB
- **Average gain:** +1.02 dB

The strongest improvements in the packaged test set occur on images with repeated edges and local texture. The repository also contains a visually very different custom stress-test set (night car, tiger, pizza, glass building, macaw, mountain lake, watch, and astronaut). Those images are intentionally kept separate from training so they can demonstrate the model's generalization limits as well as its strengths.

## New image sets for the final presentation/report

For this Section 07 version, the main comparison figures are deliberately different from the earlier project visuals. Ready-to-use report/presentation comparisons are in:

```text
support/presentation_ready/
```

The folder includes project-test comparisons such as a scarecrow, rocky shoreline, monkey, street vendor, and parliament building, plus custom stress-test examples such as a night-time sports car and astronaut scene.

## Repository structure

```text
CSE445_Sec7_Group04_SuperResolution/
|-- main.py
|-- main.ipynb
|-- README.md
|-- requirements.txt
|-- data/
|   |-- HR_256/
|   |-- LR_x4/
|   |-- source_collection/
|   `-- evaluation_set/
|       |-- source_images/
|       |-- HR_256/
|       `-- LR_x4/
|-- support/
|   |-- model.py
|   |-- train_srcnn.py
|   |-- standardize_images.py
|   |-- evaluate_results.py
|   |-- best_srcnn_x4.pth
|   |-- new_results/
|   `-- presentation_ready/
`-- others/
    `-- final report / presentation / demo material
```

## How to run

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run the custom evaluation set:

```bash
python main.py
```

The generated outputs and PSNR table are saved under `support/new_results/custom_evaluation/`.

3. To retrain the model from the project dataset:

```bash
python support/train_srcnn.py
```

Training is optional for the final demonstration because the trained checkpoint is already included.

## Team contribution areas

The repository is designed so each member can make a genuine, identifiable contribution:

- **Eraj:** core model integration, main entry point, final assembly, report and repository documentation.
- **Shamira:** image preparation, custom evaluation dataset organization, HR/LR standardization pipeline.
- **Salman:** PSNR evaluation, Bicubic-vs-SRCNN comparison generation, result export and analysis utilities.

## Limitations

SRCNN improves average PSNR on the packaged project test set, but it does not outperform bicubic interpolation on every possible image. Images far outside the training distribution can produce lower PSNR because the model was trained on a small student-scale dataset. Text, highly regular geometry, and unfamiliar fine-detail patterns are especially difficult.

## References

1. C. Dong, C. C. Loy, K. He, and X. Tang, "Image Super-Resolution Using Deep Convolutional Networks," *IEEE TPAMI*, 2016.
2. D. P. Kingma and J. Ba, "Adam: A Method for Stochastic Optimization," ICLR, 2015.
3. A. Paszke et al., "PyTorch: An Imperative Style, High-Performance Deep Learning Library," NeurIPS, 2019.
4. OpenAI tools were used for writing support and to create the separate custom stress-test images used for evaluation-only examples.
