---
layout: page
title: Semantic Segmentation of Aerial Photographs
description: Per-pixel land use classification of sattelite imagery of Mumbai
img: assets/img/model_1_prediction_grid_cropped.png
importance: 1
category: work
related_publications: true
---

## Abstract

Semantic segmentation of aerial imagery is a critical tool for applications such as environmental monitoring, urban planning, and disaster assessment. In this project, I employed the U‑Net architecture with various enhancements—loss function modifications, encoder depth adjustments, dropout regularization, and attention mechanisms—to improve segmentation accuracy on satellite images of Mumbai. Seven models were trained and evaluated using IoU, Dice, Precision, and Recall across six classes (vegetation, built‑up areas, informal settlements, impervious surfaces, barren land, and water) plus a small “unclassified” class.

## Introduction

Aerial photography has a wide variety of uses, including geology, archaeology, disaster assessment, and environmental monitoring. Semantic segmentation, or pixel‑wise classification, enables the creation of masks that classify every region of an aerial image, supporting monitoring of environmental conditions, foreign objects, and temporal changes.

### Semantic Segmentation Performance Metrics

Performance metrics are defined in terms of true positive (TP), false positive (FP), false negative (FN), and true negative (TN) pixels. The most common metric is the Jaccard Score (IoU):

$$
\mathrm{IoU} = \frac{\mathrm{TP}}{\mathrm{TP} + \mathrm{FP} + \mathrm{FN}}
$$

Precision and Recall are:

$$
\mathrm{Precision} = \frac{\mathrm{TP}}{\mathrm{TP} + \mathrm{FP}},\quad
\mathrm{Recall} = \frac{\mathrm{TP}}{\mathrm{TP} + \mathrm{FN}}
$$

The F1 Score (Dice) is the harmonic mean:

$$
\mathrm{F1} = 2 \times \frac{\mathrm{Precision} \times \mathrm{Recall}}{\mathrm{Precision} + \mathrm{Recall}}
$$

### Transposed Convolution

Transposed convolution (upconvolution) upsamples by inserting zeros between pixels before convolving:

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="/assets/img/didl_transposed_convolution.png" title="An Example Upconvolution with a 2×2 Input, 2×2 Kernel, and Stride of 1" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    An Example Upconvolution with a 2×2 Input, 2×2 Kernel, and Stride of 1
</div>

## Dataset Description

The dataset is the Manually Annotated High Resolution Satellite Image Dataset of Mumbai [DOI](https://doi.org/10.17632/xj2v49zt26.1). It contains 110 images (600×600) split into overlapping 120×120 patches. Classes: vegetation, built‑up areas, informal settlements, impervious surfaces, barren land, water, plus unclassified.

| Class                 | Informal Settlements | Built‑Up    | Impervious Surfaces | Vegetation   | Barren       | Water        | Unclassified |
|-----------------------|---------------------:|------------:|--------------------:|-------------:|-------------:|-------------:|-------------:|
| **# pixels**          | 12,921,604           | 11,358,632  | 13,436,578          | 22,423,411   | 18,735,038   | 37,789,523   | 120,366      |

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="/assets/img/Dabra_class_balance.png" title="Distribution of Labels Across Patches" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Distribution of Labels Across Patches
</div>

## Methods

### U‑Net Architecture

The original U‑Net has a contracting encoder and expanding decoder with skip connections:

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="/assets/img/Ronneberger_architecture.png" title="The Original U‑Net Architecture" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    The Original U‑Net Architecture
</div>

### Loss Functions

#### Cross‑Entropy

$$
\mathcal{L}_{ce} = -\sum_{(x,y)} y(x,y)\,\log p(x,y)
$$

#### Weighted Cross‑Entropy

$$
\mathcal{L}_{wce} = -\sum_{(x,y)} w(x,y)\;y(x,y)\,\log p(x,y)
$$

#### Focal Loss

$$
\mathcal{L}_{wfl} = -\sum_{(x,y)}\sum_{c=1}^K w_c\,(1 - p_c(x,y))^\gamma\;y_c(x,y)\,\log p_c(x,y)
$$

### Data Augmentation

Patches are upsampled to 128×128 (a multiple of 32 for the five down‑sampling stages). Random 90° rotations per epoch add rotational invariance.

### Model Selection

Used the Segmentation Models PyTorch library with an EfficientNet‑B0 encoder (4M parameters).

### Optimizer & Scheduler

Adam optimizer with initial learning rate \(1\times10^{-4}\); LambdaLR scheduler:

$$
\lambda(\mathrm{epoch}) = 0.1^{\frac{\mathrm{epoch}}{40}}
$$

### Early Stopping

Training halts if validation loss doesn’t improve for 5 epochs.

## Results

### Model 1

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="/assets/img/model_1_training.png" title="Model 1 Training" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Model 1 Training
</div>

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="/assets/img/model_1_prediction_grid.png" title="Model 1 Predictions" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Model 1 Predictions
</div>

### Model 2

This model introduces Dice loss alongside cross-entropy loss to better address class imbalance by directly optimizing for overlap between predicted and ground truth masks.

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="/assets/img/model_2_prediction_grid.png" title="Model 2 Predictions" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Model 2 Predictions
</div>

### Model 3

To further handle class imbalance, class-weighted cross-entropy is combined with Dice loss. Weights were computed based on inverse class frequency and normalized.

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="/assets/img/model_3_prediction_grid.png" title="Model 3 Predictions" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Model 3 Predictions
</div>

### Model 4

This model uses a shallower U‑Net (encoder depth 4) with decoder channels [256,128,64,32] to reduce computational overhead while maintaining performance.

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="/assets/img/model_4_prediction_grid.png" title="Model 4 Predictions" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Model 4 Predictions
</div>

### Model 5

Dropout regularization (p=0.2) is added to the decoder to improve generalization and reduce overfitting on noisy data.

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="/assets/img/model_5_prediction_grid.png" title="Model 5 Predictions" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Model 5 Predictions
</div>

### Model 6

Spatial and Channel Squeeze-and-Excitation (SCSE) blocks are incorporated in the decoder to recalibrate feature channels and improve focus on important regions.

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="/assets/img/model_6_prediction_grid.png" title="Model 6 Predictions" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Model 6 Predictions
</div>

### Model 7

Combines SCSE attention, weighted focal loss, Dice loss, AdamW optimizer, and a cosine annealing scheduler with warm restarts for the most refined performance.

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="/assets/img/model_7_training.png" title="Model 7 Training" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Model 7 Training
</div>

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="/assets/img/model_7_prediction_grid.png" title="Model 7 Predictions" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Model 7 Predictions
</div>

## Conclusion

Model 1 achieved the best balance across metrics. Model 4 offered computational efficiency with minimal accuracy loss. Model 6’s attention blocks improved performance on challenging classes. Future work includes multi‑scale feature extraction, CRF post‑processing, advanced encoders (ResNet, ViT), and exploring newer architectures (DeepLabv3+, PSPNet, Mask R‑CNN).

## References

[1] National Air and Space Museum, “The Beginnings and Basics of Aerial Photography,” Available: https://airandspace.si.edu/stories/editorial/beginnings-and-basics-aerial-photography. Accessed: Dec. 8, 2024.

[2] P. Baumann, “HISTORY OF REMOTE SENSING, AERIAL PHOTOGRAPHY,” Available: http://employees.oneonta.edu/baumanpr/geosat2/rs%20history%20i/rs-history-part-1.htm. Accessed: Dec. 8, 2024.

[3] A. Dabra, “Manually Annotated High Resolution Satellite Image Dataset of Mumbai for Semantic Segmentation,” Mendeley, 2023. Available: https://data.mendeley.com/datasets/xj2v49zt26/1.

[4] G. Csurka, R. Volpi, B. Chidlovskii, “Semantic Image Segmentation: Two Decades of Research,” arXiv:2302.06378, 2023.

[5] O. Ronneberger, P. Fischer, T. Brox, “U-Net: Convolutional Networks for Biomedical Image Segmentation,” arXiv:1505.04597, 2015.

[6] P. Iakubovskii, “Segmentation Models Pytorch,” GitHub repository, 2019. Available: https://github.com/qubvel/segmentation_models.pytorch.

[7] A. Dabra, V. Kumar, “Evaluating Green Cover and Open Spaces in Informal Settlements of Mumbai Using Deep Learning,” Neural Computing and Applications, 2023. DOI: 10.1007/s00521-023-08320-7.

[8] A. Zhang, Z. C. Lipton, M. Li, A. J. Smola, “Dive into Deep Learning,” Cambridge University Press, 2023. Available: https://d2l.ai.

[9] J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li, F.-F. Li, “ImageNet: A Large-Scale Hierarchical Image Database,” in Proc. IEEE CVPR, 2009, pp. 248–255. DOI: 10.1109/CVPR.2009.5206848.

[10] M. Tan, Q. V. Le, “EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks,” arXiv:1905.11946, 2020.

[11] A. G. Roy, N. Navab, C. Wachinger, “Recalibrating Fully Convolutional Networks with Spatial and Channel ‘Squeeze & Excitation’ Blocks,” arXiv:1808.08127, 2018.

[12] D. Marmanis, J. D. Wegner, S. Galliani, K. Schindler, M. Datcu, U. Stilla, “Semantic Segmentation of Aerial Images with an Ensemble of CNNs,” ISPRS Annals of Photogrammetry, Remote Sensing and Spatial Information Sciences, vol. III-3, pp. 473–480, 2016. DOI: 10.5194/isprs-annals-III-3-473-2016.
