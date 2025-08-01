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

Semantic segmentation of aerial imagery is a critical tool for applications such as environmental monitoring, urban planning, and disaster assessment. In this project, I employed the U-Net architecture attempting a variety of enhancements to improve segmentation accuracy on a set of satellite images of Mumbai. Seven models were trained and evaluated, each integrating specific changes to the base model in loss function, encoder depth, dropout regularization, and addition of attention mechanisms. Metrics including IoU, Dice, Precision, and Recall were used to assess model performance across six classes: vegetation, built-up areas, informal settlements, impervious surfaces, barren land, and water. A small “unclassified” class is also considered.

## Introduction

Aerial photography has a wide variety of uses, including geology, archaeology, disaster assessment, and environmental monitoring {% cite NASM_AerialPhotography %}. The technology improved and became more widely utilized for military purposes starting in World War I, and has since gone on to be used for tasks such as identifying different vegetation types, detecting diseased and damaged vegetation, and counting how many missiles, planes, and other military hardware adversaries have and where it is located {% cite Baumann_RemoteSensingHistory %}. Semantic segmentation, or pixel-wise classification, is useful in this context. Creating a mask that classifies all regions of an aerial image allows for monitoring of environmental conditions, foreign objects, and changing conditions over time. While this project explores semantic segmentation of aerial images, the technology is useful for a broad number of tasks in biology, robotics, agriculture, sports analysis, and more.

### Semantic Segmentation Performance Metrics

Performance metrics for semantic segmentation can be thought of in terms of true positive (TP), false positive (FP), true negative (TN), and false negative (FN) classifications for each pixel.

The most common performance metric for semantic segmentation is the Jaccard Score, also known as *Intersection over Union (IoU)*.

$$
\mathrm{IoU} = \frac{\mathrm{TP}}{\mathrm{TP} + \mathrm{FP} + \mathrm{FN}}
$$

Another frequently used metric for semantic segmentation is the Dice Score, also known as the *F1 Score*. This metric is formulated from two related metrics, *Precision* and *Recall*.

$$
\mathrm{Precision} = \frac{\mathrm{TP}}{\mathrm{TP} + \mathrm{FP}},
\quad
\mathrm{Recall} = \frac{\mathrm{TP}}{\mathrm{TP} + \mathrm{FN}}
$$

$$
\mathrm{F1} = 2 \times \frac{\mathrm{Precision} \times \mathrm{Recall}}{\mathrm{Precision} + \mathrm{Recall}}
$$

### Transposed Convolution

Transposed convolution, also known as *upconvolution*, is a method of upsampling similar to downsampling with convolution. Between each input pixel, zeros are inserted to increase the size of the feature map before convolution with the kernel. An example of transposed convolution is given below:

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="/assets/img/didl_transposed_convolution.png" title="An Example Upconvolution with a 2×2 Input, 2×2 Kernel, and Stride of 1" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
An Example Upconvolution with a 2×2 Input, 2×2 Kernel, and Stride of 1 {% cite zhang2023dive %}
</div>

## Dataset Description

The dataset explored in this project is the Manually Annotated High Resolution Satellite Image Dataset of Mumbai for Semantic Segmentation {% cite xj2v49zt26.1 %}. The dataset was created from high-resolution, true-color satellite imagery of Pleiades-1A acquired on March 15, 2017 over Mumbai. There are six classifications: vegetation, built-up areas, informal settlements, impervious surfaces (roads, streets, parking lots, etc.), barren land, water, and a small number of unclassified pixels. The exact pixel distribution for the training set is given in Table 1.

| Class            | Informal Settlements | Built-Up    | Impervious Surfaces | Vegetation   | Barren       | Water        | Unclassified |
|------------------|---------------------:|------------:|--------------------:|-------------:|-------------:|-------------:|-------------:|
| **# pixels**     | 12,921,604           | 11,358,632  | 13,436,578          | 22,423,411   | 18,735,038   | 37,789,523   | 120,366      |

_Table 1: Training Patch Pixel Distribution_

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="/assets/img/Dabra_class_balance.png" title="Distribution of Labels Across Patches" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
Distribution of Labels Across Patches {% cite Dabra2023 %}
</div>

## Methods

### U-Net Architecture

Ronneberger, Fischer, and Brox proposed the first U-Net architecture to segment cells in microscopic images {% cite ronneberger2015unetconvolutionalnetworksbiomedical %}. The symmetric model consists of a contracting encoder and an expanding decoder with skip connections. In the encoder, each level applies a 3×3 convolution with ReLU activation followed by 2×2 max pooling to reduce spatial dimensions by half. The decoder performs a series of 2×2 upconvolutions to double the spatial dimensions and uses skip connections to preserve spatial information. The final 1×1 convolution reduces the number of channels to the number of classes.

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="/assets/img/Ronneberger_architecture.png" title="The Original U-Net Architecture" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
The Original U-Net Architecture {% cite ronneberger2015unetconvolutionalnetworksbiomedical %}
</div>

### Loss Functions

#### Cross-Entropy

$$
\mathcal{L}_{ce} = -\sum_{(x,y)} y(x,y)\,\log p(x,y)
$$

#### Weighted Cross-Entropy

$$
\mathcal{L}_{wce} = -\sum_{(x,y)} w(x,y)\;y(x,y)\,\log p(x,y)
$$

#### Focal Loss

$$
\mathcal{L}_{wfl} = -\sum_{(x,y)}\sum_{c=1}^K w_c\,(1 - p_c(x,y))^\gamma\,y_c(x,y)\,\log p_c(x,y)
$$

### Data Augmentation

The 120×120 patches are upsampled to 128×128 to be compatible with the five downsampling stages (128 is a multiple of 32). At the start of each epoch, patches and masks are randomly rotated by 0°, 90°, 180°, or 270° to add rotational invariance and prevent overfitting.

### Model Selection

Pre-trained models from the Segmentation Models PyTorch library {% cite Iakubovskii:2019 %} were used. EfficientNet-B0 {% cite tan2020efficientnetrethinkingmodelscaling %} pre-trained on ImageNet {% cite 5206848 %} was chosen as the encoder for its small size (4M parameters), reducing overfitting and accommodating limited compute resources.

### Optimizer & Scheduler

The Adam optimizer was used with an initial learning rate of 1×10⁻⁴ {% cite Dabra2023 %}. A LambdaLR scheduler decayed the learning rate by a factor of 0.1 every 40 epochs:

$$
\lambda(\text{epoch}) = 0.1^{\frac{\text{epoch}}{40}}
$$

### Early Stopping

Training was halted if the validation loss did not improve for five consecutive epochs.

## Results

### Model 1

The initial model follows the Methods exactly and serves as a baseline.

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="/assets/img/model_1_training.png" title="Model 1 Training" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
Model 1 Training
</div>

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="/assets/img/model_1_prediction_grid.png" title="Model 1 Predictions" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
Model 1 Predictions
</div>

### Model 2

This model introduces Dice loss alongside cross-entropy to better address class imbalance by directly optimizing overlap between prediction and ground truth.

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="/assets/img/model_2_prediction_grid.png" title="Model 2 Predictions" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
Model 2 Predictions
</div>

### Model 3

Class-weighted cross-entropy is combined with Dice loss. Weights were computed as the inverse class frequency and normalized; the unclassified class was scaled down by 0.001 before normalization.

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="/assets/img/model_3_prediction_grid.png" title="Model 3 Predictions" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
Model 3 Predictions
</div>

### Model 4

A shallower U-Net (encoder depth 4, decoder channels [256,128,64,32]) reduces compute overhead while maintaining performance.

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="/assets/img/model_4_prediction_grid.png" title="Model 4 Predictions" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
Model 4 Predictions
</div>

### Model 5

Dropout (p=0.2) was added in the decoder to improve generalization by reducing overfitting.

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="/assets/img/model_5_prediction_grid.png" title="Model 5 Predictions" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
Model 5 Predictions
</div>

### Model 6

SCSE attention blocks were incorporated in the decoder to recalibrate feature channels and focus on important regions {% cite roy2018recalibratingfullyconvolutionalnetworks %}.

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="/assets/img/model_6_prediction_grid.png" title="Model 6 Predictions" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
Model 6 Predictions
</div>

### Model 7

Combines SCSE attention, weighted focal loss, Dice loss, AdamW optimizer, and a cosine annealing scheduler with warm restarts for refined performance.

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="/assets/img/model_7_training.png" title="Model 7 Training" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
Model 7 Training
</div>

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="/assets/img/model_7_prediction_grid.png" title="Model 7 Predictions" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
Model 7 Predictions
</div>

## Model Comparison

Model 1 achieved the best balance across metrics, consistently leading in Dice, IoU, Precision, and Recall. Model 4 offered computational efficiency with minimal accuracy loss. Model 6’s attention blocks improved performance on challenging classes, while Models 2 and 7 underperformed.

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="/assets/img/model_comparison_1.png" title="Overall Comparison" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
Overall Comparison
</div>

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="/assets/img/model_comparison_2.png" title="Class-Wise IoU Comparison" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
Class-Wise IoU Comparison
</div>

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="/assets/img/model_comparision_3.png" title="Class-Wise Dice Comparison" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
Class-Wise Dice Comparison
</div>

## Conclusion

This work highlights the strengths and trade-offs of various modifications to the U-Net architecture for aerial image segmentation. Model 1, despite being the most basic, was the highest-performing model across all evaluation metrics. Model 4 demonstrates that computational efficiency can be achieved without substantial loss in accuracy. Model 6 showcases the value of attention mechanisms for complex regions. However, limitations remain in capturing fine boundaries and in class imbalance. A simple improvement may be to use an ensemble of U-Nets {% cite Marmanis2016 %}.

## Future Work

Future improvements include:
- Multi-scale feature extraction (e.g., feature pyramids)  
- Conditional Random Fields for post-processing  
- Advanced encoders (ResNet, Vision Transformers)  
- Exploring other architectures (DeepLabv3+, PSPNet, Mask R-CNN)  
