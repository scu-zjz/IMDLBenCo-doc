# Introduction
## Overview
IMDLBenCo is a modular deep learning toolkit designed for image manipulation detection and localization (IMDL) tasks based on the [PyTorch](https://pytorch.org/) framework. It can be used for:
- Reproducing state-of-the-art (SoTA) models in IMDL
- Creating your own IMDL models
- Comparing the performance of various visual backbones and IMDL feature extractors in bulk

## Design Philosophy and Advantages

The design philosophy of IMDL-BenCo is to balance the ***customization needs of research code*** with the ***standardization requirements of deep learning frameworks***, aiming to improve both experimental speed and code development efficiency.

Under this philosophy, IMDL-BenCo's framework has the following features and advantages:
- Easy to use:
  - Unlike traditional frameworks (e.g., OpenMMLab and Detectron2), IMDL-BenCo does not rely on a **registration mechanism**.
    - Facilitates IDE navigation to view class and function definitions without getting lost in the extensive documentation.
  - Code style is highly similar to the native PyTorch framework, making it very easy for beginners in deep learning to start using it seamlessly.
- Fast:
  - Based on CLI (Command Line Interface) code generation mechanism.
    - Familiar with those who know CLI modes of web frontend frameworks like Vue, reducing time spent on framework code and focusing on model design and experimentation.
    - Meets flexible customization needs, allowing direct modifications to the generated code without hacking the framework source code.
  - GPU-accelerated evaluation metrics calculation, much faster than native methods of machine learning libraries like Sklearn.
  - Advanced users can still use the registration mechanism for batch experiment management and efficiently conduct ablation studies.
- Comprehensive:
  - Integrated common tampering detection dataset download and management (TODO)
  - Rich preprocessing algorithms, including various "Naive Transform" methods proposed by MVSS-Net, and support for custom preprocessing interfaces based on the [Albumentations](https://albumentations.ai/) library.
  - Multiple SoTA models integrated, ready for experimentation and testing.
  - Various excellent backbones for visual tasks, such as ResNet, Swin, and SegFormer, are available for benchmarking experiments.
  - Multiple tampering detection feature extractors, including Sobel and BayarConv.
    - Can be tested with backbones.
    - Can also be used independently of IMDL-BenCo by importing directly into other model code constructions.
  - Multiple common tampering detection evaluation metrics integrated, including image-level and pixel-level F1, AUC, etc.
  - Integrated visualization tools like Tensorboard, only the input of images and scalars into specified interfaces are required.
  - Integrated complexity analysis (parameters, FLOPs), Grad-CAM, and other analysis tools for quick and easy completion of paper charts.


## Motivation

The IMDL task has long faced issues such as inconsistent preprocessing, inconsistent training datasets, inconsistent evaluation metrics, non-open source models, and non-open source training codes, seriously affecting the fair comparison between models. 

Therefore, we hope to alleviate the code burden of open-source work through a standardized and unified code framework, encouraging more open-source work. Furthermore, we aim to correctly and accurately evaluate and compare the performance of existing models, promoting a healthier, fairer, and more sustainable development of the IMDL field.

<CommentService/>