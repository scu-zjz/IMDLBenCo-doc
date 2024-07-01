# Introduction

IMDLBenCo is a modular deep learning toolkit designed for image manipulation detection and localization (IMDL) tasks, based on the [PyTorch](https://pytorch.org/) framework. It can be used for:
- Reproducing state-of-the-art (SoTA) models in IMDL
- Creating your own IMDL models
- Comparing the performance of various visual backbones and IMDL feature extractors in bulk

## Design Philosophy and Advantages

The design philosophy of IMDL-BenCo is to balance the ***customization needs of research code*** with the ***standardization requirements of deep learning frameworks***, aiming to improve both experimental speed and code development efficiency.

Under this philosophy, IMDL-BenCo's framework has the following features and advantages:
- Easy to use:
  - Unlike traditional frameworks (e.g., OpenMMLab and Detectron2), IMDL-BenCo does not rely on a **registration mechanism**.
    - Facilitates IDE navigation to view class and function definitions without getting lost in extensive documentation.
  - Code style is highly similar to the native PyTorch framework, making it very easy for beginners in deep learning to start using it seamlessly.
- Fast:
  - Based on CLI (Command Line Interface) code generation mechanism.
    - Familiar to those who know CLI modes of web frontend frameworks like Vue, reducing time spent on framework code and focusing on model design and experimentation.
    - Meets flexible customization needs, allowing direct modifications to the generated code without hacking the framework source code.
  - GPU-accelerated evaluation metrics calculation, much faster than native methods of machine learning libraries like Sklearn.
  - Advanced users can still use the registration mechanism for batch experiment management and efficiently conduct ablation studies.
- Comprehensive:
  - Integrated common tampering detection dataset download and management (TODO)
  - Rich preprocessing algorithms, including various "Naive Transform" methods proposed by MVSS-Net, and support for custom preprocessing interfaces based on the [Albumentations](https://albumentations.ai/) library.
  - Multiple SoTA models integrated, ready for experimentation and testing.
  - Various excellent backbones for visual tasks, such as ResNet, Swin, and SegFormer, available for benchmarking experiments.
  - Multiple tampering detection feature extractors, including Sobel and BayarConv.
    - Can be tested with backbones.
    - Can also be used independently of IMDL-BenCo by importing directly into other model code constructions.
  - Multiple common tampering detection evaluation metrics integrated, including image-level and pixel-level F1, AUC, etc.
  - Integrated visualization tools like Tensorboard, requiring only the input of images and scalars into specified interfaces.
  - Integrated complexity analysis (parameters, FLOPs), Grad-CAM, and other analysis tools for quick and easy completion of paper charts.

## Framework Design

The overview of the IMDL-BenCo framework design is shown below:

![IMDL-BenCo Overview](/images/IMDLBenCo_overview.png)

The main components include:
- `Dataloader` for data loading and preprocessing
- `Model Zoo` for managing all models and feature extractors
- GPU-accelerated `Evaluator` for calculating evaluation metrics

These classes are the most carefully designed parts of the framework and can be considered the main contributions of IMDL-BenCo.

Additionally, there are auxiliary tools, including:
- `Data Library` and `Data Manager` for dataset download and management (TODO)
- Global registration mechanism `Register` for mapping `str` to specific `class` or `method`, making it easy to call corresponding models or methods via shell scripts for batch experiments.
- `Visualize tools` for visualization analysis, currently only including Tensorboard

And some miscellaneous interfaces and tools, including:
- `PyTorch optimize tools`, mainly for PyTorch training-related interfaces and tools.
- `Analysis tools`, mainly for various tools used for training or post-training analysis and archiving.

All the above tools form classes or functions independently, with corresponding interfaces, ultimately achieving their respective functions through various `Training/Testing/Visualizing Scripts`.

The CLI (Command Line Interface) of the entire IMDL-BenCo framework, similar to `git init` in Git, automatically generates all default `Training/Testing/Visualizing Scripts` scripts in an appropriate working path via `benco init`, for researchers to modify and use subsequently.

Therefore, users are especially encouraged to modify the content of `Training/Testing/Visualizing Scripts` as needed, making reasonable use of the framework's functions to meet customization needs. According to the ❄️ and 🔥 symbols in the diagram, users are advised to create new classes or modify and design corresponding functions as needed to accomplish specific research tasks.

Additionally, functions like dataset download and model checkpoint download are also achieved through CLI commands like `benco data`.

## Motivation

The IMDL task has long faced issues such as inconsistent preprocessing, inconsistent training datasets, inconsistent evaluation metrics, non-open source models, and non-open source training codes, seriously affecting the fair comparison between models. 

Therefore, we hope to alleviate the code burden of open-source work through a standardized and unified code framework, encouraging more open-source work. Furthermore, we aim to correctly and accurately evaluate and compare the performance of existing models, promoting a healthier, fairer, and more sustainable development of the IMDL field.