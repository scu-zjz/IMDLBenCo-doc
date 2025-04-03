# Framework Design

## Overview

The overview of the IMDL-BenCo framework design is shown below:

![IMDL-BenCo Overview](/images/IMDLBenCo_overview.png)

The main components include:
- `Dataloader` for data loading and preprocessing
- `Model Zoo` for managing all models and feature extractors
- GPU-accelerated `Evaluator` for calculating evaluation metrics

These classes are the most carefully designed parts of the framework and can be considered the main contributions of IMDL-BenCo.

Additionally, there are auxiliary components, including:
- `Data Library` and `Data Manager` for dataset download and management (TODO)
- Global registration mechanism `Register` for mapping `str` to specific `class` or `method`, making it easy to call corresponding models or methods via shell scripts for batch experiments.
- `Visualize tools` for visualization analysis, currently only including Tensorboard.

And some miscellaneous tools, including:
- `PyTorch optimize tools`, mainly for PyTorch training-related interfaces and tools.
- `Analysis tools`, mainly for various tools used for training or post-training analysis and archiving.

Each of the aforementioned components is independently designed as a class or function, with appropriate interfaces for interaction between components. Ultimately, they fulfill their respective roles by being imported, called, and combined in various ·、`Training/Testing/Visualizing Scripts`.

The CLI (Command Line Interface) of the entire IMDL-BenCo framework, similar to `git init` in Git, automatically generates all default `Training/Testing/Visualizing Scripts` scripts in an appropriate working path via `benco init`, for researchers to modify and use subsequently.

Therefore, users are especially encouraged to modify the content of `Training/Testing/Visualizing Scripts` as needed, making reasonable use of the framework's functions to meet customization needs. According to the ❄️ and 🔥 symbols in the diagram, users are advised to create new classes or modify and design corresponding functions as needed to accomplish specific research tasks.

Additionally, functions like dataset download and model checkpoint download are also achieved through CLI commands like `benco data`.

<CommentService/>