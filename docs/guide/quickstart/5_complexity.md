# Obtaining Model Parameters and FLOPs
We believe that the fastest way to learn is by "Learn by Doing" (learning by doing), so we will use a few examples to help users get started quickly.

In general, IMDL-BenCo helps you quickly complete the development of image tampering detection scientific research projects through command line calls similar to `git` and `conda`. If you have learned front-end technologies such as vue, understanding the design paradigm of IMDLBenCo according to vue-cli will be very easy.

Regardless, please refer to [Installation](./install.md) to complete the installation of IMDL-BenCo first.

:::tip Motivation for This Chapter
In scientific research and academic papers, we not only need to focus on whether the model's performance on the task is strong, but also need to pay attention to whether the model's performance improvement comes at the cost of huge computational overhead. This chapter can conveniently help you observe the complexity indicators of the models you implement in BenCo: FLOPs and Parameter count.
:::

## Technical Sources of This Feature
- This feature is implemented from the `fvcore.nn.FlopCountAnalysis` class in Facebook's `fvcore`.
- For more information, please refer to: [fvcore/docs/flop_count.md](https://github.com/facebookresearch/fvcore/blob/main/docs/flop_count.md)

## Update Version
This feature was added in the [v0.1.37 version](https://github.com/scu-zjz/IMDLBenCo/releases/tag/v0.1.37) of IMDLBenCo. Please use `benco -v` to check the version, and update to the latest version if necessary to use the complexity statistics feature.

## Preparation
You only need to prepare the inference model to be tested, or directly call the existing models implemented in `model_zoo` through the registration mechanism.

Since it's just calculating complexity and not focusing on inference accuracy, you don't even need a checkpoint.

## Usage
This feature can be used under `benco init` and `benco init model_zoo`. The latest version will generate [`test_complexity.py`](https://github.com/scu-zjz/IMDLBenCo/blob/main/IMDLBenCo/training_scripts/test_complexity.py) in your working path. This file only needs to explicitly specify the model name, input image size, padding or resizing strategy.

In particular, if some models require additional information such as input edge_mask to perform inference, you can refer to the startup script of `MVSS-Net`:
```shell
python ./test_complexity.py \
    --model MVSSNet \
    --test_batch_size 1 \
    --edge_mask_width 7 \
    --image_size 512 \
    --if_resizing
```

Since it's just testing complexity, a single card with a single `batch_size` is sufficient. Otherwise, FLOPs will increase proportionally with the increase in batch size.