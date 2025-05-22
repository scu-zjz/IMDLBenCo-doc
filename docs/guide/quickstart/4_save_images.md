

# Case Four: Inference and Save a Dataset's Mask and Label
We believe that the fastest way to learn is through "Learn by Doing" (learning by doing), so we will help users quickly get started with a few cases.

Overall, IMDL-BenCo assists you in quickly completing the development of image tampering detection research projects through command line calls similar to `git` and `conda`. If you have learned front-end technologies like vue, understanding the design paradigm of IMDLBenCo in the same way as vue-cli will be very easy.

Regardless, please refer to [Installation](./install.md) to complete the installation of IMDL-BenCo first.

:::tip Motivation of This Chapter
In actual engineering development, in addition to metrics, the actual masks and labels are also very important. This chapter will tell you how to easily infer and save these contents for your subsequent use.
:::

## Update Version
This feature was added in the [v0.1.36 version](https://github.com/scu-zjz/IMDLBenCo/releases/tag/v0.1.36) of IMDLBenCo. Please use `benco -v` to check the version, and update to the latest version if necessary to use the inference feature.

## Preparation
First, you need a dataset to be inferred and a model and corresponding checkpoint for inference.
- For the dataset: you can refer to [Dataset Preparation](./0_dataprepare.md) to build it.
- For the inference model, you can obtain a checkpoint from your own training or download a checkpoint provided in our `model_zoo`.

## Usage
This feature can be used under `benco init` and `benco init model_zoo`. The latest version will generate [test_save_images.py](https://github.com/scu-zjz/IMDLBenCo/blob/main/IMDLBenCo/training_scripts/test_save_images.py) in your working path. The `--checkpoint_path` of this file reads **a specific checkpoint file** (different from test.py which reads a folder containing multiple ckpts) and a well-organized dataset path.

Inference supports multi-card acceleration. You can refer to the following MVSS-Net inference script to build a shell script to start inference and save:
```shell
base_dir="./save_img_dir_mvss"
mkdir -p ${base_dir}

CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun  \
    --standalone    \
    --nnodes=1     \
    --nproc_per_node=4 \
./save_images.py \
    --model MVSSNet \
    --edge_mask_width 7 \
    --world_size 1 \
    --test_data_path "/mnt/data0/public_datasets/IML/CASIA1.0" \
    --checkpoint_path "/mnt/data0/public_datasets/IML/IMDLBenCo_ckpt/checkpoint-mvss-casiav2.pth" \
    --test_batch_size 2 \
    --image_size 512 \
    --no_model_eval \
    --if_resizing \    --output_dir ${base_dir}/ \
    --log_dir ${base_dir}/ \
2> ${base_dir}/error.log 1>${base_dir}/logs.log
```

The framework will automatically resize or remove excess padding based on `--if_resizing` and `--if_padding` to ensure that the input images of the dataset and the output masks are the same size. All output images will be saved to the path corresponding to `--output_dir` according to the file name. Note that **if there are files with the same name in the dataset, they will be overwritten**, so be careful and rename the contents of the dataset if necessary.

In addition, if the model itself has an `image-level` output, the framework will first output multiple `pred_label_rank{rank}.json` based on the number of GPUs, and then merge all these files into a final `pred_label_combined.json`. This process will also **deduplicate based on file names**. The obtained json file is the model's predicted probability that each image has been tampered with, which is a floating-point number between 0 and 1, in the following format:
```json
{
    "Sp_D_CND_A_pla0005_pla0023_0281.jpg": 0.9999610185623169,
    "Sp_D_CNN_A_art0024_ani0032_0268.jpg": 1.0,
    "Sp_D_CNN_A_nat0085_ani0027_0271.jpg": 0.9998431205749512,
    "Sp_D_CNN_A_sec0012_ani0007_0275.jpg": 0.999996542930603,
    "Sp_D_CNN_R_art0017_art0090_0277.jpg": 1.0,
    "Sp_D_CRN_A_ani0036_ani0066_0372.jpg": 1.0,
    ...
}
```
If it is a model without `image-level` output and only performs segmentation, it will not output json by default, only images.

<CommentService/>