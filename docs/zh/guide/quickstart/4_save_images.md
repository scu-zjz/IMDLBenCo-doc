# 案例四：推理并保存一个数据集的mask和label
我们认为学习最快的方式就是“Learn by Doing”（边做边学），所以通过几个案例来帮助使用者快速上手。

总的来说IMDL-BenCo通过类似`git`、`conda`这样的命令行调用方式帮助你快速完成图像篡改检测科研项目的开发。如果你学过vue等前端技术，那按照vue-cli来理解IMDLBenCo的设计范式会非常轻松。

无论如何，请先参考[安装](./install.md)完成IMDL-BenCo的安装。

:::tip 本章动机
在实际工程开发中，除了指标之外，实际的mask和label也是非常重要的，本章会告诉你如何轻松地推理并保存这些内容供您后续使用。
:::

## 更新版本
本功能在IMDLBenCo的[v0.1.36版本](https://github.com/scu-zjz/IMDLBenCo/releases/tag/v0.1.36)加入，请使用`benco -v`查看版本，必要时请更新到最新版本以使用推理功能。

## 准备
首先，你需要有一个待推理的数据集，和一个即将用于推理的模型和对应的checkpoint。
- 对于数据集：可以参考[数据集准备](./0_dataprepare.md)进行构建。
- 对于推理模型，你可以自己训练获得输出的checkpoint，也可以下载我们`model_zoo`中提供的checkpoint。

## 使用
该功能可以在`benco init`和`benco init model_zoo`下使用，最新版本会在你的工作路径下生成[`test_save_images.py`](https://github.com/scu-zjz/IMDLBenCo/blob/main/IMDLBenCo/training_scripts/test_save_images.py)，该文件的`--checkpoint_path`读入**一个具体的checkpoint文件**（区别于test.py读入的是一个包含多个ckpt的文件夹）和一个整理好的数据集路径。

推理支持多卡加速，可以参考如下MVSS-Net使用的推理脚本构建shell脚本来启动推理和保存：
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

框架会自动根据`--if_resizing`和`--if_padding`来resize或者去掉多余的padding，确保数据集输入的图片和输出的mask尺寸一致。所有输出的图片会根据文件名保存到`--output_dir`对应的路径。注意，**如果数据集中有同名的文件会覆盖**，请小心，必要时对数据集的内容进行重命名。

此外，如果模型本身带`image-level`的输出，则框架会先根据GPU数量输出多个`pred_label_rank{rank}.json`，然后将所有的这些文件合并为一个最后的`pred_label_combined.json`。这个过程也会**根据文件名去重**。得到的json文件是模型对于每每张图片是否经过篡改预测的概率值，为0到1的浮点数，格式如下：
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
如果是不带`image-level`输出，只做segmentation的模型，则默认不会输出json，只输出图片。