# Case Two: Using Model Zoo with Checkpoint for Quick Testing

We believe that the fastest way to learn is "Learn by Doing," so we provide several cases to help users get started quickly.

Overall, IMDL-BenCo helps you quickly complete the development of image tampering detection research projects through command-line calls similar to `git` and `conda`. If you have learned front-end technologies like vue, understanding the design pattern of IMDLBenCo according to vue-cli will be very easy.

Regardless, please refer to [Installation](./install.md) to complete the installation of IMDL-BenCo first.

:::tip Motivation of This Chapter
This chapter provides checkpoints used in most of the experiments in the [IMDL-BenCo paper](https://proceedings.neurips.cc/paper_files/paper/2024/hash/f280a398c243b5fdaa09f57ece880fc9-Abstract-Datasets_and_Benchmarks_Track.html), allowing you to complete reasoning and metric testing according to this case without training.
:::

## Checkpoint Links
We have made some model weights used in our reported metrics in the paper publicly available on Baidu Netdisk for reference. According to the model name and the dataset used during training, the download link on Baidu Netdisk is as follows:

- Baidu Netdisk: imdlbenco_ckpt
  - Link: [https://pan.baidu.com/s/1DtkOwLCTunvI3d_GAAj2Dg?pwd=bchm](https://pan.baidu.com/s/1DtkOwLCTunvI3d_GAAj2Dg?pwd=bchm) ,
  - Extraction Code: bchm

## Format Issues to Note
Since the checkpoint-xx.pth obtained from `train.py` needs to save model weights, optimizer parameters, and scaler parameters at the same time, when using `torch.load("checkpoint-xx.pth")`, you can see that it is organized in the form of a dictionary, including model, optimizer, and many other parameters, which makes the entire checkpoint three times the size of the pure model weights. The form is roughly as follows:

```
{
    "model": <state_dict of model>,
    "optimizer": <state_dict of optimizer>,
    ......
}
```

To save space on the netdisk, the checkpoint released in this work only retains the "model" field and discards the rest of the fields for upload. However, it does not affect normal reproduction and Evaluation. As shown below.
```python
ckpt_name = "iml_vit_casiav2.pth"
path_ckpt = "/mnt/data0/public_datasets/IML/IMDLBenCo_ckpt"

import torch
import os
full_path = os.path.join(path_ckpt, ckpt_name)
obj = torch.load(full_path)
print(obj.keys())

# Result is:
# dict_keys(['model'])
```
## How to Use the Downloaded Checkpoint for Inference
1. We recommend first using `benco init model_zoo` to generate all the code files of model_zoo in a path, for example, `/mnt/data0/xiaochen/workspace/test_benco/imlvit_inference`. (If you are not clear, please refer to the previous chapter)
2. After downloading the corresponding checkpoint, copy and modify the file name to a path in the format output by `train.py`. For example, you can change `iml_vit_casiav2.pth` to `checkpoint-0.pth`, and then place it in an empty folder `/mnt/data0/xiaochen/workspace/test_benco/imlvit_inference/ckpts/`.
3. In this way, you can modify the `--checkpoint_path` field in the corresponding `demo_test_iml_vit.sh` to this folder path. As long as the corresponding dataset path is configured, you can automatically execute the inference process and observe the results by executing the following command.
```shell
sh ./runs/demo_test_iml_vit.sh
```

<CommentService/>