# Quick Testing Using Model Zoo with Checkpoint

We have publicly shared some model weights used for the metrics reported in our paper on Baidu Netdisk for reference. The models are named according to the model name and the dataset used during training. The download link on Baidu Netdisk is as follows:

- Baidu Netdisk: imdlbenco_ckpt
  - Link: [https://pan.baidu.com/s/1DtkOwLCTunvI3d_GAAj2Dg?pwd=bchm](https://pan.baidu.com/s/1DtkOwLCTunvI3d_GAAj2Dg?pwd=bchm)
  - Extraction code: bchm

## Format Considerations
Since the `checkpoint-xx.pth` obtained from `train.py` needs to save the model weights, optimizer parameters, and scaler parameters simultaneously, when using `torch.load("checkpoint-xx.pth")`, you will see that it is organized in a dictionary format, including parameters such as model, optimizer, and many others. This results in the checkpoint being three times the size of the pure model weights. Its form is roughly as follows:

```
{
    "model": <state_dict of model>,
    "optimizer": <state_dict of model>,
    ......
}
```

To save space on the netdisk, the checkpoint released in this work only retains the "model" field, and the other fields have been discarded during upload. However, this does not affect normal reproduction and evaluation. As shown below.

```python
ckpt_name = "iml_vit_casiav2.pth"
path_ckpt = "/mnt/data0/public_datasets/IML/IMDLBenCo_ckpt"

import torch
import os
full_path = os.path.join(path_ckpt, ckpt_name)
obj = torch.load(full_path)
print(obj.keys())

# The result is:
# dict_keys(['model'])
```

## How to Use the Downloaded Checkpoint for Inference
1. We recommend first generating all the code files for the model_zoo in a path, such as `/mnt/data0/xiaochen/workspace/test_benco/imlvit_inference`, by using `benco init model_zoo`. (If unclear, please refer to the previous chapter.)
2. Then, after downloading the corresponding checkpoint, mimic the format of the checkpoint output by `train.py`, copy it, and modify the file name to a path. For example, you can rename `iml_vit_casiav2.pth` to `checkpoint-0.pth` and place it in an empty folder `/mnt/data0/xiaochen/workspace/test_benco/imlvit_inference/ckpts/`.
3. In this way, you can modify the `--checkpoint_path` field in the corresponding `demo_test_iml_vit.sh` to this folder path. As long as the corresponding dataset path is configured, you can automatically execute the inference process and observe the results by executing the following command.

```shell
sh ./runs/demo_test_iml_vit.sh
```