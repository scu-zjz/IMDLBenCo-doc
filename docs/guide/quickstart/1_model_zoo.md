# Case One: Reproduce SoTA Papers by Training with Model Zoo
We believe that the fastest way to learn is "Learn by Doing" (learning by doing), so we will help users get started quickly through several cases.

Overall, IMDL-BenCo helps you quickly complete the development of image tampering detection scientific research projects through command line calls similar to `git` and `conda`. If you have learned front-end technologies like vue, understanding the design pattern of IMDLBenCo according to vue-cli will be very easy.

Anyway, please refer to [Installation](./install.md) to complete the installation of IMDL-BenCo first.

:::tip Motivation of This Chapter
Most people who use IMDL-Benco for the first time should want to train and reproduce SoTA papers. If you have some deep learning experience (PyTorch framework, Linux shell scripts, multi-card parallelism parameters and other prerequisites), it will be very easy. This chapter will tell you all the processes needed for reproduction.
:::


## Initialize with benco init
After installing benco, create a clean empty folder as the working path, and then run the following command
```bash
benco init model_zoo
```

At this time, IMDL-BenCo will generate all the Python scripts, shell scripts, default datasets, and necessary configuration files required to reproduce model_zoo under this path. The rough folder structure is as follows:

```bash
.
├── balanced_dataset.json
├── configs
│   ├── CAT_full.yaml
│   └── trufor.yaml
├── runs
│   ├── demo_catnet_protocol_iml_vit.sh
│   ├── demo_catnet_protocol_mvss.sh
│   ├── demo_test_cat_net.sh
│   ├── demo_test_iml_vit.sh
│   ├── demo_test_mantra_net.sh
│   ├── demo_test_mvss.sh
│   ├── demo_test_object_former.sh
│   ├── demo_test_pscc.sh
│   ├── demo_test_robustness_cat_net.sh
│   ├── demo_test_robustness_iml_vit.sh
│   ├── demo_test_robustness_mantra_net.sh
│   ├── demo_test_robustness_mvss.sh
│   ├── demo_test_robustness_object_former.sh
│   ├── demo_test_robustness_pscc.sh
│   ├── demo_test_robustness_span.sh
│   ├── demo_test_robustness_trufor.sh
│   ├── demo_test_span.sh
│   ├── demo_test_trufor.sh
│   ├── demo_train_backbone_segformer.sh
│   ├── demo_train_backbone.sh
│   ├── demo_train_cat_net.sh
│   ├── demo_train_iml_vit.sh
│   ├── demo_train_mantra_net.sh
│   ├── demo_train_mvss.sh
│   ├── demo_train_object_former.sh
│   ├── demo_train_pscc.sh
│   ├── demo_train_span.sh
│   └── demo_train_trufor.sh
├── test_datasets.json
├── test.py
├── test_robust.py
└── train.py
```
Among them, the root directory contains the actual logic-bearing training and testing scripts `train.py`, `test.py`, `test_robust.py`.

::: tip Developer Tips, Must See!
Our design philosophy **encourages you to modify these scripts according to your needs**! For example, modify the `evaluator` to add more test metrics; modify the `transform` to change the training preprocessing, or rewrite the test logic, etc.

IMDL-BenCo is just a framework to improve development efficiency. In order to maximize flexibility in the face of scientific research work, we choose to generate code rather than couple various components within the source code, giving the greatest freedom back to the users. Please feel free to modify and show your wit!
:::

The `./runs` folder contains all the `shell` scripts used to start the corresponding training. These shells will call the Python scripts under the root directory, and their functions and model names can be confirmed based on the naming.

>For example: `demo_train_trufor.sh` is the script for training TruFor. `demo_test_mvss.sh` is the script for testing MVSS-Net, and `demo_test_robustness_cat_net.sh` is the script for robustness testing of CAT-Net.


The `./configs` path stores some model configuration files. You can adjust the corresponding hyperparameters by modifying them here, and the default path can be automatically read by the shell script.

## Modify the Input Dataset
Open the target Shell script you want to use, and be sure to modify the following field configurations to your dataset and checkpoint path:

- Training scripts:
  | Field Name | Field Function | Explanation |
  |-|-|-|
  |data_path|Path of the training dataset|Refer to [Dataset Preparation](./0_dataprepare.md)|
  |test_data_path|Path of the test dataset|Refer to [Dataset Preparation](./0_dataprepare.md)|
- Testing scripts:
  | Field Name | Field Function | Explanation |
  |-|-|-|
  |test_data_json|Path of the test dataset JSON, which is a JSON containing the names and paths of multiple datasets|Refer to the last section of [Dataset Preparation](./0_dataprepare.md)|
  |checkpoint_path|Path of the folder containing the checkpoint for testing|It is a folder with at least one checkpoint, and the name must have a number representing the epoch number before the extension name. For example, `checkpoint-68.pth`|
- Robustness testing scripts:
  | Field Name | Field Function | Explanation |
  |-|-|-|
  |test_data_path|Path of the training dataset|Refer to [Dataset Preparation](./0_dataprepare.md)|
  |checkpoint_path|Path of the folder containing the checkpoint for testing|It is a folder with at least one checkpoint, and the name must have a number representing the epoch number before the extension name. For example, `checkpoint-68.pth`|

Necessary PyTorch multi-card training parameter adjustments, please solve by learning or consulting ChatGPT, there are roughly the following fields:
- `CUDA_VISIBLE_DEVICES=0`, specify to use only the card with this number
- `--nproc_per_node=4`, the total number of running cards


## Pass nn.module Hyperparameters through Shell (Syntax Sugar)

In addition, each model will also have its own special hyperparameters. In BenCo, the "extra" (for example, command line arguments that are not needed inside train.py) command line arguments in the shell script can be directly passed to the `__init__` function of `nn.module`.
This feature is implemented [here](https://github.com/scu-zjz/IMDLBenCo/blob/f4d158312b8f39df07aa41f468529c417bc9a765/IMDLBenCo/training_scripts/train.py#L133)

So for now, you can understand the function by looking at the model's `__init__()` function.

Taking TruFor as an example, we can see that a large number of formal parameters need to be passed to the `__init__` function to correctly initialize the model in the specific implementation of the model's `nn.Module`, [code link here](https://github.com/scu-zjz/IMDLBenCo/blob/f4d158312b8f39df07aa41f468529c417bc9a765/IMDLBenCo/model_zoo/trufor/trufor.py#L15-L18).
```python
@MODELS.register_module()
class Trufor(nn.Module):
    def __init__(self,
                 phase: int = 2,
                 np_pretrain_weights: str = None,
                 mit_b2_pretrain_weights: str = None,
                 config_path: str = None,
                 det_resume_ckpt: str = None
                 ):
        super(Trufor, self).__init__()
```

In the BenCo framework, we can correctly initialize the corresponding model by passing the same field names and corresponding parameters into the training sh script `demo_train_trufor.sh`, [link here](https://github.com/scu-zjz/IMDLBenCo/blob/4c6a2937c3cae8d6ff26bf85e9bad0c5ec467468/IMDLBenCo/statics/model_zoo/runs/demo_train_trufor.sh#L14-L18):
```shell
    --np_pretrain_weights "/mnt/data0/dubo/workspace/IMDLBenCo/IMDLBenCo/model_zoo/trufor/noiseprint.pth" \
    --mit_b2_pretrain_weights "/mnt/data0/dubo/workspace/IMDLBenCo/IMDLBenCo/model_zoo/trufor/mit_b2.pth" \
    --config_path "./configs/trufor.yaml" \
    --phase 2 \
```


:::important Important Information
**Note!!! All hyperparameters in the shell scripts in Model_zoo are the current experimental optimal situations of the author team.**
:::



## Pre-trained Weights Download
In addition, different models will also have their own custom parameters or required pre-trained weights. This part will be supplemented in subsequent documents. TODO

Currently, you can directly refer to the README in each model folder under [this path](https://github.com/scu-zjz/IMDLBenCo/tree/main/IMDLBenCo/model_zoo) to download the required pre-trained weights.

## Run Shell Script
Switch to the root directory (the same level directory has train.py, test.py and other scripts), and then directly run the following command:
```
sh ./runs/demo_XXXX_XXXX.sh
```
Pay attention to the path relationship to ensure that the configuration files and Python scripts can be correctly indexed by the Shell command.

If there is no output, don't panic, in order to save logs, **all outputs and errors are redirected to files.**

If run correctly, a folder named `output_dir_xxx` or `eval_dir_xxx` will be generated in the current path, which outputs three logs, one is the normal standard output `logs.log`, one is warnings and errors `error.log`. There is also an independent log file specifically for statistical vectors `log.txt`

If the model runs normally, you should be able to see the model continuously iterating and outputting new logs at the end of `logs.log`:
```
......
[21:25:16.951899] Epoch: [0]  [ 0/80]  eta: 0:06:40  lr: 0.000000  predict_loss: 0.6421 (0.6421)  edge_loss: 0.9474 (0.9474)  label_loss: 0.3652 (0.3652)  combined_loss: 0.8752 (0.8752)  time: 5.0059  data: 1.5256  max mem: 18905
[21:25:52.536949] Epoch: [0]  [20/80]  eta: 0:01:55  lr: 0.000002  predict_loss: 0.6255 (0.6492)  edge_loss: 0.9415 (0.9405)  label_loss: 0.3607 (0.3609)  combined_loss: 0.8660 (0.8707)  time: 1.7791  data: 0.0004  max mem: 20519
[21:26:27.255074] Epoch: [0]  [40/80]  eta: 0:01:13  lr: 0.000005  predict_loss: 0.6497 (0.6615)  edge_loss: 0.9400 (0.9412)  label_loss: 0.3497 (0.3566)  combined_loss: 0.8729 (0.8730)  time: 1.7358  data: 0.0003  max mem: 20519
[21:27:02.311510] Epoch: [0]  [60/80]  eta: 0:00:36  lr: 0.000007  predict_loss: 0.6255 (0.6527)  edge_loss: 0.9404 (0.9404)  label_loss: 0.3400 (0.3519)  combined_loss: 0.8643 (0.8708)  time: 1.7527  data: 0.0003  max mem: 20519
......
```

If not normal, please look for error messages in `error.log` and solve them.

All `checkpoint-XX.pth` will also be output to `output_dir_xxx` for later use.

**Strongly recommend using the following command to use TensorBoard to monitor the training process. Benco provides a large number of automatic API interfaces to complete visualization, which is convenient to confirm whether the training is normal and to view some output mask results.**
```
tensorboard --logdir ./
```

So far, the reproduction process of SoTA Model has been completed.