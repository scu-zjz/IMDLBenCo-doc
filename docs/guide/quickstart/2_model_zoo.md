# Reproducing SoTA Papers with Model Zoo
Most people initially use IMDL-Benco with the intention of reproducing SoTA (State-of-the-Art) papers. If you have a certain level of deep learning experience (such as working with the PyTorch framework, Linux Shell scripts, multi-GPU parallel parameters, etc.), this will be very simple.

## Initialization with `benco init`
After installing Benco, create a clean, empty folder as your working directory, and then run the following command:

```bash
benco init model_zoo
```

IMDL-BenCo will generate all the necessary Python scripts, shell scripts, default datasets, and necessary configuration files needed to reproduce the model zoo. The basic folder structure will look like this:

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

In the root directory, you will find the main logic scripts for training and testing: `train.py`, `test.py`, and `test_robust.py`. **Our design philosophy encourages you to modify these scripts as needed!**

The `./runs` folder contains all the shell scripts used to start the corresponding training processes. These shell scripts will call the Python scripts in the root directory, and the function of each script can be inferred from its name.

> For example, `demo_train_trufor.sh` is the script used to train TruFor, `demo_test_mvss.sh` is the script to test MVSS-Net, and `demo_test_robustness_cat_net.sh` is the script for robustness testing of CAT-Net.

The `./configs` folder contains configuration files for various models. You can adjust the corresponding hyperparameters by modifying these files, and the default paths will be automatically read by the shell scripts.

## Modifying Dataset Paths
Open the target shell script you want to use and make sure to modify the following fields to point to your dataset and checkpoint paths:

- Training Scripts:
  | Field Name | Function | Description |
  |-|-|-|
  |data_path|Training dataset path|Refer to [Dataset Preparation](./0_dataprepare.md)|
  |test_data_path|Test dataset path|Refer to [Dataset Preparation](./0_dataprepare.md)|

- Testing Scripts:
  | Field Name | Function | Description |
  |-|-|-|
  |test_data_json|Path to the test dataset JSON, which contains multiple datasets and their paths|Refer to the final section of [Dataset Preparation](./0_dataprepare.md)|
  |checkpoint_path|Folder path containing the checkpoint to be tested|This is a folder containing at least one checkpoint. The filename must include a number before the extension indicating the epoch, e.g., `checkpoint-68.pth`|

- Robustness Testing Scripts:
  | Field Name | Function | Description |
  |-|-|-|
  |test_data_path|Test dataset path|Refer to [Dataset Preparation](./0_dataprepare.md)|
  |checkpoint_path|Folder path containing the checkpoint to be tested|This is a folder containing at least one checkpoint. The filename must include a number before the extension indicating the epoch, e.g., `checkpoint-68.pth`|

For necessary PyTorch multi-GPU training parameter adjustments, please learn or consult ChatGPT. Common fields include:
- `CUDA_VISIBLE_DEVICES=0`: Specifies using only the GPU with this ID
- `--nproc_per_node=4`: Total number of GPUs to use

## Passing Hyperparameters to `nn.Module` via Shell Scripts (Syntactic Sugar)

In addition, each model may have its own special hyperparameters. In BenCo, extra command line parameters (those not required by `train.py`) can be directly passed into the `__init__` function of the `nn.Module`.
This functionality is implemented [here](https://github.com/scu-zjz/IMDLBenCo/blob/f4d158312b8f39df07aa41f468529c417bc9a765/IMDLBenCo/training_scripts/train.py#L133).

You can check the `__init__()` function of a model to understand its capabilities.

For example, in the training shell script `demo_train_trufor.sh`, the following fields:
```
    --np_pretrain_weights "/mnt/data0/dubo/workspace/IMDLBenCo/IMDLBenCo/model_zoo/trufor/noiseprint.pth" \
    --mit_b2_pretrain_weights "/mnt/data0/dubo/workspace/IMDLBenCo/IMDLBenCo/model_zoo/trufor/mit_b2.pth" \
    --config_path "./configs/trufor.yaml" \
    --phase 2 \
```

will be directly passed to the `__init__` function of the TruFor `nn.Module`, specifically at [this location](https://github.com/scu-zjz/IMDLBenCo/blob/f4d158312b8f39df07aa41f468529c417bc9a765/IMDLBenCo/model_zoo/trufor/trufor.py#L15-L18).

```python
@MODELS.register_module()
class Trufor(nn.Module):
    def __init__(self,
                 phase: int = 2,
                 np_pretrain_weights: str = None,
                 mit_b2_pretrain_weights: str = None,
                 config_path: str = None,
                 det_resume_ckpt: str = None):
        super(Trufor, self).__init__()
```

<p><span style="color: red; font-weight: bold;">Note!!! All hyperparameters in the shell scripts under Model_zoo are currently the optimal experimental settings according to the authors' team.</span></p>

## Pretrained Weights Download
Additionally, different models may have custom parameters or require pretrained weights. This part will be completed in future documentation. TODO

For now, you can directly refer to the README files in each model folder under [this path](https://github.com/scu-zjz/IMDLBenCo/tree/main/IMDLBenCo/model_zoo) to download the necessary pretrained weights.

## Running Shell Scripts
Switch to the root directory (where the `train.py`, `test.py`, and other scripts are located), and then run the following command directly:
```
sh ./runs/demo_XXXX_XXXX.sh
```
Pay attention to the path relationships and ensure that the configuration files and Python scripts are correctly referenced by the shell commands.

If you don't see any output, don't panic. To save logs, **all output and errors have been redirected to files.**

If everything runs correctly, a folder named `output_dir_xxx` or `eval_dir_xxx` will be generated in the current path. Inside this folder, you will find three logs: one for standard output (`logs.log`), one for warnings and errors (`error.log`), and one specifically for scalar statistics (`log.txt`).

If the model runs successfully, you should see the model iterating and outputting new logs at the end of `logs.log`:

```
......
[21:25:16.951899] Epoch: [0]  [ 0/80]  eta: 0:06:40  lr: 0.000000  predict_loss: 0.6421 (0.6421)  edge_loss: 0.9474 (0.9474)  label_loss: 0.3652 (0.3652)  combined_loss: 0.8752 (0.8752)  time: 5.0059  data: 1.5256  max mem: 18905
[21:25:52.536949] Epoch: [0]  [20/80]  eta: 0:01:55  lr: 0.000002  predict_loss: 0.6255 (0.6492)  edge_loss: 0.9415 (0.9405)  label_loss: 0.3607 (0.3609)  combined_loss: 0.8660 (0.8707)  time: 1.7791  data: 0.0004  max mem: 20519
[21:26:27.255074] Epoch: [0]  [40/80]  eta: 0:01:13  lr: 0.000005  predict_loss: 0.6497 (0.6615)  edge_loss: 0.9400 (0.9412)  label_loss: 0.3497 (0.3566)  combined_loss: 0.8729 (0.8730)  time: 1.7358  data: 0.0003  max mem: 20519
[21:27:02.311510] Epoch: [0]  [60/80]  eta: 0:00:36  lr: 0.000007  predict_loss: 0.6255 (0.6527)  edge_loss: 0.9404 (0.9404)  label_loss: 0.3400 (0.3519)  combined_loss: 0.8643 (0.8708)  time: 1.7527  data: 0.0003  max mem: 20519
......
```
If something goes wrong, please check the error information in `error.log` and resolve the issues.

All the `checkpoint-XX.pth` files will also be output to `output_dir_xxx` for future use.

**It is highly recommended to monitor the training process using TensorBoard with the following command. Benco provides numerous automated API interfaces to facilitate visualization, making it easier to confirm that the training is proceeding correctly.**
```
tensorboard --logdir ./
```

At this point, you have completed the process of reproducing the SoTA model.
