In Tensorboard, we can see many useful metrics that can monitor the training process and the convergence of the model. They all correspond to the information in the dictionary returned by the `forward()` function of the model in `my_model.py`.
```python
# Build the output dictionary
        output_dict = {
            "backward_loss": combined_loss,
            "pred_mask": torch.sigmoid(seg_pred),
            "pred_label": torch.sigmoid(cls_pred),

            "visual_loss": {
                "total_loss": combined_loss,
                "seg_loss": seg_loss,
                "cls_loss": cls_loss
            },

            "visual_image": {
                "pred_mask": seg_pred,
            }
        }
        return output_dict
```

The visualization of the loss function corresponds to the dictionary corresponding to the `visual_loss` field:

![](/images/training/training_loss.png)


The calculation of the evaluation metrics comes from the results of `pred_mask` and `pred_label` and is obtained by calculating with `mask`:

![alt text](/images/training/pixelF1.png)

The visualization of the predicted results, obtained from the `visual_image` field, and the default input images such as `image`, `mask` are also automatically visualized and output, which can be used to directly observe where the model's current predictions are not good, facilitating further model improvement.

![](/images/training/train_test_samples.png)

:::note Note
The tutorial should have image samples here. If you do not see the images, please check your network connection or enable VPN.
:::

All the weights obtained after training (Checkpoint) are saved in the path corresponding to `base_dir` at the beginning of the training `train_mymodel.sh` script. The log files of Tensorboard are also saved here for subsequent use and log review.

In addition to Tensorboard logs, we also provide a plain text log `log.txt` for archiving, which contains simple content, including all scalar information after each Epoch. An example is as follows:
```
......
{"train_lr": 5.068414063259753e-05, "train_total_loss": 0.040027402791482446, "train_seg_loss": 0.04001608065957309, "train_cls_loss": 1.1322131909352606e-05, "test_pixel-level F1": 0.6269838496863315, "epoch": 100}
{"train_lr": 4.9894792537480576e-05, "train_total_loss": 0.03938291078949974, "train_seg_loss": 0.039372251576574625, "train_cls_loss": 1.0659212925112626e-05, "epoch": 101}
{"train_lr": 4.910553386394297e-05, "train_total_loss": 0.039195733024078264, "train_seg_loss": 0.039184275720948555, "train_cls_loss": 1.1457303129702722e-05, "epoch": 102}
{"train_lr": 4.8316563303634596e-05, "train_total_loss": 0.0385435631179897, "train_seg_loss": 0.03853294577689024, "train_cls_loss": 1.061734109946144e-05, "epoch": 103}
{"train_lr": 4.752807947567499e-05, "train_total_loss": 0.035692626619510615, "train_seg_loss": 0.03568181328162471, "train_cls_loss": 1.0813337885906548e-05, "test_pixel-level F1": 0.6672104743469334, "epoch": 104}
```
The model is tested every 4 Epochs, so only the corresponding epoch will save `test_pixel-level F1`.

Thus, we have explained all the output content during the training process.

### Conducting Tests

We saw in the previous metric tests that the model still had an upward trend at the 104th Epoch, but to save time, we stopped training here and proceeded to subsequent tests.

The gap between tamper detection datasets is quite large, and generalization performance is the most important indicator to measure model performance, so we hope to measure all indicators at once. Unlike previous tutorials, we need to use the `test_datasets.json` file that can cover multiple test datasets at once to assist in testing. The format is as follows, and we also provide a sample of this file in the files after `benco init`.

```JSON
{
    "Columbia": "/mnt/data0/public_datasets/IML/Columbia.json",
    "NIST16_1024": "/mnt/data0/public_datasets/IML/NIST16_1024",
    "NIST16_cleaned": "/mnt/data0/public_datasets/IML/NIST16_1024_cleaning",
    "coverage": "/mnt/data0/public_datasets/IML/coverage.json",
    "CASIAv1": "/mnt/data0/public_datasets/IML/CASIA1.0",
    "IMD20_1024": "/mnt/data0/public_datasets/IML/IMD_20_1024"
}
```

You need to preprocess each dataset in advance, and the dataset index is in the [Tamper Detection Dataset Index](../../imdl_data_model_hub/data/IMDLdatasets.md) section, and the format requirements are in the [Dataset Preparation Section](./0_dataprepare.md). The above paths must be organized in the format of `ManiDataset` or `JsonDataset`.

Then we modify the `test_mymodel.sh` file to pass in the correct parameters, mainly including the following fields:
- Change `--mymodel` to `MyConvNeXt`
- Remove unnecessary `--MyModel_Customized_param` and `--pre_trained_weights` especially `pretrained_path` is usually initialized before model training. It is irrelevant during the testing phase.
- Set `--checkpoint_path` to the folder where all `checkpoint-xx.pth` are output during training. It will automatically read all files ending with `.pth` under this folder and determine which epoch the checkpoint comes from based on the number in the filename to correctly plot the test metrics line chart.
- Set `--test_data_json` to the path of the JSON containing multiple test set information mentioned earlier.
- Other parameters can be set according to the memory and other conditions as appropriate,

:::important Note
If you chose `--if_padding` during training, this means that the dataloader will organize all images in the 0-padding manner of [IML-ViT](https://github.com/SunnyHaze/IML-ViT), not the `--if_resizing` of most models. Then make sure that this parameter is consistent between training and testing, otherwise, there will definitely be performance loss due to the inconsistency between the training set and the test set.

You can double-check whether padding or resizing is correctly selected through the pictures visualized by Tensorboard!
:::

A modified `test_mymodel.sh` is as follows:
```shell
base_dir="./eval_dir"
mkdir -p ${base_dir}

CUDA_VISIBLE_DEVICES=0,1 \
torchrun  \
    --standalone    \
    --nnodes=1     \
    --nproc_per_node=2 \
./test.py \
    --model MyConvNeXt \
    --world_size 1 \
    --test_data_json "./test_datasets.json" \
    --checkpoint_path "./output_dir/" \
    --test_batch_size 32 \
    --image_size 512 \
    --if_resizing \
    --output_dir ${base_dir}/ \
    --log_dir ${base_dir}/ \
2> ${base_dir}/error.log 1>${base_dir}/logs.log
```

Then, remember to change `from mymodel import MyModel` at the beginning of `test.py` to `from mymodel import MyConvNeXt`.

At this point, running the following command can start batch testing metrics on various test sets:
```shell
sh test_mymodel.sh
```

At this time, you can also view the test progress and results through Tensorboard. You can filter `eval_dir` in the `filter` box on the left to only view the output results of this test.
```shell
tensorboard --logdir ./
```

The line charts of metrics obtained from multiple datasets after testing are as follows. Choose the Checkpoint with the best comprehensive performance and record the corresponding data in the paper.

![](/images/training/testing_results.png)

### Robustness Testing
Robustness testing introduces two dimensions of "attack type" and "attack strength" for grid search (`gird search`), so it is generally only the best-performing checkpoint during the test phase that undergoes robustness testing.

Accordingly, in the `test_robust_mymodel.sh` file, unlike `test_mymodel.sh`, the `--checkpoint_path` field here points to a specific checkpoint, not a folder.

The other fields are the same, remove unnecessary parameters, fill in the required parameters, and remember to change `from mymodel import MyModel` at the beginning of `test_robust.py` to `from mymodel import MyConvNeXt`.

The `test_robust_mymodel.sh` I finally used is as follows:
```shell
base_dir="./eval_robust_dir"
mkdir -p ${base_dir}

CUDA_VISIBLE_DEVICES=0,1 \
torchrun  \
    --standalone    \
    --nnodes=1     \
    --nproc_per_node=2 \
./test_robust.py \
    --model MyConvNeXt \
    --world_size 1 \
    --test_data_path "/mnt/data0/public_datasets/IML/CASIA1.0" \
    --checkpoint_path "/mnt/data0/xiaochen/workspace/IMDLBenCo_pure/guide/benco/output_dir/checkpoint-92.pth" \
    --test_batch_size 32 \
    --image_size 512 \
    --if_resizing \
    --output_dir ${base_dir}/ \
    --log_dir ${base_dir}/ \
2> ${base_dir}/error.log 1>${base_dir}/logs.log
```

The specific adjustment of attack strategies and intensities in robustness testing requires modification of `test_robust.py`. Please locate this modifiable code by searching for `TODO`:
```python
    """=================================================
    Modify here to Set the robustness test parameters TODO
    ==================================================="""
    robustness_list = [
            GaussianBlurWrapper([0, 3, 7, 11, 15, 19, 23]),
            GaussianNoiseWrapper([3, 7, 11, 15, 19, 23]), 
            JpegCompressionWrapper([50, 60, 70, 80, 90, 100])
    ]
```
The lists behind these `wrapper` represent the specific strengths of the attacks, and they internally encapsulate the Transform provided by [Albumentation](https://github.com/albumentations-team/albumentations) to implement the attack. The implementation of the `wrapper` itself, please refer to this [link](https://github.com/scu-zjz/IMDLBenCo/blob/main/IMDLBenCo/transforms/robustness_wrapper.py).

In particular, you can refer to the implementation of the `wrapper` in the current path to encapsulate a new custom `wrapper`, and then import your own wrapper here like `from mymodel import MyConvNeXt`. This way, you can achieve a custom flexible robustness test without modifying the source code.

*****

For test results, similarly, you can view them through Tensorboard:
```shell
tensorboard --logdir ./
```

At this time, there may be many different records. Please make good use of the filter function in the upper left corner of Tensorboard to filter the attack types and corresponding results you need to record.

![](/images/training/robustness_test_plot.png)

*****

### Conclusion
In this way, we have completed the process of designing a model from scratch, training the model, completing testing and robustness testing. If you have any questions or incomplete places, please feel free to raise issues in our repository or contact the author team by email. The suggestions from the first-hand users will be of great help to us and to future scholars!