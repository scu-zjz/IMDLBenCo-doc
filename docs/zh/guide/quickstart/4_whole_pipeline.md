在Tensorboard中，我们可以看到很多有用的指标，可以监测训练过程和模型的收敛情况。它们全部和`my_model.py`中模型的`forward()`函数返回的字典中的信息对应。
```python
# 构建输出字典
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


损失函数的可视化对应`visual_loss`字段对应的字典:

![](/images/training/training_loss.png)


评价指标的计算来自于`pred_mask`和`pred_label`的结果和`mask`计算得到：

![alt text](/images/training/pixelF1.png)

可视化的预测结果，由`visual_image`字段得到，且默认输入的`image`，`mask`等图像也会自动被可视化输出，用于直接观察模型当前预测不好的地方在哪，方便进一步改进模型。

![](/images/training/train_test_samples.png)

:::note 注意
教程这里应该有图片样例，如果没看到图片，请检查网络连接或开启VPN。
:::

所有训练后得到的权重（Checkpoint）都保存在了训练`train_mymodel.sh`脚本开头的`base_dir`所对应的路径中。相应的`Tensorboard`的日志文件也保存在这里，以供后续取用和查阅日志。

除了Tensorboard日志，我们还提供了一个纯文本的日志`log.txt`留作档案，它保存的内容比较简单，只包含每一个Epoch结束后所有的标量信息。样例如下：
```
......
{"train_lr": 5.068414063259753e-05, "train_total_loss": 0.040027402791482446, "train_seg_loss": 0.04001608065957309, "train_cls_loss": 1.1322131909352606e-05, "test_pixel-level F1": 0.6269838496863315, "epoch": 100}
{"train_lr": 4.9894792537480576e-05, "train_total_loss": 0.03938291078949974, "train_seg_loss": 0.039372251576574625, "train_cls_loss": 1.0659212925112626e-05, "epoch": 101}
{"train_lr": 4.910553386394297e-05, "train_total_loss": 0.039195733024078264, "train_seg_loss": 0.039184275720948555, "train_cls_loss": 1.1457303129702722e-05, "epoch": 102}
{"train_lr": 4.8316563303634596e-05, "train_total_loss": 0.0385435631179897, "train_seg_loss": 0.03853294577689024, "train_cls_loss": 1.061734109946144e-05, "epoch": 103}
{"train_lr": 4.752807947567499e-05, "train_total_loss": 0.035692626619510615, "train_seg_loss": 0.03568181328162471, "train_cls_loss": 1.0813337885906548e-05, "test_pixel-level F1": 0.6672104743469334, "epoch": 104}
```
模型每4个Epoch测试一次，所以只有对应的epoch才会保存`test_pixel-level F1`。


至此，我们就解释完了所有训练过程中的输出内容。

### 开展测试

我们在前面的指标测试中看到在104Epoch时，模型还有上涨趋势，但为了节约时间，我们将训练停在这里，开展后续的测试。

篡改检测数据集之间的鸿沟较大，泛化性能是衡量模型性能最重要的指标，所以我们希望一次性测量所有指标。区别于前面的教程，我们要用到一次性可以涵盖多个测试集的`test_datasets.json`文件来帮助测试。其格式如下，我们也在`benco init`后的文件中提供了此文件的样例。

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

这里需要您预先处理好每一个数据集，数据集索引在[篡改检测数据集索引](../../imdl_data_model_hub/data/IMDLdatasets.md)章节，格式要求在[数据集准备章节](./0_dataprepare.md)。上述路径必须整理为`ManiDataset`或者`JsonDataset`的格式。，

然后我们修改`test_mymodel.sh`文件来传入正确的参数，主要包含如下字段：
- `--mymodel` 改为`MyConvNeXt`
- 去掉多余的`--MyModel_Customized_param` 和 `--pre_trained_weights`尤其是`pretrained_path`一般是在模型训练前初始化的。测试阶段无关。
- 将`--checkpoint_path`设置为训练时输出所有`checkpoint-xx.pth`的文件夹。它会自动读取这下面所有的以`.pth`结尾的文件，并根据文件名中的数字确认该checkpoint来自于第几个epoch以正确的绘制测试时的指标折线图。
- 将`--test_data_json`设置为前文含有多个测试集信息的JSON的路径。
- 其它参数按照显存等条件要求，酌情设置即可，

:::important 注意
如果你在训练时选择了`--if_padding`，这代表dataloader会将所有图像按照[IML-ViT](https://github.com/SunnyHaze/IML-ViT)的0-padding方式组织，而非大多数模型的`--if_resizing`。那一定要确认测试时该参数与训练时保持一致，否则训练集和测试集不一致，一定会有性能损失。

可以通过Tensorboard可视化的图片双重检查是否正确选择padding或者resizing！
:::

一个修改好的`test_mymodel.sh`如下：
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



然后，记得修改`test.py`开头的`from mymodel import MyModel`为`from mymodel import MyConvNeXt`。

此时，运行如下指令即可开始批量在各种测试集上测试指标：
```shell
sh test_mymodel.sh
```

此时也可以通过Tensorboard来查看测试进度和结果。可以在左侧的`filter`框过滤`eval_dir`来仅查看此次测试的输出结果。
```shell
tensorboard --logdir ./
```

测试后得到的多个数据集的指标折线图如下，选择综合性能最好的Checkpoint，并在论文中记录相应的数据即可。

![](/images/training/testing_results.png)

### 鲁棒性测试
鲁棒性测试因为对于“攻击类型”和“攻击强度”引入了两个维度进行网格搜索（`gird search`），所以一般只对测试阶段性能最好的那一个checkpoint进行鲁棒性测试。

所以相应的`test_robust_mymodel.sh`文件中，区别于`test_mymodel.sh`，这里的`--checkpoint_path`字段填入的路径指向一个具体的checkpoint，而非一个文件夹。

其他的字段同上，去掉无用的参数，填入需要的参数，并且记得修改`test_robust.py`开头的`from mymodel import MyModel`为`from mymodel import MyConvNeXt`。

我最终使用的`test_robust_mymodel.sh`如下
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

鲁棒性测试具体的攻击策略和强度的调整需要修改`test_robust.py`，请通过搜索`TODO`来定位到这段可以修改的代码：
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
这些`wrapper`后面的列表代表具体攻击的强度，他们内部封装了[Albumentation](https://github.com/albumentations-team/albumentations)提供的Transform来实现攻击。`wrapper`本身的实现请参考此[链接](https://github.com/scu-zjz/IMDLBenCo/blob/main/IMDLBenCo/transforms/robustness_wrapper.py)。

特别的，你可以在当前路径下参考源码中`wrapper`的实现封装新的自定义`wrapper`，然后像`from mymodel import MyConvNeXt`一样import你自己的wrapper到这里使用。这样无需修改源码，也能实现自定义灵活的鲁棒性测试。

*****

对于测试结果，同样的，你可以通过Tensorboard查看：
```shell
tensorboard --logdir ./
```

这时候可能会产生很多很多不同的记录，请活用Tensorboard左上角的filter功能，过滤你当前需要记录的攻击类型和响应的结果。

![](/images/training/robustness_test_plot.png)

*****

### 结语
这样我们就完成了一次从头到尾自己设计模型，训练模型，完成测试及鲁棒性测试的过程。有任何疑惑或者不完善的地方，欢迎向我们的仓库提issue或者给作者团队发邮件联系。第一手用户的建议对我们，以及对今后的学者帮助都会很大！