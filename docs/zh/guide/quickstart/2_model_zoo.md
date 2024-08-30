# 使用Model Zoo复现SoTA论文
大部分人初次使用IMDL-Benco应该都是想复现SoTA论文，如果你有一定深度学习经验（PyTorch框架，Linux的Shell脚本，多卡并行的参数等等前置知识），这会非常简单。

## 通过benco init初始化
安装好benco后，创建一个干净的空文件夹作为工作路径，然后运行如下指令
```bash
benco init model_zoo
```

这时IMDL-BenCo会在该路径下生成复现model_zoo所需的所有Python脚本、shell脚本、默认数据集和必要的配置文件。大致的文件夹结构如下：

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
其中，根目录下包含了实际承担逻辑的训练及测试脚本`train.py`,`test.py`, `test_robust.py`。**我们的设计理念鼓励按照您的需求对于这些脚本进行修改！**


其中`./runs`文件夹下包含了所有用于启动对应训练的`shell`脚本，这些shell会调用根目录下的Python脚本，根据命名可以确认该脚本的功能、模型名。
 
>比如: `demo_train_trufor.sh`是用于训练TruFor的脚本。`demo_test_mvss.sh`就是测试MVSS-Net的脚本，`demo_test_robustness_cat_net.sh`是对CAT-Net进行鲁棒性测试的脚本。


其中`./configs`路径下存放了一些模型的配置文件，需要调整对应超参数可以通过修改这里调整，默认路径即可由shell脚本自动读取。

## 修改传入数据集
打开你想要使用的目标Shell脚本，务必修改如下字段配置为你的数据集和checkpoint路径：

- 训练脚本：
  | 字段名 |字段功能|解释|
  |-|-|-|
  |data_path|训练数据集路径|参考[数据集准备](./0_dataprepare.md)|
  |test_data_path|测试数据集路径|参考[数据集准备](./0_dataprepare.md)|
- 测试脚本：
  | 字段名 |字段功能|解释|
  |-|-|-|
  |test_data_json|测试数据集JSON的路径，是一个包含了多个数据集名称和路径的JSON|参考[数据集准备](./0_dataprepare.md)的末尾章节|
  |checkpoint_path|存有准备测试的checkpoint的文件夹路径|是一个文件夹，里面至少有一个checkpoint，名称后，拓展名前必须有数字代表epoch数。比如`checkpoint-68.pth`|
- 鲁棒性测试脚本：
  | 字段名 |字段功能|解释|
  |-|-|-|
  |test_data_path|训练数据集路径|参考[数据集准备](./0_dataprepare.md)|
  |checkpoint_path|存有准备测试的checkpoint的文件夹路径|是一个文件夹，里面至少有一个checkpoint，名称后，拓展名前必须有数字代表epoch数。比如`checkpoint-68.pth`|

必要的PyTorch多卡训练参数调整，请通过学习或咨询ChatGPT解决，大致有如下字段：
- `CUDA_VISIBLE_DEVICES=0`，指定仅使用该编号显卡
- `--nproc_per_node=4`，总运行显卡数量


## 通过shell传入nn.module的超参数（语法糖）

此外，各个模型也会有自己的特殊的超参数，在BenCo中，shell脚本内部的“多余”（比如train.py内部不需要的命令行参数）命令行是可以直接传递到`nn.module`的`__init__`函数中的。
该功能实现于[这里](https://github.com/scu-zjz/IMDLBenCo/blob/f4d158312b8f39df07aa41f468529c417bc9a765/IMDLBenCo/training_scripts/train.py#L133)

所以暂时可以通过查看模型的`__init__()`函数来理解功能。

以TruFor为例，训练sh脚本`demo_train_trufor.sh`中的这几个字段：
```
    --np_pretrain_weights "/mnt/data0/dubo/workspace/IMDLBenCo/IMDLBenCo/model_zoo/trufor/noiseprint.pth" \
    --mit_b2_pretrain_weights "/mnt/data0/dubo/workspace/IMDLBenCo/IMDLBenCo/model_zoo/trufor/mit_b2.pth" \
    --config_path "./configs/trufor.yaml" \
    --phase 2 \
```

会被Benco直接传递到TruFor这个`nn.module`的`__init__`函数中，即[这个位置](https://github.com/scu-zjz/IMDLBenCo/blob/f4d158312b8f39df07aa41f468529c417bc9a765/IMDLBenCo/model_zoo/trufor/trufor.py#L15-L18)。

```
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


<p><span style="color: red; font-weight: bold;">注意！！！Model_zoo中各个shell脚本中所有的超参数均为作者团队官方目前的实验最优情况。</span></p>


## 预训练权重下载
此外，不同的模型还会有自己的自定义参数，或者需要的预训练权重，这部分会在后续文档中补齐。TODO

目前可以直接参考[此路径](https://github.com/scu-zjz/IMDLBenCo/tree/main/IMDLBenCo/model_zoo)下的各个模型的文件夹内的README，下载所需的预训练权重。

## 运行Shell脚本
切换到根目录（同级目录下有train.py，test.py等脚本），然后直接运行如下指令即可：
```
sh ./runs/demo_XXXX_XXXX.sh
```
注意路径关系，保证配置文件和Python脚本能正确被Shell的指令索引到。

发现没有输出，不要慌张，为了保存日志，**所有的输出和报错均被重定向到了文件。**

如果正确运行，则会在当前路径下生成一个名为`output_dir_xxx`或者`eval_dir_xxx`的文件夹，该文件夹内输出了三个日志，一个是正常的标准输出`logs.log`，一个是警告和报错`error.log`。还有一个独立的专门统计标量的`log.txt`

如果模型运行正常，则应该可以在`logs.log`末尾看到模型不断地迭代输出新的日志：
```
......
[21:25:16.951899] Epoch: [0]  [ 0/80]  eta: 0:06:40  lr: 0.000000  predict_loss: 0.6421 (0.6421)  edge_loss: 0.9474 (0.9474)  label_loss: 0.3652 (0.3652)  combined_loss: 0.8752 (0.8752)  time: 5.0059  data: 1.5256  max mem: 18905
[21:25:52.536949] Epoch: [0]  [20/80]  eta: 0:01:55  lr: 0.000002  predict_loss: 0.6255 (0.6492)  edge_loss: 0.9415 (0.9405)  label_loss: 0.3607 (0.3609)  combined_loss: 0.8660 (0.8707)  time: 1.7791  data: 0.0004  max mem: 20519
[21:26:27.255074] Epoch: [0]  [40/80]  eta: 0:01:13  lr: 0.000005  predict_loss: 0.6497 (0.6615)  edge_loss: 0.9400 (0.9412)  label_loss: 0.3497 (0.3566)  combined_loss: 0.8729 (0.8730)  time: 1.7358  data: 0.0003  max mem: 20519
[21:27:02.311510] Epoch: [0]  [60/80]  eta: 0:00:36  lr: 0.000007  predict_loss: 0.6255 (0.6527)  edge_loss: 0.9404 (0.9404)  label_loss: 0.3400 (0.3519)  combined_loss: 0.8643 (0.8708)  time: 1.7527  data: 0.0003  max mem: 20519
......
```

如果不正常，请在`error.log`中查找错误信息并解决。

所有的`checkpoint-XX.pth`也会输出到`output_dir_xxx`中，以供后续使用。

**强烈推荐通过如下指令使用TensorBoard监视训练过程，Benco提供了大量的自动API接口完成可视化，便于确认训练是否正常。**
```
tensorboard --logdir ./
```

至此，就完成了对于SoTA Model的复现过程。