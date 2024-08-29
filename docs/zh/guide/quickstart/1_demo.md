# 上手案例
我们认为学习最快的方式就是“Learn by Doing”（边做边学），所以通过几个案例来帮助使用者快速上手。

总的来说IMDL-BenCo通过类似`git`、`conda`这样的命令行调用方式帮助你快速完成图像篡改检测科研项目的开发。如果你学过vue等前端技术，那按照vue-cli来理解IMDLBenCo的设计范式会非常轻松。

无论如何，请先参考[安装](./install.md)完成IMDL-BenCo的安装。
## 案例零：快速理解设计范式

### 生成默认脚本

在一个干净的工作路径下，执行如下命令行指令即可生成最简运行所需的全部脚本
```bash
benco init base
```
作为默认的指令，该命令也可以这样省略：

```bash
benco init
```

正常执行后会看到在当前路径下生成了如下文件，它们的用途如注释所示：
```bash
.
├── balanced_dataset.json       # 存放按照Protocol-CAT组织的数据集路径
├── mymodel.py                  # 核心的模型实现
├── README-IMDLBenCo.md         # 一个简单的readme
├── test_datasets.json          # 存放测试用的数据集路径
├── test_mymodel.sh             # 传参运行测试的shell脚本
├── test.py                     # 测试脚本的实际Python代码
├── test_robust_mymodel.sh      # 传参运行鲁棒性测试的shell脚本
├── test_robust.py              # 鲁棒性测试的实际Python代码
├── train_mymodel.sh            # 传参运行的训练shell脚本
└── train.py                    # 训练脚本的实际Python代码
```

**特别注意**：如果已经生成过脚本，并且做出一定修改后，请一定**小心二次调用**`benco init`，IMDLBenCo会在询问后逐一覆盖文件，如果误操作可能导致丢失你已有的修改，务必小心。推荐使用git版本管理来避免该操作导致丢失已有代码。


### 模型文件设计范式
IMDLBenCo需要按照一定格式组织模型文件，以保证入口可以和`DataLoader`、出口可以和后续的`Evaluator`和`Visualize tools`对齐。

执行`benco init`后默认以最简单的**单层卷积**生成一个模型在`mymodel.py`中，你可以先通过[Github中的mymodel.py链接](https://github.com/scu-zjz/IMDLBenCo/blob/main/IMDLBenCo/statics/base/mymodel.py)快速查看它的内容。


```python
from IMDLBenCo.registry import MODELS
import torch.nn as nn
import torch

@MODELS.register_module()
class MyModel(nn.Module):
    def __init__(self, MyModel_Customized_param:int, pre_trained_weights:str) -> None:
        """
        The parameters of the `__init__` function will be automatically converted into the parameters expected by the argparser in the training and testing scripts by the framework according to their annotated types and variable names. 
        
        In other words, you can directly pass in parameters with the same names and types from the `run.sh` script to initialize the model.
        """
        super().__init__()
        
        # Useless, just an example
        self.MyModel_Customized_param = MyModel_Customized_param
        self.pre_trained_weights = pre_trained_weights

        # A single layer conv2d 
        self.demo_layer = nn.Conv2d(
            in_channels=3,
            out_channels=1,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        # A simple loss
        self.loss_func_a = nn.BCEWithLogitsLoss()
        
    def forward(self, image, mask, label, *args, **kwargs):
        # simple forwarding
        pred_mask = self.demo_layer(image)
        
        # simple loss
        loss_a = self.loss_func_a(pred_mask, mask)
        loss_b = torch.abs(torch.mean(pred_mask - mask))
        combined_loss = loss_a + loss_b
        
        
        pred_label = torch.mean(pred_mask)
        
        inverse_mask = 1 - mask
        
        # ----------Output interface--------------------------------------
        output_dict = {
            # loss for backward
            "backward_loss": combined_loss,
            # predicted mask, will calculate for metrics automatically
            "pred_mask": pred_mask,
            # predicted binaray label, will calculate for metrics automatically
            "pred_label": pred_label,

            # ----values below is for visualization----
            # automatically visualize with the key-value pairs
            "visual_loss": {
                # keys here can be customized by yourself.
                "predict_loss": combined_loss,
                'loss_a' : loss_a,
                "I am loss_b :)": loss_b, 
            },

            "visual_image": {
                # keys here can be customized by yourself.
                # Various intermediate masks, heatmaps, and feature maps can be appropriately converted into RGB or single-channel images for visualization here.
                "pred_mask": pred_mask,
                "reverse_mask" : inverse_mask, 
            }
            # -------------------------------------------------------------
        }
        
        return output_dict
    
    
if __name__ == "__main__":
    print(MODELS)
```


**按照代码从前到后的顺序介绍，IMDLBenCo的model文件需要满足以下设计才能正常运行：**
- 第5行：`@MODELS.register_module()`
  - 基于注册机制注册该模型到IMDLBenCo的全局注册器中，便于其他脚本通过字符串快速调用该类。
  - 如果对注册机制不熟悉，一句话解释就是：**自动维护了一个从字符串到对应类的字典映射**，便于“自由地”传递参数。
- 第29行、第37行：**损失函数必须定义在`__init__()`或者`forward()`函数中**
- 第31行：定义forward函数时`def forward(self, image, mask, label, *args, **kwargs):`
  - 必须要带Python函数解包所需的`*args, **kwargs`，以接收未使用的参数。
    - 如果你不熟悉请参考[Python官方文档-4.8.2. Keyword Arguments](https://docs.python.org/3/tutorial/controlflow.html#keyword-arguments)，[Python官方文档中文版-4.8.2 关键字参数](https://docs.python.org/zh-cn/3/tutorial/controlflow.html#keyword-arguments)
  - 形参变量名必须与[`abstract_dataset.py`](https://github.com/scu-zjz/IMDLBenCo/blob/main/IMDLBenCo/datasets/abstract_dataset.py)中返回的字典`data_dict`包含的字段名完全一致。默认字段如下表所示：
    - |Key|含义|类型|
      |:-:|:-:|:-:|
      |image|输入的原始图片|Tensor(B,3,H,W)|
      |mask|预测目标的掩码|Tensor(B,1,H,W)|
      |lable|Image-level预测的零一标签|Tensor(B,1)|
      |shape|输入的图片的形状|Tensor(B,1,1), 分别代表H和W|
      |name|图片的路径和文件名|str|
      |shape_mask|在padding的情况下，仅计算该mask内为1的全部像素作为最终指标，1默认为和原图一样大的方形区域|Tensor(B,1,H,W)|
    - 对于不同的任务，可以按需取用这些字段输入到模型中使用。
    - 此外，对于CAT-Net需要的Jpeg相关的图片素材，我们设计了后处理函数`post_func`来根据已有的字段，生成更多需要的内容，此时也需要保证对应的forward函数的字段对齐。**有类似需求的自定义模型也可以使用这个范式来在dataloader引入其他模态的信息。** 以下是CAT-Net的案例：
      - [`cat_net_post_function`的Github链接](https://github.com/scu-zjz/IMDLBenCo/blob/c2d6dc03eab3f33461690d5026b43afdac22f70c/IMDLBenCo/model_zoo/cat_net/cat_net_post_function.py#L7-L10)，可以看到包含额外的`DCT_coef`和`q_tables`两个字段为模型输入额外的模态
      - [`cat_net_model`的Github链接](https://github.com/scu-zjz/IMDLBenCo/blob/c2d6dc03eab3f33461690d5026b43afdac22f70c/IMDLBenCo/model_zoo/cat_net/cat_net.py#L30)，`forward`函数的形参列表需要有相应字段接收上述额外输入的信息。
- 第36行到第38行：所有的损失函数必须在`forward`函数中完成计算
- 第45行到第70行：输出结果的字典。<span style="color: red;font-weight: bold;">非常重要！</span>，字典各字段的功能介绍如下：
  - |Key|含义|类型|
    |:-:|:-:|:-:|
    |backward_loss|直接用于反向传播的损失函数|Tensor(1)|
    |pred_mask|预测的mask，会直接用于后续指标计算|Tensor(B,1,H,W)|
    |pred_label|预测的零一标签，会直接用于后续指标计算|Tensor(B,1)|
    |visual_loss|传入需要可视化的标量。可以命名任意数量、任意名称的key并传入相应标量，后续会自动根据Key名进行可视化|Dict()|
    |visual_imagwe|传入需要可视化的图片，特征图，各种mask。可以命名任意数量，任意名称的key并传入相应Tensor，后续会自动根据Key名进行可视化|Dict()|
  - 务必按照此格式组织，才能正常融入后续指标运算、可视化等流程。 


至此，关于IMDLBenCo中模型代码实现的特点和注意事项已经介绍完了。

### 脚本运行设计范式



## 案例一：实现自己的新模型


## 案例二：复现已有模型

