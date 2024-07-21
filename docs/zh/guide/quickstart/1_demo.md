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

按照代码从前到后的顺序介绍，IMDLBenCo的model文件需要满足以下设计才能正常运行：
- 在模型类名前的`@MODELS.register_module()`
  - 基于注册机制注册该模型到IMDLBenCo的全局注册器中，便于其他脚本通过字符串快速调用该类。
  - 如果对注册机制不熟悉，一句话解释就是：**自动维护了一个从字符串到对应类的字典映射**，便于“自由地”传递参数。
- **损失函数必须定义在`__init__()`或者`forward()`函数中**
- 定义forward函数时`def forward(self, image, mask, label, *args, **kwargs):`
  - 必须要带Python函数解包所需的`*args, **kwargs`，以接收未使用的参数。如果你不熟悉请参考[Python官方文档-4.8.2. Keyword Arguments](https://docs.python.org/3/tutorial/controlflow.html#keyword-arguments)，[Python官方文档中文版-4.8.2 关键字参数](https://docs.python.org/zh-cn/3/tutorial/controlflow.html#keyword-arguments)
  - 形参变量名必须与[`abstract_dataset.py`](https://github.com/scu-zjz/IMDLBenCo/blob/main/IMDLBenCo/datasets/abstract_dataset.py)中返回的字典`data_dict`包含的字段名完全一致。
    - 


## 案例一：实现自己的新模型


## 案例二：复现已有模型

