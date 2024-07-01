# 简介
IMDLBenCo 是一个模块化的，针对图像篡改检测和定位任务设计的深度学习工具包，基于的[PyTorch](https://pytorch.org/)框架设计。可以用于：
- 复现篡改检测领域SoTA模型
- 创建自己的篡改检测模型
- 批量比较多种视觉Backbone和篡改检测特征提取器的性能

## 设计理念与优势
IMDL-BenCo的设计理念为：尽最大可能平衡 ***科研代码对于自定义的需求*** 与 ***深度学习框架对于标准化的要求*** 二者的矛盾，并力求提高实验速度与代码开发效率。

在这个理念下，IMDL-BenCo代码框架上具有如下特点和优势：
- 易于上手
  - 相比于传统框架（比如OpenMMLab和Detectron2），IMDL-BenCo不依赖于**注册机制**
    - 方便配合IDE跳转查看类与函数的定义，而无需在文档的海洋中焦头烂额。
  - 代码风格与PyTorch原生框架高度相似，非常利于深度学习的初学者无缝上手使用。
- 速度很快：
  - 基于CLI（命令行界面，Command Line Interface）的代码生成机制。
    - 如果了解Vue等Web前端框架的CLI会非常熟悉这种模式，可以减少花费在框架代码上的时间，专注于模型设计与实验。
    - 同时满足灵活的自定义需求，直接在生成的代码修改即可，而不无需对框架源码hack。
  - GPU加速的评价指标计算，远超Sklearn等机器学习库原生方法的速度。
  - 熟练者仍可以使用注册机制实现大批量实验管理，高效完成消融实验。
- 功能全面：
  - 集成常见篡改检测数据集的下载和管理（TODO）
  - 集成丰富的预处理算法，包括MVSS-Net提出的多种“Naive Transform”，并支持根据[Albumentations](https://albumentations.ai/)库的接口自定义新的预处理接口。
  - 集成多种SoTA模型，可以直接用于实验和测试。
  - 集成了多种视觉任务优秀的Backbone，比如ResNet，Swin和SegFormer等等，可以用于基准进行实验。
  - 集成了多种篡改检测特征提取器，包含Sobel，BayarConv等等。
    - 可以配合Backbone进行测试。
    - 也可以在框架之外通过import直接使用于其他不依赖于IMDL-BenCo的模型代码构建。
  - 集成多种常见篡改检测领域常见评价指标，包含Image-level和Pixel-level的F1、AUC等等。
  - 集成Tensorboard等可视化工具，只需要向指定接口传入图像，标量即可。
  - 集成复杂度分析（参数量，FLOPs），Grad-CAM等分析工具，方便快捷完成论文图表。

## 框架设计
IMDL-BenCo代码框架的设计概览图如下所示：

![](/images/IMDLBenCo_overview.png)

主要的组件包含：
- 负责引入数据并进行预处理的`Dataloader`
- 管理全部模型，特征提取器的`Model Zoo`
- 基于GPU加速的`Evaluator`，用于计算评价指标
  
上述类是整个框架中最精心设计的部分，可以认为是IMDL-BenCo的主要贡献。

此外还有辅助的一些工具，包含：
- 数据集下载和管理工具`Data Library`和`Data Manager`（TODO）
- 全局的注册机制`Register`，可以实现从`str`到具体`class`或者`method`的映射，便于直接通过shell脚本调用相应的模型或方法，以便批量完成实验。
- 用于可视化分析结果的`visualize tools`，暂时只包含Tensorboard

以及一些零碎的接口和工具，包含：
- `PyTorch optimize tools`，主要是PyTorch训练相关的接口和工具。
- `Analysis tools`，主要是各种训练时或训练后，分析存档用的工具。

所有上述工具，各自独立地构成了类或者函数，并留有相应接口，最终通过多种`Training/Testing/Visualizing Scrips`调用并实现相应功能。

而整个IMDL-BenCo框架的CLI（命令行界面，Command Line Interface）则以类似Git中`git init`的行为，通过`benco init`自动地在合适的工作路径生成所有**默认**的`Training/Testing/Visualizing Scrips`脚本，供研究人员进行后续修改使用。

所以，我们尤其鼓励使用者按照需求修改`Training/Testing/Visualizing Scrips`的内容，完成对于框架功能的合理取用，满足自定义的需求。并根据图中的❄️、🔥标志建议、酌情对于标注为🔥的类按需创建新的类或修改、设计相应功能完成最相应的科研任务。

此外，数据集下载，模型checkpoint下载等等功能也是通过`benco data`等等CLI指令实现。


## 动机
篡改检测任务长久以来都面临着预处理不统一，训练数据集不统一，评价指标不统一，模型不开源，训练代码不开源等等问题，严重影响了模型之间的公平比较。

因此，我们希望通过一套规范而统一的代码框架，减轻开源工作所需的代码压力，鼓励更多的开源工作。并正确且准确地完成对于现有模型性能的评估与比较。推进整个图像篡改检测领域更加健康、公平的可持续发展。

