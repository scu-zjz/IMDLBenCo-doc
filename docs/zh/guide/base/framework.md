# 框架设计
## 概览
IMDL-BenCo代码框架的设计概览图如下所示：

![](/images/IMDLBenCo_overview.png)

主要的组件包含：
- 负责引入数据并进行预处理的`Dataloader`
- 管理全部模型，特征提取器的`Model Zoo`
- 基于GPU加速的`Evaluator`，用于计算评价指标
  
上述类是整个框架中最精心设计的部分，可以认为是IMDL-BenCo的主要贡献。

此外还有辅助的一些组件，包含：
- 数据集下载和管理工具`Data Library`和`Data Manager`（TODO）
- 全局的注册机制`Register`，可以实现从`str`到具体`class`或者`method`的映射，便于直接通过shell脚本调用相应的模型或方法，以便批量完成实验。
- 用于可视化分析结果的`visualize tools`，暂时只包含Tensorboard。

以及一些零碎的工具，包含：
- `PyTorch optimize tools`，主要是PyTorch训练相关的接口和工具。
- `Analysis tools`，主要是各种训练时或训练后，分析存档用的工具。

所有上述工具，各自独立地构成了类或者函数，存在交互的组件间留有相应接口。最终，通过在多种`Training/Testing/Visualizing Scrips`中import调用并组合来实现相应的职责。

而整个IMDL-BenCo框架的CLI（命令行界面，Command Line Interface）则以类似Git中`git init`的行为，通过`benco init`自动地在合适的工作路径生成所有**默认**的`Training/Testing/Visualizing Scrips`脚本，供研究人员进行后续修改使用。

所以，我们尤其鼓励使用者按照需求修改`Training/Testing/Visualizing Scrips`的内容，完成对于框架功能的合理取用，满足自定义的需求。并根据图中的❄️、🔥标志建议、酌情对于标注为🔥的类按需创建新的类或修改、设计相应功能完成最相应的科研任务。

此外，数据集下载，模型checkpoint下载等等功能也是通过`benco data`等等CLI指令实现。

<CommentService/>