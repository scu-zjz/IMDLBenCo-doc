# 获得模型的参数量和FLOPs
我们认为学习最快的方式就是“Learn by Doing”（边做边学），所以通过几个案例来帮助使用者快速上手。

总的来说IMDL-BenCo通过类似`git`、`conda`这样的命令行调用方式帮助你快速完成图像篡改检测科研项目的开发。如果你学过vue等前端技术，那按照vue-cli来理解IMDLBenCo的设计范式会非常轻松。

无论如何，请先参考[安装](./install.md)完成IMDL-BenCo的安装。

:::tip 本章动机
科研和学术paper中，我们不仅要关注模型在任务上的指标是否强大，也需要关注模型涨点是否是以巨大的计算开销为代价的。本章可以方便的帮你观测你在BenCo中实现的模型复杂度指标：FLOPs和Parameter数量
:::

## 本功能技术来源
- 本功能实现自Facebook的`fvcore`中的`fvcore.nn.FlopCountAnalysis`类
- 详细信息请参考：[fvcore/docs/flop_count.md](https://github.com/facebookresearch/fvcore/blob/main/docs/flop_count.md)

## 更新版本
本功能在IMDLBenCo的[v0.1.37版本](https://github.com/scu-zjz/IMDLBenCo/releases/tag/v0.1.37)加入，请使用`benco -v`查看版本，必要时请更新到最新版本以使用复杂度统计功能。


## 准备
你只需要准备好待测的推理模型即可，或者直接通过注册机制调用`model_zoo`中已经实现好的现有模型。

因为只是计算复杂度，而无需关注推理准确率，你甚至不需要checkpoint。

## 使用
该功能可以在`benco init`和`benco init model_zoo`下使用，最新版本会在你的工作路径下生成[`test_complexity.py`](https://github.com/scu-zjz/IMDLBenCo/blob/main/IMDLBenCo/training_scripts/test_complexity.py)，该文件只需要显式指明模型名称，输入图像尺寸，padding还是resizing策略即可。

特别的，如果有的模型需要输入edge_mask等等额外信息才能推理的话，可以参考`MVSS-Net`的启动脚本：
```shell
python ./test_complexity.py \
    --model MVSSNet \
    --test_batch_size 1 \
    --edge_mask_width 7 \
    --image_size 512 \
    --if_resizing
```

因为只是测试复杂度，所以单卡单`batch_size`即可。否则FLOPs会随着batchsize的增大而随之成倍增长。