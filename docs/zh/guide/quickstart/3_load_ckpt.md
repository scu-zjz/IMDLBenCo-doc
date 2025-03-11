# 使用Model Zoo配合checkpoint快速测试
我们在百度网盘公开了一些我们论文中report的指标所使用的模型权重以供参考。根据模型名称以及训练时使用的数据集命名，在百度网盘的下载链接如下：


- 百度网盘：imdlbenco_ckpt
  - 链接: [https://pan.baidu.com/s/1DtkOwLCTunvI3d_GAAj2Dg?pwd=bchm](https://pan.baidu.com/s/1DtkOwLCTunvI3d_GAAj2Dg?pwd=bchm) 、
  - 提取码: bchm

## 需要注意的格式问题
因为`train.py`得到的checkpoint-xx.pth需要同时保存模型权重，优化器参数，以及scaler参数，所以当使用`torch.load("checkpoint-xx.pth")`后可以看到它按照字典形式组织，包括了model，optimizer等等很多参数，会导致整个checkpoint是三倍于单纯的模型权重的大小。其形式大致如下：

```
{
    "model": <state_dict of model>,
    "optimizer": <state_dict of model>,
    ......
}
```

为了节省网盘空间，所以本工作release的checkpoint只保留了“model"字段，丢弃了其余字段进行上传。但不影响正常复现和Evaluation。如下所示。
```python
ckpt_name = "iml_vit_casiav2.pth"
path_ckpt = "/mnt/data0/public_datasets/IML/IMDLBenCo_ckpt"

import torch
import os
full_path = os.path.join(path_ckpt, ckpt_name)
obj = torch.load(full_path)
print(obj.keys())

# 结果为：
# dict_keys(['model'])
```
## 如何使用下载的checkpoint完成推理
1. 我们建议首先通过`benco init model_zoo`在一个路径下，比如`/mnt/data0/xiaochen/workspace/test_benco/imlvit_inference`，生成model_zoo的所有代码文件。（如果不清楚请参考上一章）
2. 而后，下载对应的checkpoint后，模仿`train.py`输出checkpoint的格式将其拷贝并修改文件名到一个路径下。比如你可以将`iml_vit_casiav2.pth`修改为`checkpoint-0.pth`，然后将其放在一个空文件夹`/mnt/data0/xiaochen/workspace/test_benco/imlvit_inference/ckpts/`下。
3. 这样，你修改对应的`demo_test_iml_vit.sh`中的`--checkpoint_path`字段为这个文件夹路径即可。只要配置好相应的数据集路径，即可通过执行如下指令自动执行推理过程并观察结果。
```shell
sh ./runs/demo_test_iml_vit.sh
```