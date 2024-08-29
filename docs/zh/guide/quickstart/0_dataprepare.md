# 数据集准备

## 重要
数据集部分的功能和接口会在后续版本中由benco CLI统一管理。

目前暂时需要手动在每一个工作路径下管理相应的`json`或数据集路径完成部署。

## 数据集格式
- BenCo内部实现了3种不同的数据集格式。将数据集按照其中任何一种组织即可读取。

1. `JsonDataset`，传入一个json文件路径，使用如下JSON格式组织图片和对应mask：
```
[
    [
      "/Dataset/CASIAv2/Tp/Tp_D_NRN_S_N_arc00013_sec00045_11700.jpg",
      "/Dataset/CASIAv2/Gt/Tp_D_NRN_S_N_arc00013_sec00045_11700_gt.png"
    ],
    ......
    [
      "/Dataset/CASIAv2/Au/Au_nat_30198.jpg",
      "Negative"
    ],
    ......
]
```
其中“Negative”表示全黑的mask，即完全真实的图片，所以也不需要输入path。

2. `ManiDataset`，传入一个文件夹路径，该文件夹包含两个子文件夹`Tp`和`Gt`，benco自动从`Tp`中读取图片，从`Gt`中读取对应mask，并自动按照`os.listdir()`得到的文件名字典序进行配对。一般情况下，默认的CASIA数据集就是按照这个格式组织的。可以参考[IML-ViT中的样例文件夹](https://github.com/SunnyHaze/IML-ViT/tree/main/images/sample_iml_dataset)。
3. `BalancedDatast`, 传入一个json文件路径，专门用来组织[CAT-Net](https://openaccess.thecvf.com/content/WACV2021/html/Kwon_CAT-Net_Compression_Artifact_Tracing_Network_for_Detection_and_Localization_of_WACV_2021_paper.html)和[TruFor](https://openaccess.thecvf.com/content/CVPR2023/html/Guillaro_TruFor_Leveraging_All-Round_Clues_for_Trustworthy_Image_Forgery_Detection_and_CVPR_2023_paper.html)中使用的协议。
   1. 协议具体定义：Protocol-CAT使用到了9个大数据集进行训练，但是每一个Epoch只从每个数据集中随机采样1800张图组成一个16200张图的数据集完成训练。
   2. Json组织形式：
    ```JSON
    [
       [
           "ManiDataset",
           "/mnt/data0/public_datasets/IML/CASIA2.0"
       ],
       [
           "JsonDataset",
           "/mnt/data0/public_datasets/IML/FantasticReality_v1/FantasticReality.json"
       ],
       [
           "ManiDataset",
           "/mnt/data0/public_datasets/IML/IMD_20_1024"
       ],
       [
           "JsonDataset",
           "/mnt/data0/public_datasets/IML/tampCOCO/sp_COCO_list.json"
       ],
       [
           "JsonDataset",
           "/mnt/data0/public_datasets/IML/tampCOCO/cm_COCO_list.json"
       ],
       [
           "JsonDataset",
           "/mnt/data0/public_datasets/IML/tampCOCO/bcm_COCO_list.json"
       ],
       [
           "JsonDataset",
           "/mnt/data0/public_datasets/IML/tampCOCO/bcmc_COCO_list.json"
       ]
    ]
    ```
    二维数组，每一行代表一个数据集，第一列代表使用到的数据集Class类型的字符串，第二列是该类型需要读取数据集的路径。

将需要用的数据集按照需求，组织好后，即可开始考虑复现模型或实现自己的模型。