# # 篡改检测数据集索引

| 数据集名称        | 真实/篡改图像数量                            | 最小分辨率                                   | 最大分辨率                                    | 图像特点简介                                                 | 目前的下载链接                                               |
| ----------------- | -------------------------------------------- | -------------------------------------------- | --------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| CASIA v1.0        | 800/921                                      | 384x256                                      | 384x256                                       | 主要针对splice操作。                                         | [链接](https://github.com/namtpham/casia1groundtruth)        |
| CASIA v2.0        | 7491/5123                                    | 320x240                                      | 800x600                                       | 区分copy-move和splice两种篡改方式。                          | [链接](https://github.com/namtpham/casia2groundtruth)<br />[修正版链接](https://github.com/SunnyHaze/CASIA2.0-Corrected-Groundtruth) |
| Columbia          | 黑白数据集：933/921<br />彩色数据集：183/180 | 黑白数据集：128x128<br />彩色数据集：757x568 | 黑白数据集：128x128<br />彩色数据集：1152x768 | 基于未压缩图像的splice篡改，图像分辨率较高。                 | [黑白数据集]()<br />[彩色数据集](https://www.ee.columbia.edu/ln/dvmm/downloads/authsplcuncmp/) |
| Coverage          | 100/100                                      | 334x190                                      | 752x472                                       | 针对Copy-move篡改操作，通常将将一堆相似物品中其中一个进行copy-move操作。 | [链接](https://github.com/wenbihan/coverage)                 |
| NIST16            | 0/564                                        | 5616x3744                                    | 500x500                                       | 包括splice、remove、copy-move三种篡改，数据集内图像分辨率较高。 | [链接](https://mig.nist.gov/MFC/PubData/Resources.html)<br />[网盘链接](https://pan.baidu.com/share/init?surl=nZVUaqaBOD1u3xtxVDKmFw) |
| Defacto           |                                              |                                              |                                               | 数据量较大，篡改区域占全图面积的比例非常小。                 | [链接](https://www.kaggle.com/defactodataset/datasets)       |
| IMD2020           | 414/2020                                     | 260x193                                      | 2958x4437                                     | 篡改类型包括复杂现实编辑（如拼接、局部修饰），分辨率不固定。 | [链接](https://staff.utia.cas.cz/novozada/db/IMD2020.zip)    |
| FantasticReality  | 16592/19423                                  | 500x333                                      | 6000x4000                                     | 结合篡改定位与语义分割的多任务标注数据集，提供像素级篡改区域掩膜、实例分割和类别标签。 | [链接1](https://drive.google.com/drive/folders/1AIKseHoL0ebux7tyeBDWN5zkHzAgYBzJ?usp=sharing)<br />[链接2](https://github.com/mjkwon2021/CAT-Net/issues/51) |
| PhotoShop-battle  | 11142/91886                                  | 高度：136<br />宽度：68                      | 高度：20000<br />宽度：12024                  | 面向创意篡改检测的大规模真实场景数据集，图像分辨率跨度大。   | [链接1](https://github.com/dbisUnibas/PS-Battles)<br />[链接2](https://www.kaggle.com/datasets/timocasti/psbattles/data) |
| Carvalho（DSO-1） | 100/100                                      | 2048x1536                                    | 2048x1536                                     | 通过拼接添加人物的splice操作和其他辅助处理实现伪造。         | [链接](http://ic.unicamp.br/~rocha/pub/downloads/2014-tiago-carvalho-thesis/tifs-database.zip) |
| GRIP Dataset      | 80/80                                        | 1024x768                                     | 1024x768                                      | 专为评估复制-移动篡改检测算法在复杂后处理下的鲁棒性设计。数据集篡改区域较小。 | [链接](https://www.grip.unina.it/download/prog/CMFD/)        |
| CoMoFoD           | 260/260                                      | 512x512                                      | 3000x3000                                     | 专为copy-move检测算法评估设计，伪造图像和原始图像均应用了多种不同类型的后处理方法。 | [链接](https://www.vcl.fer.hr/comofod/download.html)         |
| CocoGlide         | 512/512                                      | 256x256                                      | 256x256                                       | 针对生成式篡改研究，结合 GLIDE 扩散模型与语义提示生成篡改内容，模拟语义级局部篡改。 | [链接](https://www.grip.unina.it/download/prog/TruFor/CocoGlide.zip) |
| tampCOCO          | 0/800000                                     | 72x51                                        | 640x640                                       | 基于COCO 2017数据集构建，包括copy-move和splice两部分，所有图像均经过JPEG压缩，保留清晰边界以支持模型学习低级篡改痕迹。。 | [链接](https://www.kaggle.com/datasets/qsii24/tampcoco)      |
| compRAISE         | 24462/0                                      | 2278x1515                                    | 6159x4079                                     | 基于高分辨率图像库和COCO实例标注构建的数据集。采取copy-move伪造，伪造区域为不规则形状，非矩形非对称。 | [链接](https://www.kaggle.com/datasets/qsii24/compraise)     |
| OpenForensics     | 0/2000                                       | 512x512                                      | 512x512                                       | 首个面向多张人脸伪造检测与分割的大规模数据集。内容较丰富，图像场景涵盖室内与室外；人脸情况多样，尺寸变化点。 | [链接](https://zenodo.org/records/5528418)                   |



## 1 CASIA v1.0和CASIA v2.0

### 1.1 基本信息

- 简介：CASIA两个数据集均是由**中科院自动化所**提供的篡改检测数据集，主要针对Splicing操作。特别的，官方<font color=red>没有提供0-1Mask作为groundtruth，只有篡改图像和原始图像！</font>
  - CASIA V1.0包含921张篡改后的图像以及对应的原始图像，分辨率固定为384x256，篡改方式只有Splicing
  - CASIA V2.0包含5123张篡改后的图像以及对应的原始图像，分辨率从320x240到 800x600不等，在Splicing之外还使用了blurring。
- APA格式引用：Dong, J., Wang, W., & Tan, T. (2013). CASIA Image Tampering Detection Evaluation Database. 2013 IEEE China Summit and International Conference on Signal and Information Processing, 422–426. https://doi.org/10.1109/ChinaSIP.2013.6625374
- 论文链接：[CASIA Image Tampering Detection Evaluation Database | IEEE Conference Publication | IEEE Xplore](https://ieeexplore.ieee.org/abstract/document/6625374)

### 1.2 下载使用

​	需要注意的是，原始的中科院自动化所网站目前仍处于维护中，无法使用。而数据集中有一些些命名相关的标注错误，且原始数据集没有提供0-1 mask作为GroundTruth。

​	综上，目前一般使用的是由“Pham”等人在2019年一篇论文“Hybrid Image-Retrieval Method for Image-Splicing Validation”中，开源的带有0-1mask的CASIA数据集，由该论文的作者自己通过算法对原始图像和篡改图像求差值来得到带有0-1mask和对应图片的数据集，并且修正了文件名相关的错误。该数据集也广泛流传在Kaggle等数据科学平台中，下方提供Pham等人提供的Github链接：

- CASIA v1.0：https://github.com/namtpham/casia1groundtruth
- CASIA v2.0：https://github.com/namtpham/casia2groundtruth

​	但是，Pham等人修正的CASIA v2.0仍然存在一些漏洞，基于该数据集，在DICALAB 的周吉喆老师指导下，马晓晨同学对于其中存在的一些谬误进行了修正后上传了一份新的修正后的数据集。除数十张图片的修改外，其余信息与Pham等人提供的数据集完全一致，可以通过如下Github链接下载：

- 修正后的CASIA v2.0：https://github.com/SunnyHaze/CASIA2.0-Corrected-Groundtruth

## 2 Columbia

### 2.1 基本信息

- 简介：Columbia数据集是哥伦比亚大学DVMM Lab制作的篡改检测数据集，主要针对的也是Splicing操作。需要注意的是，官网有两个不同版本的数据集，其中一个为黑白固定尺寸（128 x 128），另一个为高清彩色splicing数据集，**一般采用的是彩色的作为Benchmark。**
  - 黑白图片块数据集中有1800多个128x128的图片块
  - 彩色Splicing数据集中有180张分辨率在757x568 到1152x768之间的图片块。

- 论文链接：一般篡改图像相关论文只引用该数据集官网，但是原则上作者要求了需要引用他们的这篇论文：“Detecting Image Splicing Using Geometry Invariants And Camera Characteristics Consistency”（[hsu06ICMEcrf.pdf (columbia.edu)](https://www.ee.columbia.edu/ln/dvmm/publications/06/hsu06ICMEcrf.pdf)）。

- APA格式引用：Hsu, Y., & Chang, S. (2006). Detecting Image Splicing using Geometry Invariants and Camera Characteristics Consistency. 2006 IEEE International Conference on Multimedia and Expo, 549–552. https://doi.org/10.1109/ICME.2006.262447

### 2.2 下载使用

- Columbia Image Splicing Detection Evaluation Dataset（黑白图片块数据集）：https://www.ee.columbia.edu/ln/dvmm/downloads/AuthSplicedDataSet/AuthSplicedDataSet.htm

- Columbia Uncompressed Image Splicing Detection Evaluation Dataset（彩色Splicing数据集）：https://www.ee.columbia.edu/ln/dvmm/downloads/authsplcuncmp/

## 3 Coverage

### 3.1 基本信息

- 简介：包含100对篡改图像及对应真实图像，主要应用Copy-move篡改操作。其图像具有很高的欺骗性，往往是将一堆相似物品中其中一个进行copy-move操作。

- 论文：COVERAGE – A NOVEL DATABASE FOR COPY-MOVE FORGERY DETECTION

- APA格式引用：Wen, B., Zhu, Y., Subramanian, R., Ng, T.-T., Shen, X., & Winkler, S. (2016). COVERAGE — A novel database for copy-move forgery detection. 2016 IEEE International Conference on Image Processing (ICIP), 161–165. https://doi.org/10.1109/ICIP.2016.7532339

### 3.2 下载使用

- 官方的github链接：https://github.com/wenbihan/coverage

## 4 NIST16

### 4.1 基本信息

- 简介：有splice、remove、copy-move三种篡改，不到1k张样本。一般使用的是NIST16做为篡改检测的评估。数据集内图像分辨率较高，一般需要进行resize处理或以滑动窗口进行Evaluation。需要注意的是数据集划分存在一定的“泄露”，即测试集图片再训练集中也存在的问题，但是大家一般对这个问题“睁一只眼闭一只眼”。

- 论文链接：https://ieeexplore.ieee.org/abstract/document/8638296/

- APA格式引用：Guan, H., Kozak, M., Robertson, E., Lee, Y., Yates, A. N., Delgado, A., Zhou, D., Kheyrkhah, T., Smith, J., & Fiscus, J. (2019). MFC Datasets: Large-Scale Benchmark Datasets for Media Forensic Challenge Evaluation. 2019 IEEE Winter Applications of Computer Vision Workshops (WACVW), 63–72. https://doi.org/10.1109/WACVW.2019.00018

### 4.2 下载使用

这个数据集的官网使用比较麻烦，需要在NIST(National Institute of Standards and Technology，美国国家标准与技术研究所)官网注册后，获得许可才能进行下载。

- https://mfc.nist.gov/

- https://mfc.nist.gov/participant

 

理论上在一通操作后进入此网址https://mig.nist.gov/MFC/PubData/Resources.html下载上方的数据集，也就是一般常说的NIST16（更新的20的一般需要license）：

![image-20250401140137512](C:\Users\Sylence\AppData\Roaming\Typora\typora-user-images\image-20250401140137512.png)

**不这么麻烦的话，知乎有大哥传网盘了，非官方的获取方式如下（不保证长久使用）：**

百度网盘链接：https://pan.baidu.com/share/init?surl=nZVUaqaBOD1u3xtxVDKmFw

提取码：lik7

## 5 Defacto

### 5.1 基本信息

- 简介：Defacto数据集是基于MSCOCO生成的篡改图像数据集，内容非常的巨大，但是“总体篡改区域占全图面积的比例”非常小，即平均下来大部分篡改只是对于图片中非常小的物体进行操作，分布与其他的数据集还是区别较大的。

- 论文链接：https://ieeexplore.ieee.org/abstract/document/8903181/

- APA格式引用：Mahfoudi, G., Tajini, B., Retraint, F., Morain-Nicolier, F., Dugelay, J. L., & Pic, M. (2019). DEFACTO: Image and Face Manipulation Dataset. 2019 27th European Signal Processing Conference (EUSIPCO), 1–5. https://doi.org/10.23919/EUSIPCO.2019.8903181

 

此外，因为生成自COCO，作者还要求对于MS COCO数据集的进行引用：

- 论文链接：https://arxiv.org/abs/1405.0312

- APA格式引用：Lin, T.-Y., Maire, M., Belongie, S., Bourdev, L., Girshick, R., Hays, J., Perona, P., Ramanan, D., Zitnick, C. L., & Dollár, P. (2015). Microsoft COCO: Common Objects in Context (arXiv:1405.0312). arXiv. http://arxiv.org/abs/1405.0312 

### 5.2 下载使用

该数据集已经完全上传至Kaggle，所以还是比较好下载使用的。因为非常巨大，作者按照篡改类别分为了几个部分提供下载：

Inpainting：

https://www.kaggle.com/datasets/defactodataset/defactoinpainting

Copy-move：

https://www.kaggle.com/datasets/defactodataset/defactocopymove

Splicing：

https://www.kaggle.com/datasets/defactodataset/defactosplicing

## 6 IMD2020

### 6.1 基本信息

- 简介：IMD2020是由捷克科学院信息理论与自动化研究所团队构建的大规模篡改图像检测数据集，包含合成生成和真实篡改两部分，前者是基于35,000 张由 2,322 种不同相机型号拍摄的真实图像，通过使用大量图像操作方法（包括图像处理技术以及基于 GAN 或修复的方法）合成了一组被篡改的图像；后者为从互联网上收集的2000 张由随机人员创建的“真实生活”（不受控）的被篡改图像。提供精确的二进制掩码（0-1 Mask）标注。
  - 合成生成数据集：
    - 包含35,000张真实图像（来自2,322种相机模型）及对应35,000张篡改图像，总计70,000张。	
    - 篡改方法涵盖传统处理（JPEG压缩、模糊、噪声等）、GAN生成（如FaceApp）、图像修复（Inpainting）等，篡改区域占比5%-30%。
    - 所有篡改图像均提供二进制掩码标注篡改区域。
  - 真实生活数据集：
    - 包含2,010张从互联网收集的未控制篡改图像，每张均匹配原始图像，并手动标注二进制掩码。
    - 篡改类型包括复杂现实编辑（如拼接、局部修饰），分辨率不固定

- 论文链接：[https://ieeexplore.ieee.org/document/9096940/](https://ieeexplore.ieee.org/document/9096940)

- APA格式引用：Novozamsky, A., Mahdian, B., & Saic, S. (2020). IMD2020: A large-scale annotated dataset tailored for detecting manipulated images. In Proceedings of the IEEE/CVF winter conference on applications of computer vision workshops (pp. 71-80).[https://doi.org/10.1109/WACVW50321.2020.9096940](https://doi.org/10.1109/WACVW50321.2020.9096940)

### 6.2 下载使用

官方下载网址：https://staff.utia.cas.cz/novozada/db/

- 真实篡改数据集为：IMD2020 Real-Life Manipulated Images部分  

- 合成篡改数据集：IMD2020 Large-Scale Set of Inpainting Images部分

## 7 FantasticReality

### 7.1 基本信息

- 简介FantasticReality数据集是由俄罗斯国家航空系统研究院（GosNIIAS）、莫斯科物理技术学院（MIPT）和布鲁诺·凯斯勒基金会（FBK）联合构建的大规模篡改检测数据集，旨在解决现有数据集规模小、标注不全面的问题。同时提供像素级篡改区域掩膜（ground truth mask）、实例分割和类别标签，涵盖10个常见对象类别（如人、车、建筑等），是首个<font color=red>结合篡改定位与语义分割的多任务</font>标注数据集。包含16k真实图像和16k篡改图像，总计32k张图像，篡改方式主要为Splicing。
- 论文链接：[ The Point Where Reality Meets Fantasy: Mixed Adversarial Generators for Image Splice Detection](https://papers.nips.cc/paper_files/paper/2019/hash/98dce83da57b0395e163467c9dae521b-Abstract.html)
- APA格式引用：Kniaz, V. V., Knyaz, V., & Remondino, F. (2019). The point where reality meets fantasy: Mixed adversarial generators for image splice detection. *Advances in neural information processing systems*, *32*.

### 7.2 下载使用

- 原论文给出数据集下载链接：http://zefirus.org/MAG  （已失效）

- 作者在CAT-Net的github仓库中给出了新的下载链接:https://drive.google.com/drive/folders/1AIKseHoL0ebux7tyeBDWN5zkHzAgYBzJ?usp=sharing, 但是好像还是存在问题。

- 其他方式可以考虑联系作者获取下载权限，具体参见https://github.com/mjkwon2021/CAT-Net/issues/51， 作者给出了具体的联系方式以及相关要求。

## 8 PhotoShop-battle

###  8.1 基本信息

- 简介：PS-Battles数据集由瑞士巴塞尔大学基于Reddit社区**r/photoshopbattles**构建，是首个面向<font color=red>创意篡改检测</font>的大规模真实场景数据集。该数据集聚焦于社区用户生成的多样化、高语义性图像篡改内容，包含11,142组图像（总计103,028张），篡改类型包括：幽默合成、场景替换、角色融合（包括Splicing, copy-move, removal）。图像分辨率跨度大（宽度68\~12,024像素，高度136~20,000像素）。
- 论文链接：[The PS-Battles Dataset - an Image Collection for Image Manipulation Detection](https://arxiv.org/abs/1804.04866)
- APA格式引用：Heller, S., Rossetto, L., & Schuldt, H. (2018). The ps-battles dataset-an image collection for image manipulation detection. *arXiv preprint arXiv:1804.04866*.https://arxiv.org/abs/1804.04866、

### 8.2 下载使用

数据集官方github仓库：https://github.com/dbisUnibas/PS-Battles

Kaggle：https://www.kaggle.com/datasets/timocasti/psbattles/data

- ubuntu&MacOS下载：具体见github仓库，运行提供的`download.sh`脚本即可
- Windows下载：具体见Kaggle网址，将提供的download.py脚本和Originals.tsv＆photoshops.tsv放在同一目录中，然后运行`download.py`脚本即可。

## 9 Carvalho（DSO-1）

### 9.1 基本信息

- 简介：DSO-1数据集包含200张高分辨率图像（2048×1536像素），其中100张为原始未修改图像，100张为伪造图像。通过<font color=red>拼接添加人物</font>（在已经包含一个人或多个人的源图像中添加一个人或多个人，splicing操作）并辅以色彩、亮度调整实现伪造。
- 论文链接：[Exposing Digital Image Forgeries by Illumination Color Classification](https://ieeexplore.ieee.org/document/6522874)
- APA格式引用：De Carvalho, T. J., Riess, C., Angelopoulou, E., Pedrini, H., & de Rezende Rocha, A. (2013). Exposing digital image forgeries by illumination color classification. *IEEE Transactions on Information Forensics and Security*, *8*(7), 1182-1194.[https://doi.org/10.1109/TIFS.2013.2265677](https://doi.org/10.1109/TIFS.2013.2265677)

### 9.2 下载使用

下载地址：http://ic.unicamp.br/~rocha/pub/downloads/2014-tiago-carvalho-thesis/tifs-database.zip（貌似现在有问题，不会成功下载）

相关数据集合集：https://recodbr.wordpress.com/code-n-data/#porno

## 10 GRIP Dataset

### 10.1 基本信息

- 简介：数据集专为评估<font color =red>复制-移动篡改检测</font>算法在复杂后处理干扰下的鲁棒性设计。该数据集聚焦小规模篡改区域检测，包含80张图像，关注过往数据集篡改区域过大、后处理类型单一等问题。包含高分辨率图像（如768×1024像素），篡改区域面积覆盖4000像素（<1%）至50,000像素，分为平滑、混合、纹理三类背景复杂度。提供像素级真值掩膜（0-1 mask）
- 论文链接：[Efficient Dense-Field Copy–Move Forgery Detection](https://ieeexplore.ieee.org/document/7154457)
- APA格式引用：Cozzolino, D., Poggi, G., & Verdoliva, L. (2015). Efficient dense-field copy–move forgery detection. *IEEE Transactions on Information Forensics and Security*, *10*(11), 2284-2297.[https://doi.org/10.1109/TIFS.2015.2455334](https://doi.org/10.1109/TIFS.2015.2455334)

### 10.2 下载使用

论文作者 [University Federico II of Naples](http://www.unina.it/) 的 [GRIP组](https://www.grip.unina.it/)提供了该数据集的下载内容：https://www.grip.unina.it/download/prog/CMFD/

## 11 CoMoFoD

### 11.1 基本信息

- 简介：CoMoFoD（Copy-Move Forgery Detection）数据库由克罗地亚萨格勒布大学电气工程与计算学院开发，是专为<font color=red>复制-移动篡改</font>检测算法评估设计的综合性基准数据集。系统化整合多种几何变换与后处理操作，并提供像素级篡改掩膜标注，旨在解决现有数据集后处理类型单一、规模不足的问题。包含 260 组伪造图像，分为两个类别，其中小图像类别（512x512）有 200 组图像集，大图像类别（3000x2000）有 60 组图像图像集。所有伪造图像和原始图像都应用了不同类型的后处理方法，例如 JPEG 压缩、模糊、添加噪声、颜色减少等。

- 论文链接：[CoMoFoD — New database for copy-move forgery detection](https://ieeexplore.ieee.org/document/6658316)
- APA格式引用：Tralic, D., Zupancic, I., Grgic, S., & Grgic, M. (2013, September). CoMoFoD—New database for copy-move forgery detection. In *Proceedings ELMAR-2013* (pp. 49-54). IEEE.

### 11.2 下载使用

- 数据集官方网址:https://www.vcl.fer.hr/comofod/download.html ，其中
  - Small image category database（512x512）共200 组，下载连接：https://www.vcl.fer.hr/comofod/comofod_small.rar
  - Large image category database（3000x2000）共60 组，无下载链接，需联系作者申请。

## 12 CocoGlide

### 12.1 基本信息

- 简介：CocoGlide 是基于 COCO 2017 验证集构建的篡改检测数据集，专为评估现代生成模型（如扩散模型）的局部篡改检测能力设计。其通过结合 GLIDE 扩散模型与语义提示生成逼真篡改内容，替换原始图像中的对应区域（如动物、交通工具等），模拟真实场景下的语义级局部篡改。包含 **512 张篡改图像**，均通过 COCO 验证集的 256×256 像素裁剪块生成，填补了传统数据集中缺乏<font color=red>生成式篡改</font>样本的空白。
- 论文链接：[TruFor: Leveraging all-round clues for trustworthy image forgery detection and localization](https://arxiv.org/abs/2212.10957)
- APA格式引用：Guillaro, F., Cozzolino, D., Sud, A., Dufour, N., & Verdoliva, L. (2023). Trufor: Leveraging all-round clues for trustworthy image forgery detection and localization. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition* (pp. 20606-20615).

### 12.2 下载使用

论文作者 [University Federico II of Naples](http://www.unina.it/) 的 [GRIP组](https://www.grip.unina.it/)提供了该数据集的下载内容：https://www.grip.unina.it/download/prog/TruFor/CocoGlide.zip

## 13 tampCOCO

### 13.1 基本信息

- 简介：tampCOCO是由COCO 2017数据集构建的篡改检测数据集，包含SP COCO（跨图像拼接）和CM COCO（同图像复制-移动）两部分。提供像素级二值掩码（0-1 Mask）。
  - **SP COCO**：
    - 基于COCO图像，从一张图中随机选取物体（如人物、车辆等），经旋转/缩放后粘贴至另一张图的随机位置。
     - 总计20万张伪造图像，所有图像均经过JPEG压缩（质量因子60-100），未添加模糊等后处理。
  - **CM COCO**：
    - 在单张COCO图像内复制选定区域（如物体或背景）并粘贴至其他位置，生成复制-移动篡改样本。
     - 总计60万张图像，JPEG压缩参数与SP COCO一致，保留清晰边界以支持模型学习低级篡改痕迹。

- 论文链接：[Learning JPEG Compression Artifacts for Image Manipulation Detection and Localization](https://arxiv.org/abs/2108.12947)
- APA格式引用：Kwon, M. J., Nam, S. H., Yu, I. J., Lee, H. K., & Kim, C. (2022). Learning jpeg compression artifacts for image manipulation detection and localization. *International Journal of Computer Vision*, *130*(8), 1875-1895.https://arxiv.org/abs/2108.12947

### 13.2 下载使用

该数据集已经完全上传至Kaggle，https://www.kaggle.com/datasets/qsii24/tampcoco。

由于数据集较大，一共分为了13个部分供下载，以上链接导向全部下载的索引。

## 14 compRAISE

### 14.1 基本信息

- 简介：compRAISE（CM RAISE）是基于**RAISE高分辨率图像库**与**COCO实例标注**构建的复杂篡改检测数据集。数据集篡改方式为从RAISE数据集（含8,156张未压缩RAW图像）中选取高分辨率自然场景图像（分辨率范围：2,000×3,008 ~ 4,928×3,264）,之后借用COCO 2017的随机多边形实例掩码（约120万标注），从RAISE图像中提取不规则形状区域，确保篡改边界非矩形、非对称，并在单张单张RAISE图像内，执行**复制-移动策略**伪造。
- 论文链接：[Learning JPEG Compression Artifacts for Image Manipulation Detection and Localization](https://arxiv.org/abs/2108.12947)
- APA格式引用：Kwon, M. J., Nam, S. H., Yu, I. J., Lee, H. K., & Kim, C. (2022). Learning jpeg compression artifacts for image manipulation detection and localization. *International Journal of Computer Vision*, *130*(8), 1875-1895.https://arxiv.org/abs/2108.12947

### 14.2 下载使用

该数据集已经完全上传至Kaggle，https://www.kaggle.com/datasets/qsii24/compraise。

由于数据集较大，一共分为了15个部分供下载，以上链接导向全部下载的索引。

## 15 OpenForensics

### 15.1 基本信息

- 简介：OpenForensics是由日本国立信息学研究所、综合研究大学院大学和东京大学联合构建的首个面向<font color=red>多张人脸伪造检测与分割</font>的大规模数据集。该数据集专为复杂自然场景下的多任务研究设计，提供像素级精细标注，支持伪造检测、实例分割、伪造边界识别等多维度任务。
  - 包含115,325张图像，总计334,136张人脸（平均每图2.9张人脸），其中真实人脸160,670张，伪造人脸173,660张。
  - 划分为训练集（44K+图像）、验证集（7K+图像）、测试开发集（18K+图像）和测试挑战集（45K+图像）。
  - 图像场景涵盖室内（63.7%）与室外（36.3%）；人脸姿态、年龄、性别、遮挡情况高度多样，包含微小至大尺寸人脸。
  - 伪造人脸分辨率达512×512。
- 论文链接：[OpenForensics: Large-Scale Challenging Dataset For Multi-Face Forgery Detection And Segmentation In-The-Wild](https://arxiv.org/abs/2107.14480)
- APA格式引用：Le, T. N., Nguyen, H. H., Yamagishi, J., & Echizen, I. (2021). Openforensics: Large-scale challenging dataset for multi-face forgery detection and segmentation in-the-wild. In *Proceedings of the IEEE/CVF international conference on computer vision* (pp. 10117-10127).

### 15.2 下载使用

官方下载链接：https://zenodo.org/records/5528418

其中数据集划分为多个部分供下载，以上链接导向全部下载的索引。


<CommentService/>
