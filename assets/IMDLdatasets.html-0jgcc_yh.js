import{_ as d,c as p,a as r,b as t,e as l,d as a,w as i,r as o,o as h}from"./app-DCerHgAi.js";const g={};function u(c,e){const n=o("font"),s=o("CommentService");return h(),p("div",null,[e[47]||(e[47]=r('<h1 id="篡改检测数据集索引" tabindex="-1"><a class="header-anchor" href="#篡改检测数据集索引"><span># 篡改检测数据集索引</span></a></h1><table><thead><tr><th>数据集名称</th><th>真实/篡改图像数量</th><th>最小分辨率</th><th>最大分辨率</th><th>图像特点简介</th><th>目前的下载链接</th></tr></thead><tbody><tr><td>CASIA v1.0</td><td>800/921</td><td>384x256</td><td>384x256</td><td>主要针对splice操作。</td><td><a href="https://github.com/namtpham/casia1groundtruth" target="_blank" rel="noopener noreferrer">链接</a></td></tr><tr><td>CASIA v2.0</td><td>7491/5123</td><td>320x240</td><td>800x600</td><td>区分copy-move和splice两种篡改方式。</td><td><a href="https://github.com/namtpham/casia2groundtruth" target="_blank" rel="noopener noreferrer">链接</a><br><a href="https://github.com/SunnyHaze/CASIA2.0-Corrected-Groundtruth" target="_blank" rel="noopener noreferrer">修正版链接</a></td></tr><tr><td>Columbia</td><td>黑白数据集：933/921<br>彩色数据集：183/180</td><td>黑白数据集：128x128<br>彩色数据集：757x568</td><td>黑白数据集：128x128<br>彩色数据集：1152x768</td><td>基于未压缩图像的splice篡改，图像分辨率较高。</td><td><a href="">黑白数据集</a><br><a href="https://www.ee.columbia.edu/ln/dvmm/downloads/authsplcuncmp/" target="_blank" rel="noopener noreferrer">彩色数据集</a></td></tr><tr><td>Coverage</td><td>100/100</td><td>334x190</td><td>752x472</td><td>针对Copy-move篡改操作，通常将将一堆相似物品中其中一个进行copy-move操作。</td><td><a href="https://github.com/wenbihan/coverage" target="_blank" rel="noopener noreferrer">链接</a></td></tr><tr><td>NIST16</td><td>0/564</td><td>5616x3744</td><td>500x500</td><td>包括splice、remove、copy-move三种篡改，数据集内图像分辨率较高。</td><td><a href="https://mig.nist.gov/MFC/PubData/Resources.html" target="_blank" rel="noopener noreferrer">链接</a><br><a href="https://pan.baidu.com/share/init?surl=nZVUaqaBOD1u3xtxVDKmFw" target="_blank" rel="noopener noreferrer">网盘链接</a></td></tr><tr><td>Defacto</td><td></td><td></td><td></td><td>数据量较大，篡改区域占全图面积的比例非常小。</td><td><a href="https://www.kaggle.com/defactodataset/datasets" target="_blank" rel="noopener noreferrer">链接</a></td></tr><tr><td>IMD2020</td><td>414/2020</td><td>260x193</td><td>2958x4437</td><td>篡改类型包括复杂现实编辑（如拼接、局部修饰），分辨率不固定。</td><td><a href="https://staff.utia.cas.cz/novozada/db/IMD2020.zip" target="_blank" rel="noopener noreferrer">链接</a></td></tr><tr><td>FantasticReality</td><td>16592/19423</td><td>500x333</td><td>6000x4000</td><td>结合篡改定位与语义分割的多任务标注数据集，提供像素级篡改区域掩膜、实例分割和类别标签。</td><td><a href="https://drive.google.com/drive/folders/1AIKseHoL0ebux7tyeBDWN5zkHzAgYBzJ?usp=sharing" target="_blank" rel="noopener noreferrer">链接1</a><br><a href="https://github.com/mjkwon2021/CAT-Net/issues/51" target="_blank" rel="noopener noreferrer">链接2</a></td></tr><tr><td>PhotoShop-battle</td><td>11142/91886</td><td>高度：136<br>宽度：68</td><td>高度：20000<br>宽度：12024</td><td>面向创意篡改检测的大规模真实场景数据集，图像分辨率跨度大。</td><td><a href="https://github.com/dbisUnibas/PS-Battles" target="_blank" rel="noopener noreferrer">链接1</a><br><a href="https://www.kaggle.com/datasets/timocasti/psbattles/data" target="_blank" rel="noopener noreferrer">链接2</a></td></tr><tr><td>Carvalho（DSO-1）</td><td>100/100</td><td>2048x1536</td><td>2048x1536</td><td>通过拼接添加人物的splice操作和其他辅助处理实现伪造。</td><td><a href="http://ic.unicamp.br/~rocha/pub/downloads/2014-tiago-carvalho-thesis/tifs-database.zip" target="_blank" rel="noopener noreferrer">链接</a></td></tr><tr><td>GRIP Dataset</td><td>80/80</td><td>1024x768</td><td>1024x768</td><td>专为评估复制-移动篡改检测算法在复杂后处理下的鲁棒性设计。数据集篡改区域较小。</td><td><a href="https://www.grip.unina.it/download/prog/CMFD/" target="_blank" rel="noopener noreferrer">链接</a></td></tr><tr><td>CoMoFoD</td><td>260/260</td><td>512x512</td><td>3000x3000</td><td>专为copy-move检测算法评估设计，伪造图像和原始图像均应用了多种不同类型的后处理方法。</td><td><a href="https://www.vcl.fer.hr/comofod/download.html" target="_blank" rel="noopener noreferrer">链接</a></td></tr><tr><td>CocoGlide</td><td>512/512</td><td>256x256</td><td>256x256</td><td>针对生成式篡改研究，结合 GLIDE 扩散模型与语义提示生成篡改内容，模拟语义级局部篡改。</td><td><a href="https://www.grip.unina.it/download/prog/TruFor/CocoGlide.zip" target="_blank" rel="noopener noreferrer">链接</a></td></tr><tr><td>tampCOCO</td><td>0/800000</td><td>72x51</td><td>640x640</td><td>基于COCO 2017数据集构建，包括copy-move和splice两部分，所有图像均经过JPEG压缩，保留清晰边界以支持模型学习低级篡改痕迹。。</td><td><a href="https://www.kaggle.com/datasets/qsii24/tampcoco" target="_blank" rel="noopener noreferrer">链接</a></td></tr><tr><td>compRAISE</td><td>24462/0</td><td>2278x1515</td><td>6159x4079</td><td>基于高分辨率图像库和COCO实例标注构建的数据集。采取copy-move伪造，伪造区域为不规则形状，非矩形非对称。</td><td><a href="https://www.kaggle.com/datasets/qsii24/compraise" target="_blank" rel="noopener noreferrer">链接</a></td></tr><tr><td>OpenForensics</td><td>0/2000</td><td>512x512</td><td>512x512</td><td>首个面向多张人脸伪造检测与分割的大规模数据集。内容较丰富，图像场景涵盖室内与室外；人脸情况多样，尺寸变化点。</td><td><a href="https://zenodo.org/records/5528418" target="_blank" rel="noopener noreferrer">链接</a></td></tr></tbody></table><h2 id="_1-casia-v1-0和casia-v2-0" tabindex="-1"><a class="header-anchor" href="#_1-casia-v1-0和casia-v2-0"><span>1 CASIA v1.0和CASIA v2.0</span></a></h2><h3 id="_1-1-基本信息" tabindex="-1"><a class="header-anchor" href="#_1-1-基本信息"><span>1.1 基本信息</span></a></h3>',4)),t("ul",null,[t("li",null,[e[1]||(e[1]=a("简介：CASIA两个数据集均是由")),e[2]||(e[2]=t("strong",null,"中科院自动化所",-1)),e[3]||(e[3]=a("提供的篡改检测数据集，主要针对Splicing操作。特别的，官方")),l(n,{color:"red"},{default:i(()=>e[0]||(e[0]=[a("没有提供0-1Mask作为groundtruth，只有篡改图像和原始图像！")])),_:1}),e[4]||(e[4]=t("ul",null,[t("li",null,"CASIA V1.0包含921张篡改后的图像以及对应的原始图像，分辨率固定为384x256，篡改方式只有Splicing"),t("li",null,"CASIA V2.0包含5123张篡改后的图像以及对应的原始图像，分辨率从320x240到 800x600不等，在Splicing之外还使用了blurring。")],-1))]),e[5]||(e[5]=t("li",null,"APA格式引用：Dong, J., Wang, W., & Tan, T. (2013). CASIA Image Tampering Detection Evaluation Database. 2013 IEEE China Summit and International Conference on Signal and Information Processing, 422–426. https://doi.org/10.1109/ChinaSIP.2013.6625374",-1)),e[6]||(e[6]=t("li",null,[a("论文链接："),t("a",{href:"https://ieeexplore.ieee.org/abstract/document/6625374",target:"_blank",rel:"noopener noreferrer"},"CASIA Image Tampering Detection Evaluation Database | IEEE Conference Publication | IEEE Xplore")],-1))]),e[48]||(e[48]=r('<h3 id="_1-2-下载使用" tabindex="-1"><a class="header-anchor" href="#_1-2-下载使用"><span>1.2 下载使用</span></a></h3><p>​ 需要注意的是，原始的中科院自动化所网站目前仍处于维护中，无法使用。而数据集中有一些些命名相关的标注错误，且原始数据集没有提供0-1 mask作为GroundTruth。</p><p>​ 综上，目前一般使用的是由“Pham”等人在2019年一篇论文“Hybrid Image-Retrieval Method for Image-Splicing Validation”中，开源的带有0-1mask的CASIA数据集，由该论文的作者自己通过算法对原始图像和篡改图像求差值来得到带有0-1mask和对应图片的数据集，并且修正了文件名相关的错误。该数据集也广泛流传在Kaggle等数据科学平台中，下方提供Pham等人提供的Github链接：</p><ul><li>CASIA v1.0：https://github.com/namtpham/casia1groundtruth</li><li>CASIA v2.0：https://github.com/namtpham/casia2groundtruth</li></ul><p>​ 但是，Pham等人修正的CASIA v2.0仍然存在一些漏洞，基于该数据集，在DICALAB 的周吉喆老师指导下，马晓晨同学对于其中存在的一些谬误进行了修正后上传了一份新的修正后的数据集。除数十张图片的修改外，其余信息与Pham等人提供的数据集完全一致，可以通过如下Github链接下载：</p><ul><li>修正后的CASIA v2.0：https://github.com/SunnyHaze/CASIA2.0-Corrected-Groundtruth</li></ul><h2 id="_2-columbia" tabindex="-1"><a class="header-anchor" href="#_2-columbia"><span>2 Columbia</span></a></h2><h3 id="_2-1-基本信息" tabindex="-1"><a class="header-anchor" href="#_2-1-基本信息"><span>2.1 基本信息</span></a></h3><ul><li><p>简介：Columbia数据集是哥伦比亚大学DVMM Lab制作的篡改检测数据集，主要针对的也是Splicing操作。需要注意的是，官网有两个不同版本的数据集，其中一个为黑白固定尺寸（128 x 128），另一个为高清彩色splicing数据集，<strong>一般采用的是彩色的作为Benchmark。</strong></p><ul><li>黑白图片块数据集中有1800多个128x128的图片块</li><li>彩色Splicing数据集中有180张分辨率在757x568 到1152x768之间的图片块。</li></ul></li><li><p>论文链接：一般篡改图像相关论文只引用该数据集官网，但是原则上作者要求了需要引用他们的这篇论文：“Detecting Image Splicing Using Geometry Invariants And Camera Characteristics Consistency”（<a href="https://www.ee.columbia.edu/ln/dvmm/publications/06/hsu06ICMEcrf.pdf" target="_blank" rel="noopener noreferrer">hsu06ICMEcrf.pdf (columbia.edu)</a>）。</p></li><li><p>APA格式引用：Hsu, Y., &amp; Chang, S. (2006). Detecting Image Splicing using Geometry Invariants and Camera Characteristics Consistency. 2006 IEEE International Conference on Multimedia and Expo, 549–552. https://doi.org/10.1109/ICME.2006.262447</p></li></ul><h3 id="_2-2-下载使用" tabindex="-1"><a class="header-anchor" href="#_2-2-下载使用"><span>2.2 下载使用</span></a></h3><ul><li><p>Columbia Image Splicing Detection Evaluation Dataset（黑白图片块数据集）：https://www.ee.columbia.edu/ln/dvmm/downloads/AuthSplicedDataSet/AuthSplicedDataSet.htm</p></li><li><p>Columbia Uncompressed Image Splicing Detection Evaluation Dataset（彩色Splicing数据集）：https://www.ee.columbia.edu/ln/dvmm/downloads/authsplcuncmp/</p></li></ul><h2 id="_3-coverage" tabindex="-1"><a class="header-anchor" href="#_3-coverage"><span>3 Coverage</span></a></h2><h3 id="_3-1-基本信息" tabindex="-1"><a class="header-anchor" href="#_3-1-基本信息"><span>3.1 基本信息</span></a></h3><ul><li><p>简介：包含100对篡改图像及对应真实图像，主要应用Copy-move篡改操作。其图像具有很高的欺骗性，往往是将一堆相似物品中其中一个进行copy-move操作。</p></li><li><p>论文：COVERAGE – A NOVEL DATABASE FOR COPY-MOVE FORGERY DETECTION</p></li><li><p>APA格式引用：Wen, B., Zhu, Y., Subramanian, R., Ng, T.-T., Shen, X., &amp; Winkler, S. (2016). COVERAGE — A novel database for copy-move forgery detection. 2016 IEEE International Conference on Image Processing (ICIP), 161–165. https://doi.org/10.1109/ICIP.2016.7532339</p></li></ul><h3 id="_3-2-下载使用" tabindex="-1"><a class="header-anchor" href="#_3-2-下载使用"><span>3.2 下载使用</span></a></h3><ul><li>官方的github链接：https://github.com/wenbihan/coverage</li></ul><h2 id="_4-nist16" tabindex="-1"><a class="header-anchor" href="#_4-nist16"><span>4 NIST16</span></a></h2><h3 id="_4-1-基本信息" tabindex="-1"><a class="header-anchor" href="#_4-1-基本信息"><span>4.1 基本信息</span></a></h3><ul><li><p>简介：有splice、remove、copy-move三种篡改，不到1k张样本。一般使用的是NIST16做为篡改检测的评估。数据集内图像分辨率较高，一般需要进行resize处理或以滑动窗口进行Evaluation。需要注意的是数据集划分存在一定的“泄露”，即测试集图片再训练集中也存在的问题，但是大家一般对这个问题“睁一只眼闭一只眼”。</p></li><li><p>论文链接：https://ieeexplore.ieee.org/abstract/document/8638296/</p></li><li><p>APA格式引用：Guan, H., Kozak, M., Robertson, E., Lee, Y., Yates, A. N., Delgado, A., Zhou, D., Kheyrkhah, T., Smith, J., &amp; Fiscus, J. (2019). MFC Datasets: Large-Scale Benchmark Datasets for Media Forensic Challenge Evaluation. 2019 IEEE Winter Applications of Computer Vision Workshops (WACVW), 63–72. https://doi.org/10.1109/WACVW.2019.00018</p></li></ul><h3 id="_4-2-下载使用" tabindex="-1"><a class="header-anchor" href="#_4-2-下载使用"><span>4.2 下载使用</span></a></h3><p>这个数据集的官网使用比较麻烦，需要在NIST(National Institute of Standards and Technology，美国国家标准与技术研究所)官网注册后，获得许可才能进行下载。</p><ul><li><p>https://mfc.nist.gov/</p></li><li><p>https://mfc.nist.gov/participant</p></li></ul><p>理论上在一通操作后进入此网址https://mig.nist.gov/MFC/PubData/Resources.html下载上方的数据集，也就是一般常说的NIST16（更新的20的一般需要license）：</p><p>NC2016 Nimble Science Seta: NC2016 Test0613.SCI.tgz NC2016 Testset June 2013: NC2016 Test0613.tarbz2</p><p><strong>不这么麻烦的话，知乎有大哥传网盘了，非官方的获取方式如下（不保证长久使用）：</strong></p><p>百度网盘链接：https://pan.baidu.com/share/init?surl=nZVUaqaBOD1u3xtxVDKmFw</p><p>提取码：lik7</p><h2 id="_5-defacto" tabindex="-1"><a class="header-anchor" href="#_5-defacto"><span>5 Defacto</span></a></h2><h3 id="_5-1-基本信息" tabindex="-1"><a class="header-anchor" href="#_5-1-基本信息"><span>5.1 基本信息</span></a></h3><ul><li><p>简介：Defacto数据集是基于MSCOCO生成的篡改图像数据集，内容非常的巨大，但是“总体篡改区域占全图面积的比例”非常小，即平均下来大部分篡改只是对于图片中非常小的物体进行操作，分布与其他的数据集还是区别较大的。</p></li><li><p>论文链接：https://ieeexplore.ieee.org/abstract/document/8903181/</p></li><li><p>APA格式引用：Mahfoudi, G., Tajini, B., Retraint, F., Morain-Nicolier, F., Dugelay, J. L., &amp; Pic, M. (2019). DEFACTO: Image and Face Manipulation Dataset. 2019 27th European Signal Processing Conference (EUSIPCO), 1–5. https://doi.org/10.23919/EUSIPCO.2019.8903181</p></li></ul><p>此外，因为生成自COCO，作者还要求对于MS COCO数据集的进行引用：</p><ul><li><p>论文链接：https://arxiv.org/abs/1405.0312</p></li><li><p>APA格式引用：Lin, T.-Y., Maire, M., Belongie, S., Bourdev, L., Girshick, R., Hays, J., Perona, P., Ramanan, D., Zitnick, C. L., &amp; Dollár, P. (2015). Microsoft COCO: Common Objects in Context (arXiv:1405.0312). arXiv. http://arxiv.org/abs/1405.0312</p></li></ul><h3 id="_5-2-下载使用" tabindex="-1"><a class="header-anchor" href="#_5-2-下载使用"><span>5.2 下载使用</span></a></h3><p>该数据集已经完全上传至Kaggle，所以还是比较好下载使用的。因为非常巨大，作者按照篡改类别分为了几个部分提供下载：</p><p>Inpainting：</p><p>https://www.kaggle.com/datasets/defactodataset/defactoinpainting</p><p>Copy-move：</p><p>https://www.kaggle.com/datasets/defactodataset/defactocopymove</p><p>Splicing：</p><p>https://www.kaggle.com/datasets/defactodataset/defactosplicing</p><h2 id="_6-imd2020" tabindex="-1"><a class="header-anchor" href="#_6-imd2020"><span>6 IMD2020</span></a></h2><h3 id="_6-1-基本信息" tabindex="-1"><a class="header-anchor" href="#_6-1-基本信息"><span>6.1 基本信息</span></a></h3><ul><li><p>简介：IMD2020是由捷克科学院信息理论与自动化研究所团队构建的大规模篡改图像检测数据集，包含合成生成和真实篡改两部分，前者是基于35,000 张由 2,322 种不同相机型号拍摄的真实图像，通过使用大量图像操作方法（包括图像处理技术以及基于 GAN 或修复的方法）合成了一组被篡改的图像；后者为从互联网上收集的2000 张由随机人员创建的“真实生活”（不受控）的被篡改图像。提供精确的二进制掩码（0-1 Mask）标注。</p><ul><li>合成生成数据集： <ul><li>包含35,000张真实图像（来自2,322种相机模型）及对应35,000张篡改图像，总计70,000张。</li><li>篡改方法涵盖传统处理（JPEG压缩、模糊、噪声等）、GAN生成（如FaceApp）、图像修复（Inpainting）等，篡改区域占比5%-30%。</li><li>所有篡改图像均提供二进制掩码标注篡改区域。</li></ul></li><li>真实生活数据集： <ul><li>包含2,010张从互联网收集的未控制篡改图像，每张均匹配原始图像，并手动标注二进制掩码。</li><li>篡改类型包括复杂现实编辑（如拼接、局部修饰），分辨率不固定</li></ul></li></ul></li><li><p>论文链接：<a href="https://ieeexplore.ieee.org/document/9096940" target="_blank" rel="noopener noreferrer">https://ieeexplore.ieee.org/document/9096940/</a></p></li><li><p>APA格式引用：Novozamsky, A., Mahdian, B., &amp; Saic, S. (2020). IMD2020: A large-scale annotated dataset tailored for detecting manipulated images. In Proceedings of the IEEE/CVF winter conference on applications of computer vision workshops (pp. 71-80).<a href="https://doi.org/10.1109/WACVW50321.2020.9096940" target="_blank" rel="noopener noreferrer">https://doi.org/10.1109/WACVW50321.2020.9096940</a></p></li></ul><h3 id="_6-2-下载使用" tabindex="-1"><a class="header-anchor" href="#_6-2-下载使用"><span>6.2 下载使用</span></a></h3><p>官方下载网址：https://staff.utia.cas.cz/novozada/db/</p><ul><li><p>真实篡改数据集为：IMD2020 Real-Life Manipulated Images部分</p></li><li><p>合成篡改数据集：IMD2020 Large-Scale Set of Inpainting Images部分</p></li></ul><h2 id="_7-fantasticreality" tabindex="-1"><a class="header-anchor" href="#_7-fantasticreality"><span>7 FantasticReality</span></a></h2><h3 id="_7-1-基本信息" tabindex="-1"><a class="header-anchor" href="#_7-1-基本信息"><span>7.1 基本信息</span></a></h3>',48)),t("ul",null,[t("li",null,[e[8]||(e[8]=a("简介FantasticReality数据集是由俄罗斯国家航空系统研究院（GosNIIAS）、莫斯科物理技术学院（MIPT）和布鲁诺·凯斯勒基金会（FBK）联合构建的大规模篡改检测数据集，旨在解决现有数据集规模小、标注不全面的问题。同时提供像素级篡改区域掩膜（ground truth mask）、实例分割和类别标签，涵盖10个常见对象类别（如人、车、建筑等），是首个")),l(n,{color:"red"},{default:i(()=>e[7]||(e[7]=[a("结合篡改定位与语义分割的多任务")])),_:1}),e[9]||(e[9]=a("标注数据集。包含16k真实图像和16k篡改图像，总计32k张图像，篡改方式主要为Splicing。"))]),e[10]||(e[10]=t("li",null,[a("论文链接："),t("a",{href:"https://papers.nips.cc/paper_files/paper/2019/hash/98dce83da57b0395e163467c9dae521b-Abstract.html",target:"_blank",rel:"noopener noreferrer"}," The Point Where Reality Meets Fantasy: Mixed Adversarial Generators for Image Splice Detection")],-1)),e[11]||(e[11]=t("li",null,[a("APA格式引用：Kniaz, V. V., Knyaz, V., & Remondino, F. (2019). The point where reality meets fantasy: Mixed adversarial generators for image splice detection. "),t("em",null,"Advances in neural information processing systems"),a(", "),t("em",null,"32"),a(".")],-1))]),e[49]||(e[49]=r('<h3 id="_7-2-下载使用" tabindex="-1"><a class="header-anchor" href="#_7-2-下载使用"><span>7.2 下载使用</span></a></h3><ul><li><p>原论文给出数据集下载链接：http://zefirus.org/MAG （已失效）</p></li><li><p>作者在CAT-Net的github仓库中给出了新的下载链接:https://drive.google.com/drive/folders/1AIKseHoL0ebux7tyeBDWN5zkHzAgYBzJ?usp=sharing, 但是好像还是存在问题。</p></li><li><p>其他方式可以考虑联系作者获取下载权限，具体参见https://github.com/mjkwon2021/CAT-Net/issues/51， 作者给出了具体的联系方式以及相关要求。</p></li></ul><h2 id="_8-photoshop-battle" tabindex="-1"><a class="header-anchor" href="#_8-photoshop-battle"><span>8 PhotoShop-battle</span></a></h2><h3 id="_8-1-基本信息" tabindex="-1"><a class="header-anchor" href="#_8-1-基本信息"><span>8.1 基本信息</span></a></h3>',4)),t("ul",null,[t("li",null,[e[13]||(e[13]=a("简介：PS-Battles数据集由瑞士巴塞尔大学基于Reddit社区")),e[14]||(e[14]=t("strong",null,"r/photoshopbattles",-1)),e[15]||(e[15]=a("构建，是首个面向")),l(n,{color:"red"},{default:i(()=>e[12]||(e[12]=[a("创意篡改检测")])),_:1}),e[16]||(e[16]=a("的大规模真实场景数据集。该数据集聚焦于社区用户生成的多样化、高语义性图像篡改内容，包含11,142组图像（总计103,028张），篡改类型包括：幽默合成、场景替换、角色融合（包括Splicing, copy-move, removal）。图像分辨率跨度大（宽度68~12,024像素，高度136~20,000像素）。"))]),e[17]||(e[17]=t("li",null,[a("论文链接："),t("a",{href:"https://arxiv.org/abs/1804.04866",target:"_blank",rel:"noopener noreferrer"},"The PS-Battles Dataset - an Image Collection for Image Manipulation Detection")],-1)),e[18]||(e[18]=t("li",null,[a("APA格式引用：Heller, S., Rossetto, L., & Schuldt, H. (2018). The ps-battles dataset-an image collection for image manipulation detection. "),t("em",null,"arXiv preprint arXiv:1804.04866"),a(".https://arxiv.org/abs/1804.04866、")],-1))]),e[50]||(e[50]=r('<h3 id="_8-2-下载使用" tabindex="-1"><a class="header-anchor" href="#_8-2-下载使用"><span>8.2 下载使用</span></a></h3><p>数据集官方github仓库：https://github.com/dbisUnibas/PS-Battles</p><p>Kaggle：https://www.kaggle.com/datasets/timocasti/psbattles/data</p><ul><li>ubuntu&amp;MacOS下载：具体见github仓库，运行提供的<code>download.sh</code>脚本即可</li><li>Windows下载：具体见Kaggle网址，将提供的download.py脚本和Originals.tsv＆photoshops.tsv放在同一目录中，然后运行<code>download.py</code>脚本即可。</li></ul><h2 id="_9-carvalho-dso-1" tabindex="-1"><a class="header-anchor" href="#_9-carvalho-dso-1"><span>9 Carvalho（DSO-1）</span></a></h2><h3 id="_9-1-基本信息" tabindex="-1"><a class="header-anchor" href="#_9-1-基本信息"><span>9.1 基本信息</span></a></h3>',6)),t("ul",null,[t("li",null,[e[20]||(e[20]=a("简介：DSO-1数据集包含200张高分辨率图像（2048×1536像素），其中100张为原始未修改图像，100张为伪造图像。通过")),l(n,{color:"red"},{default:i(()=>e[19]||(e[19]=[a("拼接添加人物")])),_:1}),e[21]||(e[21]=a("（在已经包含一个人或多个人的源图像中添加一个人或多个人，splicing操作）并辅以色彩、亮度调整实现伪造。"))]),e[22]||(e[22]=t("li",null,[a("论文链接："),t("a",{href:"https://ieeexplore.ieee.org/document/6522874",target:"_blank",rel:"noopener noreferrer"},"Exposing Digital Image Forgeries by Illumination Color Classification")],-1)),e[23]||(e[23]=t("li",null,[a("APA格式引用：De Carvalho, T. J., Riess, C., Angelopoulou, E., Pedrini, H., & de Rezende Rocha, A. (2013). Exposing digital image forgeries by illumination color classification. "),t("em",null,"IEEE Transactions on Information Forensics and Security"),a(", "),t("em",null,"8"),a("(7), 1182-1194."),t("a",{href:"https://doi.org/10.1109/TIFS.2013.2265677",target:"_blank",rel:"noopener noreferrer"},"https://doi.org/10.1109/TIFS.2013.2265677")],-1))]),e[51]||(e[51]=r('<h3 id="_9-2-下载使用" tabindex="-1"><a class="header-anchor" href="#_9-2-下载使用"><span>9.2 下载使用</span></a></h3><p>下载地址：http://ic.unicamp.br/~rocha/pub/downloads/2014-tiago-carvalho-thesis/tifs-database.zip（貌似现在有问题，不会成功下载）</p><p>相关数据集合集：https://recodbr.wordpress.com/code-n-data/#porno</p><h2 id="_10-grip-dataset" tabindex="-1"><a class="header-anchor" href="#_10-grip-dataset"><span>10 GRIP Dataset</span></a></h2><h3 id="_10-1-基本信息" tabindex="-1"><a class="header-anchor" href="#_10-1-基本信息"><span>10.1 基本信息</span></a></h3>',5)),t("ul",null,[t("li",null,[e[25]||(e[25]=a("简介：数据集专为评估")),l(n,{color:"red"},{default:i(()=>e[24]||(e[24]=[a("复制-移动篡改检测")])),_:1}),e[26]||(e[26]=a("算法在复杂后处理干扰下的鲁棒性设计。该数据集聚焦小规模篡改区域检测，包含80张图像，关注过往数据集篡改区域过大、后处理类型单一等问题。包含高分辨率图像（如768×1024像素），篡改区域面积覆盖4000像素（<1%）至50,000像素，分为平滑、混合、纹理三类背景复杂度。提供像素级真值掩膜（0-1 mask）"))]),e[27]||(e[27]=t("li",null,[a("论文链接："),t("a",{href:"https://ieeexplore.ieee.org/document/7154457",target:"_blank",rel:"noopener noreferrer"},"Efficient Dense-Field Copy–Move Forgery Detection")],-1)),e[28]||(e[28]=t("li",null,[a("APA格式引用：Cozzolino, D., Poggi, G., & Verdoliva, L. (2015). Efficient dense-field copy–move forgery detection. "),t("em",null,"IEEE Transactions on Information Forensics and Security"),a(", "),t("em",null,"10"),a("(11), 2284-2297."),t("a",{href:"https://doi.org/10.1109/TIFS.2015.2455334",target:"_blank",rel:"noopener noreferrer"},"https://doi.org/10.1109/TIFS.2015.2455334")],-1))]),e[52]||(e[52]=r('<h3 id="_10-2-下载使用" tabindex="-1"><a class="header-anchor" href="#_10-2-下载使用"><span>10.2 下载使用</span></a></h3><p>论文作者 <a href="http://www.unina.it/" target="_blank" rel="noopener noreferrer">University Federico II of Naples</a> 的 <a href="https://www.grip.unina.it/" target="_blank" rel="noopener noreferrer">GRIP组</a>提供了该数据集的下载内容：https://www.grip.unina.it/download/prog/CMFD/</p><h2 id="_11-comofod" tabindex="-1"><a class="header-anchor" href="#_11-comofod"><span>11 CoMoFoD</span></a></h2><h3 id="_11-1-基本信息" tabindex="-1"><a class="header-anchor" href="#_11-1-基本信息"><span>11.1 基本信息</span></a></h3>',4)),t("ul",null,[t("li",null,[t("p",null,[e[30]||(e[30]=a("简介：CoMoFoD（Copy-Move Forgery Detection）数据库由克罗地亚萨格勒布大学电气工程与计算学院开发，是专为")),l(n,{color:"red"},{default:i(()=>e[29]||(e[29]=[a("复制-移动篡改")])),_:1}),e[31]||(e[31]=a("检测算法评估设计的综合性基准数据集。系统化整合多种几何变换与后处理操作，并提供像素级篡改掩膜标注，旨在解决现有数据集后处理类型单一、规模不足的问题。包含 260 组伪造图像，分为两个类别，其中小图像类别（512x512）有 200 组图像集，大图像类别（3000x2000）有 60 组图像图像集。所有伪造图像和原始图像都应用了不同类型的后处理方法，例如 JPEG 压缩、模糊、添加噪声、颜色减少等。"))])]),e[32]||(e[32]=t("li",null,[t("p",null,[a("论文链接："),t("a",{href:"https://ieeexplore.ieee.org/document/6658316",target:"_blank",rel:"noopener noreferrer"},"CoMoFoD — New database for copy-move forgery detection")])],-1)),e[33]||(e[33]=t("li",null,[t("p",null,[a("APA格式引用：Tralic, D., Zupancic, I., Grgic, S., & Grgic, M. (2013, September). CoMoFoD—New database for copy-move forgery detection. In "),t("em",null,"Proceedings ELMAR-2013"),a(" (pp. 49-54). IEEE.")])],-1))]),e[53]||(e[53]=r('<h3 id="_11-2-下载使用" tabindex="-1"><a class="header-anchor" href="#_11-2-下载使用"><span>11.2 下载使用</span></a></h3><ul><li>数据集官方网址:https://www.vcl.fer.hr/comofod/download.html ，其中 <ul><li>Small image category database（512x512）共200 组，下载连接：https://www.vcl.fer.hr/comofod/comofod_small.rar</li><li>Large image category database（3000x2000）共60 组，无下载链接，需联系作者申请。</li></ul></li></ul><h2 id="_12-cocoglide" tabindex="-1"><a class="header-anchor" href="#_12-cocoglide"><span>12 CocoGlide</span></a></h2><h3 id="_12-1-基本信息" tabindex="-1"><a class="header-anchor" href="#_12-1-基本信息"><span>12.1 基本信息</span></a></h3>',4)),t("ul",null,[t("li",null,[e[35]||(e[35]=a("简介：CocoGlide 是基于 COCO 2017 验证集构建的篡改检测数据集，专为评估现代生成模型（如扩散模型）的局部篡改检测能力设计。其通过结合 GLIDE 扩散模型与语义提示生成逼真篡改内容，替换原始图像中的对应区域（如动物、交通工具等），模拟真实场景下的语义级局部篡改。包含 ")),e[36]||(e[36]=t("strong",null,"512 张篡改图像",-1)),e[37]||(e[37]=a("，均通过 COCO 验证集的 256×256 像素裁剪块生成，填补了传统数据集中缺乏")),l(n,{color:"red"},{default:i(()=>e[34]||(e[34]=[a("生成式篡改")])),_:1}),e[38]||(e[38]=a("样本的空白。"))]),e[39]||(e[39]=t("li",null,[a("论文链接："),t("a",{href:"https://arxiv.org/abs/2212.10957",target:"_blank",rel:"noopener noreferrer"},"TruFor: Leveraging all-round clues for trustworthy image forgery detection and localization")],-1)),e[40]||(e[40]=t("li",null,[a("APA格式引用：Guillaro, F., Cozzolino, D., Sud, A., Dufour, N., & Verdoliva, L. (2023). Trufor: Leveraging all-round clues for trustworthy image forgery detection and localization. In "),t("em",null,"Proceedings of the IEEE/CVF conference on computer vision and pattern recognition"),a(" (pp. 20606-20615).")],-1))]),e[54]||(e[54]=r('<h3 id="_12-2-下载使用" tabindex="-1"><a class="header-anchor" href="#_12-2-下载使用"><span>12.2 下载使用</span></a></h3><p>论文作者 <a href="http://www.unina.it/" target="_blank" rel="noopener noreferrer">University Federico II of Naples</a> 的 <a href="https://www.grip.unina.it/" target="_blank" rel="noopener noreferrer">GRIP组</a>提供了该数据集的下载内容：https://www.grip.unina.it/download/prog/TruFor/CocoGlide.zip</p><h2 id="_13-tampcoco" tabindex="-1"><a class="header-anchor" href="#_13-tampcoco"><span>13 tampCOCO</span></a></h2><h3 id="_13-1-基本信息" tabindex="-1"><a class="header-anchor" href="#_13-1-基本信息"><span>13.1 基本信息</span></a></h3><ul><li><p>简介：tampCOCO是由COCO 2017数据集构建的篡改检测数据集，包含SP COCO（跨图像拼接）和CM COCO（同图像复制-移动）两部分。提供像素级二值掩码（0-1 Mask）。</p><ul><li><strong>SP COCO</strong>： <ul><li>基于COCO图像，从一张图中随机选取物体（如人物、车辆等），经旋转/缩放后粘贴至另一张图的随机位置。</li><li>总计20万张伪造图像，所有图像均经过JPEG压缩（质量因子60-100），未添加模糊等后处理。</li></ul></li><li><strong>CM COCO</strong>： <ul><li>在单张COCO图像内复制选定区域（如物体或背景）并粘贴至其他位置，生成复制-移动篡改样本。</li><li>总计60万张图像，JPEG压缩参数与SP COCO一致，保留清晰边界以支持模型学习低级篡改痕迹。</li></ul></li></ul></li><li><p>论文链接：<a href="https://arxiv.org/abs/2108.12947" target="_blank" rel="noopener noreferrer">Learning JPEG Compression Artifacts for Image Manipulation Detection and Localization</a></p></li><li><p>APA格式引用：Kwon, M. J., Nam, S. H., Yu, I. J., Lee, H. K., &amp; Kim, C. (2022). Learning jpeg compression artifacts for image manipulation detection and localization. <em>International Journal of Computer Vision</em>, <em>130</em>(8), 1875-1895.https://arxiv.org/abs/2108.12947</p></li></ul><h3 id="_13-2-下载使用" tabindex="-1"><a class="header-anchor" href="#_13-2-下载使用"><span>13.2 下载使用</span></a></h3><p>该数据集已经完全上传至Kaggle，https://www.kaggle.com/datasets/qsii24/tampcoco。</p><p>由于数据集较大，一共分为了13个部分供下载，以上链接导向全部下载的索引。</p><h2 id="_14-compraise" tabindex="-1"><a class="header-anchor" href="#_14-compraise"><span>14 compRAISE</span></a></h2><h3 id="_14-1-基本信息" tabindex="-1"><a class="header-anchor" href="#_14-1-基本信息"><span>14.1 基本信息</span></a></h3><ul><li>简介：compRAISE（CM RAISE）是基于<strong>RAISE高分辨率图像库</strong>与<strong>COCO实例标注</strong>构建的复杂篡改检测数据集。数据集篡改方式为从RAISE数据集（含8,156张未压缩RAW图像）中选取高分辨率自然场景图像（分辨率范围：2,000×3,008 ~ 4,928×3,264）,之后借用COCO 2017的随机多边形实例掩码（约120万标注），从RAISE图像中提取不规则形状区域，确保篡改边界非矩形、非对称，并在单张单张RAISE图像内，执行<strong>复制-移动策略</strong>伪造。</li><li>论文链接：<a href="https://arxiv.org/abs/2108.12947" target="_blank" rel="noopener noreferrer">Learning JPEG Compression Artifacts for Image Manipulation Detection and Localization</a></li><li>APA格式引用：Kwon, M. J., Nam, S. H., Yu, I. J., Lee, H. K., &amp; Kim, C. (2022). Learning jpeg compression artifacts for image manipulation detection and localization. <em>International Journal of Computer Vision</em>, <em>130</em>(8), 1875-1895.https://arxiv.org/abs/2108.12947</li></ul><h3 id="_14-2-下载使用" tabindex="-1"><a class="header-anchor" href="#_14-2-下载使用"><span>14.2 下载使用</span></a></h3><p>该数据集已经完全上传至Kaggle，https://www.kaggle.com/datasets/qsii24/compraise。</p><p>由于数据集较大，一共分为了15个部分供下载，以上链接导向全部下载的索引。</p><h2 id="_15-openforensics" tabindex="-1"><a class="header-anchor" href="#_15-openforensics"><span>15 OpenForensics</span></a></h2><h3 id="_15-1-基本信息" tabindex="-1"><a class="header-anchor" href="#_15-1-基本信息"><span>15.1 基本信息</span></a></h3>',16)),t("ul",null,[t("li",null,[e[42]||(e[42]=a("简介：OpenForensics是由日本国立信息学研究所、综合研究大学院大学和东京大学联合构建的首个面向")),l(n,{color:"red"},{default:i(()=>e[41]||(e[41]=[a("多张人脸伪造检测与分割")])),_:1}),e[43]||(e[43]=a("的大规模数据集。该数据集专为复杂自然场景下的多任务研究设计，提供像素级精细标注，支持伪造检测、实例分割、伪造边界识别等多维度任务。 ")),e[44]||(e[44]=t("ul",null,[t("li",null,"包含115,325张图像，总计334,136张人脸（平均每图2.9张人脸），其中真实人脸160,670张，伪造人脸173,660张。"),t("li",null,"划分为训练集（44K+图像）、验证集（7K+图像）、测试开发集（18K+图像）和测试挑战集（45K+图像）。"),t("li",null,"图像场景涵盖室内（63.7%）与室外（36.3%）；人脸姿态、年龄、性别、遮挡情况高度多样，包含微小至大尺寸人脸。"),t("li",null,"伪造人脸分辨率达512×512。")],-1))]),e[45]||(e[45]=t("li",null,[a("论文链接："),t("a",{href:"https://arxiv.org/abs/2107.14480",target:"_blank",rel:"noopener noreferrer"},"OpenForensics: Large-Scale Challenging Dataset For Multi-Face Forgery Detection And Segmentation In-The-Wild")],-1)),e[46]||(e[46]=t("li",null,[a("APA格式引用：Le, T. N., Nguyen, H. H., Yamagishi, J., & Echizen, I. (2021). Openforensics: Large-scale challenging dataset for multi-face forgery detection and segmentation in-the-wild. In "),t("em",null,"Proceedings of the IEEE/CVF international conference on computer vision"),a(" (pp. 10117-10127).")],-1))]),e[55]||(e[55]=t("h3",{id:"_15-2-下载使用",tabindex:"-1"},[t("a",{class:"header-anchor",href:"#_15-2-下载使用"},[t("span",null,"15.2 下载使用")])],-1)),e[56]||(e[56]=t("p",null,"官方下载链接：https://zenodo.org/records/5528418",-1)),e[57]||(e[57]=t("p",null,"其中数据集划分为多个部分供下载，以上链接导向全部下载的索引。",-1)),l(s)])}const f=d(g,[["render",u]]),b=JSON.parse('{"path":"/zh/imdl_data_model_hub/data/IMDLdatasets.html","title":"# 篡改检测数据集索引","lang":"zh-CN","frontmatter":{"description":"# 篡改检测数据集索引 1 CASIA v1.0和CASIA v2.0 1.1 基本信息 简介：CASIA两个数据集均是由中科院自动化所提供的篡改检测数据集，主要针对Splicing操作。特别的，官方 CASIA V1.0包含921张篡改后的图像以及对应的原始图像，分辨率固定为384x256，篡改方式只有Splicing CASIA V2.0包含512...","head":[["link",{"rel":"alternate","hreflang":"en-us","href":"https://scu-zjz.github.io/IMDLBenCo-doc/IMDLBenCo-doc/imdl_data_model_hub/data/IMDLdatasets.html"}],["meta",{"property":"og:url","content":"https://scu-zjz.github.io/IMDLBenCo-doc/IMDLBenCo-doc/zh/imdl_data_model_hub/data/IMDLdatasets.html"}],["meta",{"property":"og:site_name","content":"IMDLBenCo 文档"}],["meta",{"property":"og:title","content":"# 篡改检测数据集索引"}],["meta",{"property":"og:description","content":"# 篡改检测数据集索引 1 CASIA v1.0和CASIA v2.0 1.1 基本信息 简介：CASIA两个数据集均是由中科院自动化所提供的篡改检测数据集，主要针对Splicing操作。特别的，官方 CASIA V1.0包含921张篡改后的图像以及对应的原始图像，分辨率固定为384x256，篡改方式只有Splicing CASIA V2.0包含512..."}],["meta",{"property":"og:type","content":"article"}],["meta",{"property":"og:locale","content":"zh-CN"}],["meta",{"property":"og:locale:alternate","content":"en-US"}],["meta",{"property":"og:updated_time","content":"2025-04-09T03:30:10.000Z"}],["meta",{"property":"article:modified_time","content":"2025-04-09T03:30:10.000Z"}],["script",{"type":"application/ld+json"},"{\\"@context\\":\\"https://schema.org\\",\\"@type\\":\\"Article\\",\\"headline\\":\\"# 篡改检测数据集索引\\",\\"image\\":[\\"\\"],\\"dateModified\\":\\"2025-04-09T03:30:10.000Z\\",\\"author\\":[]}"]]},"headers":[{"level":2,"title":"1 CASIA v1.0和CASIA v2.0","slug":"_1-casia-v1-0和casia-v2-0","link":"#_1-casia-v1-0和casia-v2-0","children":[{"level":3,"title":"1.1 基本信息","slug":"_1-1-基本信息","link":"#_1-1-基本信息","children":[]},{"level":3,"title":"1.2 下载使用","slug":"_1-2-下载使用","link":"#_1-2-下载使用","children":[]}]},{"level":2,"title":"2 Columbia","slug":"_2-columbia","link":"#_2-columbia","children":[{"level":3,"title":"2.1 基本信息","slug":"_2-1-基本信息","link":"#_2-1-基本信息","children":[]},{"level":3,"title":"2.2 下载使用","slug":"_2-2-下载使用","link":"#_2-2-下载使用","children":[]}]},{"level":2,"title":"3 Coverage","slug":"_3-coverage","link":"#_3-coverage","children":[{"level":3,"title":"3.1 基本信息","slug":"_3-1-基本信息","link":"#_3-1-基本信息","children":[]},{"level":3,"title":"3.2 下载使用","slug":"_3-2-下载使用","link":"#_3-2-下载使用","children":[]}]},{"level":2,"title":"4 NIST16","slug":"_4-nist16","link":"#_4-nist16","children":[{"level":3,"title":"4.1 基本信息","slug":"_4-1-基本信息","link":"#_4-1-基本信息","children":[]},{"level":3,"title":"4.2 下载使用","slug":"_4-2-下载使用","link":"#_4-2-下载使用","children":[]}]},{"level":2,"title":"5 Defacto","slug":"_5-defacto","link":"#_5-defacto","children":[{"level":3,"title":"5.1 基本信息","slug":"_5-1-基本信息","link":"#_5-1-基本信息","children":[]},{"level":3,"title":"5.2 下载使用","slug":"_5-2-下载使用","link":"#_5-2-下载使用","children":[]}]},{"level":2,"title":"6 IMD2020","slug":"_6-imd2020","link":"#_6-imd2020","children":[{"level":3,"title":"6.1 基本信息","slug":"_6-1-基本信息","link":"#_6-1-基本信息","children":[]},{"level":3,"title":"6.2 下载使用","slug":"_6-2-下载使用","link":"#_6-2-下载使用","children":[]}]},{"level":2,"title":"7 FantasticReality","slug":"_7-fantasticreality","link":"#_7-fantasticreality","children":[{"level":3,"title":"7.1 基本信息","slug":"_7-1-基本信息","link":"#_7-1-基本信息","children":[]},{"level":3,"title":"7.2 下载使用","slug":"_7-2-下载使用","link":"#_7-2-下载使用","children":[]}]},{"level":2,"title":"8 PhotoShop-battle","slug":"_8-photoshop-battle","link":"#_8-photoshop-battle","children":[{"level":3,"title":"8.1 基本信息","slug":"_8-1-基本信息","link":"#_8-1-基本信息","children":[]},{"level":3,"title":"8.2 下载使用","slug":"_8-2-下载使用","link":"#_8-2-下载使用","children":[]}]},{"level":2,"title":"9 Carvalho（DSO-1）","slug":"_9-carvalho-dso-1","link":"#_9-carvalho-dso-1","children":[{"level":3,"title":"9.1 基本信息","slug":"_9-1-基本信息","link":"#_9-1-基本信息","children":[]},{"level":3,"title":"9.2 下载使用","slug":"_9-2-下载使用","link":"#_9-2-下载使用","children":[]}]},{"level":2,"title":"10 GRIP Dataset","slug":"_10-grip-dataset","link":"#_10-grip-dataset","children":[{"level":3,"title":"10.1 基本信息","slug":"_10-1-基本信息","link":"#_10-1-基本信息","children":[]},{"level":3,"title":"10.2 下载使用","slug":"_10-2-下载使用","link":"#_10-2-下载使用","children":[]}]},{"level":2,"title":"11 CoMoFoD","slug":"_11-comofod","link":"#_11-comofod","children":[{"level":3,"title":"11.1 基本信息","slug":"_11-1-基本信息","link":"#_11-1-基本信息","children":[]},{"level":3,"title":"11.2 下载使用","slug":"_11-2-下载使用","link":"#_11-2-下载使用","children":[]}]},{"level":2,"title":"12 CocoGlide","slug":"_12-cocoglide","link":"#_12-cocoglide","children":[{"level":3,"title":"12.1 基本信息","slug":"_12-1-基本信息","link":"#_12-1-基本信息","children":[]},{"level":3,"title":"12.2 下载使用","slug":"_12-2-下载使用","link":"#_12-2-下载使用","children":[]}]},{"level":2,"title":"13 tampCOCO","slug":"_13-tampcoco","link":"#_13-tampcoco","children":[{"level":3,"title":"13.1 基本信息","slug":"_13-1-基本信息","link":"#_13-1-基本信息","children":[]},{"level":3,"title":"13.2 下载使用","slug":"_13-2-下载使用","link":"#_13-2-下载使用","children":[]}]},{"level":2,"title":"14 compRAISE","slug":"_14-compraise","link":"#_14-compraise","children":[{"level":3,"title":"14.1 基本信息","slug":"_14-1-基本信息","link":"#_14-1-基本信息","children":[]},{"level":3,"title":"14.2 下载使用","slug":"_14-2-下载使用","link":"#_14-2-下载使用","children":[]}]},{"level":2,"title":"15 OpenForensics","slug":"_15-openforensics","link":"#_15-openforensics","children":[{"level":3,"title":"15.1 基本信息","slug":"_15-1-基本信息","link":"#_15-1-基本信息","children":[]},{"level":3,"title":"15.2 下载使用","slug":"_15-2-下载使用","link":"#_15-2-下载使用","children":[]}]}],"git":{"updatedTime":1744169410000,"contributors":[{"name":"Ma Xiaochen (马晓晨)","username":"","email":"mxch1122@126.com","commits":2},{"name":"Sylence8","username":"Sylence8","email":"98306351+Sylence8@users.noreply.github.com","commits":2,"url":"https://github.com/Sylence8"}],"changelog":[{"hash":"66b7b2107aa4fa6e367a97269ff5bbec7f5b9fbe","time":1744169410000,"email":"98306351+Sylence8@users.noreply.github.com","author":"Sylence8","message":"Update IMDLdatasets.md"},{"hash":"4ee557065e458935b27244864205048069f2c42b","time":1744169217000,"email":"98306351+Sylence8@users.noreply.github.com","author":"Sylence8","message":"Update IMDLdatasets.md"},{"hash":"3adb8aa2f1fd3408eb9837aa011bf7048ae6230b","time":1743675138000,"email":"mxch1122@126.com","author":"Ma Xiaochen (马晓晨)","message":"[update] add comment plugin to all pages."},{"hash":"2e30180fb8a110e32a3aa5fe308d2e0a030c47ec","time":1743364035000,"email":"mxch1122@126.com","author":"Ma Xiaochen (马晓晨)","message":"Revise sturcture, add more infor about data &amp; models."}]},"filePathRelative":"zh/imdl_data_model_hub/data/IMDLdatasets.md","autoDesc":true}');export{f as comp,b as data};
