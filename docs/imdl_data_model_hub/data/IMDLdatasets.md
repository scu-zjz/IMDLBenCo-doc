# Tampering Detection Dataset Index

| Dataset Name      | Genuine/Tampered Image Count                                 | Minimum Resolution                                           | Maximum Resolution                                           | Image Features Summary                                       | Current Download Links                                       |
| ----------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| CASIA v1.0        | 800/921                                                      | 384x256                                                      | 384x256                                                      | Primarily targeting splice operations.                       | [Link](https://github.com/namtpham/casia1groundtruth)        |
| CASIA v2.0        | 7491/5123                                                    | 320x240                                                      | 800x600                                                      | Distinguishes between copy-move and splice tampering methods. | [Link](https://github.com/namtpham/casia2groundtruth)<br />[Corrected Version Link](https://github.com/SunnyHaze/CASIA2.0-Corrected-Groundtruth) |
| Columbia          | Black and White Dataset: 933/921<br />Color Dataset: 183/180 | Black and White Dataset: 128x128<br />Color Dataset: 757x568 | Black and White Dataset: 128x128<br />Color Dataset: 1152x768 | Splice tampering based on uncompressed images, with higher image resolutions. | [Black and White Dataset]()<br />[Color Dataset](https://www.ee.columbia.edu/ln/dvmm/downloads/authsplcuncmp/) |
| Coverage          | 100/100                                                      | 334x190                                                      | 752x472                                                      | Targeting copy-move tampering operations, typically copying one item from a group of similar items. | [Link](https://github.com/wenbihan/coverage)                 |
| NIST16            | 0/564                                                        | 5616x3744                                                    | 500x500                                                      | Including splice, remove, and copy-move tampering, with high-resolution images in the dataset. | [Link](https://mig.nist.gov/MFC/PubData/Resources.html)<br />[Cloud Disk Link](https://pan.baidu.com/share/init?surl=nZVUaqaBOD1u3xtxVDKmFw) |
| Defacto           |                                                              |                                                              |                                                              | Large dataset with tampering areas occupying a very small proportion of the total image area. | [Link](https://www.kaggle.com/defactodataset/datasets)       |
| IMD2020           | 414/2020                                                     | 260x193                                                      | 2958x4437                                                    | Tampering types include complex real-world editing (such as splicing, local modification), with variable resolutions. | [Link](https://staff.utia.cas.cz/novozada/db/IMD2020.zip)    |
| FantasticReality  | 16592/19423                                                  | 500x333                                                      | 6000x4000                                                    | A multi-task annotated dataset combining tampering localization and semantic segmentation, providing pixel-level tampering area masks, instance segmentation, and category labels. | [Link1](https://drive.google.com/drive/folders/1AIKseHoL0ebux7tyeBDWN5zkHzAgYBzJ?usp=sharing)<br />[Link2](https://github.com/mjkwon2021/CAT-Net/issues/51) |
| PhotoShop-battle  | 11142/91886                                                  | Height: 136<br />Width: 68                                   | Height: 20000<br />Width: 12024                              | A large-scale real-world dataset for creative tampering detection, with a wide range of image resolutions. | [Link1](https://github.com/dbisUnibas/PS-Battles)<br />[Link2](https://www.kaggle.com/datasets/timocasti/psbattles/data) |
| Carvalho（DSO-1） | 100/100                                                      | 2048x1536                                                    | 2048x1536                                                    | Forging through splicing by adding people and other auxiliary processing. | [Link](http://ic.unicamp.br/~rocha/pub/downloads/2014-tiago-carvalho-thesis/tifs-database.zip) |
| GRIP Dataset      | 80/80                                                        | 1024x768                                                     | 1024x768                                                     | Designed to assess the robustness of copy-move tampering detection algorithms under complex post-processing. The tampered area in the dataset is small. | [Link](https://www.grip.unina.it/download/prog/CMFD/)        |
| CoMoFoD           | 260/260                                                      | 512x512                                                      | 3000x3000                                                    | Designed specifically for the evaluation of copy-move detection algorithms, with various types of post-processing methods applied to both forged and original images. | [Link](https://www.vcl.fer.hr/comofod/download.html)         |
| CocoGlide         | 512/512                                                      | 256x256                                                      | 256x256                                                      | Targeting generative tampering research, combining GLIDE diffusion models with semantic prompts to generate tampered content, simulating semantic-level local tampering. | [Link](https://www.grip.unina.it/download/prog/TruFor/CocoGlide.zip) |
| tampCOCO          | 0/800000                                                     | 72x51                                                        | 640x640                                                      | Built based on the COCO 2017 dataset, including copy-move and splice, all images are JPEG compressed, retaining clear boundaries to support model learning of low-level tampering traces. | [Link](https://www.kaggle.com/datasets/qsii24/tampcoco)      |
| compRAISE         | 24462/0                                                      | 2278x1515                                                    | 6159x4079                                                    | Built based on a high-resolution image library and COCO instance annotations. Copy-move forgeries with irregular shapes, non-rectangular and asymmetric. | [Link](https://www.kaggle.com/datasets/qsii24/compraise)     |
| OpenForensics     | 0/2000                                                       | 512x512                                                      | 512x512                                                      | The first large-scale dataset for multi-face forgery detection and segmentation. Rich content, image scenes cover indoor and outdoor; diverse face situations, size variations. | [Link](https://zenodo.org/records/5528418)                   |

## 1 CASIA v1.0 and CASIA v2.0

### 1.1 Basic Information

- Introduction: Both CASIA datasets are provided by the **Institute of Automation, Chinese Academy of Sciences** and are primarily aimed at detecting Splicing operations. Notably, the official site does not provide 0-1 Masks as ground truth; only tampered and original images are available!
  - CASIA V1.0 includes 921 tampered images and their corresponding original images, with a fixed resolution of 384x256, and the tampering method is limited to Splicing.
  - CASIA V2.0 includes 5123 tampered images and their corresponding original images, with resolutions ranging from 320x240 to 800x600, and in addition to Splicing, blurring is also used.
- APA citation: Dong, J., Wang, W., & Tan, T. (2013). CASIA Image Tampering Detection Evaluation Database. 2013 IEEE China Summit and International Conference on Signal and Information Processing, 422–426. https://doi.org/10.1109/ChinaSIP.2013.6625374
- Paper link: [CASIA Image Tampering Detection Evaluation Database | IEEE Conference Publication | IEEE Xplore](https://ieeexplore.ieee.org/abstract/document/6625374)

### 1.2 Download and Usage

​	It should be noted that the original website of the Institute of Automation, Chinese Academy of Sciences, is currently under maintenance and cannot be used. There are some naming-related annotation errors in the dataset, and the original dataset does not provide 0-1 masks as GroundTruth.

​	To summarize, the widely used CASIA datasets are those provided by "Pham" and others in a 2019 paper titled "Hybrid Image-Retrieval Method for Image-Splicing Validation," which are open-sourced with 0-1 masks. The authors themselves used algorithms to subtract the original images from the tampered images to obtain datasets with 0-1 masks and corresponding images, and corrected file name-related errors. These datasets are also widely circulated on data science platforms such as Kaggle, and the GitHub links provided by Pham and others are as follows:

- CASIA v1.0: https://github.com/namtpham/casia1groundtruth
- CASIA v2.0: https://github.com/namtpham/casia2groundtruth

​	However, the CASIA v2.0 corrected by Pham and others still has some vulnerabilities. Based on this dataset, under the guidance of Professor Zhou Jizhe from DICALAB, student Ma Xiaochen made corrections to some of the errors and uploaded a new corrected dataset. Apart from changes to dozens of images, the rest of the information is identical to the dataset provided by Pham and others, which can be downloaded through the following GitHub link:

- Corrected CASIA v2.0: https://github.com/SunnyHaze/CASIA2.0-Corrected-Groundtruth

## 2 Columbia

### 2.1 Basic Information

- Introduction: The Columbia dataset is a tampering detection dataset produced by Columbia University's DVMM Lab, also primarily aimed at Splicing operations. It should be noted that there are two different versions of the dataset on the official website, one of which is a black and white fixed-size (128 x 128) dataset, and the other is a high-definition color splicing dataset, **the color one is generally used as the Benchmark.**
  - The black and white image block dataset contains more than 1800 128x128 image blocks.
  - The color Splicing dataset contains 180 image blocks with resolutions ranging from 757x568 to 1152x768.

- Paper link: Generally, tampering image-related papers only cite the dataset's official website, but in principle, the authors require citation of their paper: "Detecting Image Splicing Using Geometry Invariants And Camera Characteristics Consistency" ([hsu06ICMEcrf.pdf (columbia.edu)](https://www.ee.columbia.edu/ln/dvmm/publications/06/hsu06ICMEcrf.pdf)).

- APA citation: Hsu, Y., & Chang, S. (2006). Detecting Image Splicing using Geometry Invariants and Camera Characteristics Consistency. 2006 IEEE International Conference on Multimedia and Expo, 549–552. https://doi.org/10.1109/ICME.2006.262447

### 2.2 Download and Usage

- Columbia Image Splicing Detection Evaluation Dataset (Black and White Image Block Dataset): https://www.ee.columbia.edu/ln/dvmm/downloads/AuthSplicedDataSet/AuthSplicedDataSet.htm

- Columbia Uncompressed Image Splicing Detection Evaluation Dataset (Color Splicing Dataset): https://www.ee.columbia.edu/ln/dvmm/downloads/authsplcuncmp/

## 3 Coverage

### 3.1 Basic Information

- Introduction: Includes 100 pairs of tampered and corresponding genuine images, primarily applying Copy-move tampering operations. The images are highly deceptive, often involving copying one item from a group of similar items.

- Paper: COVERAGE – A NOVEL DATABASE FOR COPY-MOVE FORGERY DETECTION

- APA citation: Wen, B., Zhu, Y., Subramanian, R., Ng, T.-T., Shen, X., & Winkler, S. (2016). COVERAGE — A novel database for copy-move forgery detection. 2016 IEEE International Conference on Image Processing (ICIP), 161–165. https://doi.org/10.1109/ICIP.2016.7532339

### 3.2 Download and Usage

- The official GitHub link: https://github.com/wenbihan/coverage

## 4 NIST16

### 4.1 Basic Information

- Introduction: Includes splice, remove, and copy-move tampering, with less than 1k samples. Generally, NIST16 is used for tampering detection evaluation. The images in the dataset have high resolutions and usually require resizing or evaluation using a sliding window. It should be noted that there is a certain "leakage" in the dataset division, meaning that some test set images also exist in the training set, but everyone generally turns a blind eye to this issue.

- Paper link: https://ieeexplore.ieee.org/abstract/document/8638296/

- APA citation: Guan, H., Kozak, M., Robertson, E., Lee, Y., Yates, A. N., Delgado, A., Zhou, D., Kheyrkhah, T., Smith, J., & Fiscus, J. (2019). MFC Datasets: Large-Scale Benchmark Datasets for Media Forensic Challenge Evaluation. 2019 IEEE Winter Applications of Computer Vision Workshops (WACVW), 63–72. https://doi.org/10.1109/WACVW.2019.00018

### 4.2 Download and Usage

The official website for this dataset is a bit cumbersome to use, requiring registration on the NIST (National Institute of Standards and Technology) website to obtain permission before downloading.

- https://mfc.nist.gov/

- https://mfc.nist.gov/participant

In theory, after a series of operations, you can access this website https://mig.nist.gov/MFC/PubData/Resources.html to download the dataset mentioned above, which is commonly referred to as NIST16 (the updated version 20 generally requires a license):

NC2016 Nimble Science Seta: NC2016 Test0613.SCI.tgz
NC2016 Testset June 2013: NC2016 Test0613.tarbz2

**If it's not so troublesome, there is a big brother on Zhihu who has uploaded it to a cloud disk, a non-official way to obtain it as follows (not guaranteed for long-term use):**

Baidu Cloud Disk link: https://pan.baidu.com/share/init?surl=nZVUaqaBOD1u3xtxVDKmFw

Extraction code: lik7

## 5 Defacto

### 5.1 Basic Information

- Introduction: The Defacto dataset is a tampered image dataset generated based on MSCOCO, with a very large content size, but the "overall tampered area as a proportion of the total image area" is very small, meaning that on average, most tampering only involves very small objects in the image, which is quite different from other datasets.

- Paper link: https://ieeexplore.ieee.org/abstract/document/8903181/

- APA citation: Mahfoudi, G., Tajini, B., Retraint, F., Morain-Nicolier, F., Dugelay, J. L., & Pic, M. (2019). DEFACTO: Image and Face Manipulation Dataset. 2019 27th European Signal Processing Conference (EUSIPCO), 1–5. https://doi.org/10.23919/EUSIPCO.2019.8903181

Additionally, since it is generated from COCO, the authors also require citation of the MS COCO dataset:

- Paper link: https://arxiv.org/abs/1405.0312

- APA citation: Lin, T.-Y., Maire, M., Belongie, S., Bourdev, L., Girshick, R., Hays, J., Perona, P., Ramanan, D., Zitnick, C. L., & Dollár, P. (2015). Microsoft COCO: Common Objects in Context (arXiv:1405.0312). arXiv. http://arxiv.org/abs/1405.0312 

### 5.2 Download and Usage

The dataset has been fully uploaded to Kaggle, so it is relatively easy to download and use. Due to its large size, the authors have divided it into several parts for download:

Inpainting:

https://www.kaggle.com/datasets/defactodataset/defactoinpainting

Copy-move:

https://www.kaggle.com/datasets/defactodataset/defactocopymove

Splicing:

https://www.kaggle.com/datasets/defactodataset/defactosplicing

## 6 IMD2020

### 6.1 Basic Information

- Introduction: IMD2020 is a large-scale tampered image detection dataset constructed by the team from the Institute of Information Theory and Automation of the Czech Academy of Sciences, consisting of both synthetically generated and real tampered parts. The former is based on 35,000 real images taken by 2,322 different camera models, and a set of tampered images was synthesized using a large number of image manipulation techniques (including image processing technologies and GAN-based or repair methods); the latter consists of 2,000 "real-life" (uncontrolled) tampered images collected from the internet. Provides precise binary mask (0-1 Mask) annotations.
  - Synthetically generated dataset:
    - Includes 35,000 real images (from 2,322 camera models) and corresponding 35,000 tampered images, totaling 70,000 images.
    - Tampering methods include traditional processing (JPEG compression, blurring, noise, etc.), GAN generation (such as FaceApp), image repair (Inpainting), etc., with tampered area ratios ranging from 5% to 30%.
    - All tampered images provide binary mask annotations for the tampered areas.
  - Real-life dataset:
    - Includes 2,010 uncontrolled tampered images collected from the internet, each matched with the original image and manually annotated with binary masks.
    - Tampering types include complex real-world editing (such as splicing, local modification), with variable resolutions.

- Paper link: [https://ieeexplore.ieee.org/document/9096940/](https://ieeexplore.ieee.org/document/9096940)

- APA citation: Novozamsky, A., Mahdian, B., & Saic, S. (2020). IMD2020: A large-scale annotated dataset tailored for detecting manipulated images. In Proceedings of the IEEE/CVF winter conference on applications of computer vision workshops (pp. 71-80). [https://doi.org/10.1109/WACVW50321.2020.9096940](https://doi.org/10.1109/WACVW50321.2020.9096940)

### 6.2 Download and Usage

Official download website: https://staff.utia.cas.cz/novozada/db/

- Real-life tampered dataset: IMD2020 Real-Life Manipulated Images section

- Synthetic tampered dataset: IMD2020 Large-Scale Set of Inpainting Images section

## 7 FantasticReality

### 7.1 Basic Information

- Introduction: The FantasticReality dataset is a large-scale tampered detection dataset jointly constructed by the GosNIIAS (Russian National Research Institute for Aviation Systems), MIPT (Moscow Institute of Physics and Technology), and FBK (Bruno Kessler Foundation), aiming to address the issues of small scale and incomplete annotation in existing datasets. It also provides pixel-level tampered area masks (ground truth masks), instance segmentation, and category labels, covering 10 common object categories (such as people, vehicles, buildings, etc.), and is the first <font color=red>multi-task</font> annotated dataset combining tampered localization and semantic segmentation. It includes 16k real images and 16k tampered images, totaling 32k images, with the main tampering method being Splicing.
- Paper link: [The Point Where Reality Meets Fantasy: Mixed Adversarial Generators for Image Splice Detection](https://papers.nips.cc/paper_files/paper/2019/hash/98dce83da57b0395e163467c9dae521b-Abstract.html)
- APA citation: Kniaz, V. V., Knyaz, V., & Remondino, F. (2019). The point where reality meets fantasy: Mixed adversarial generators for image splice detection. *Advances in neural information processing systems*, *32*.

### 7.2 Download and Usage

- The original paper provides a dataset download link: [http://zefirus.org/MAG](http://zefirus.org/MAG) (defunct).

- The authors have provided a new download link in the CAT-Net GitHub repository: [https://drive.google.com/drive/folders/1AIKseHoL0ebux7tyeBDWN5zkHzAgYBzJ?usp=sharing](https://drive.google.com/drive/folders/1AIKseHoL0ebux7tyeBDWN5zkHzAgYBzJ?usp=sharing), but it seems there are still issues.

- Other options include contacting the authors for download permissions. For details, see [https://github.com/mjkwon2021/CAT-Net/issues/51](https://github.com/mjkwon2021/CAT-Net/issues/51), where the authors provide specific contact information and requirements.

## 8 PhotoShop-battle

### 8.1 Basic Information

- Introduction: The PS-Battles dataset, developed by the University of Basel, Switzerland, based on the Reddit community **r/photoshopbattles**, is the first large-scale real-world dataset for *creative tampering detection*. It focuses on diverse and high-semantic image tampering content generated by community users, including humorous composites, scene replacements, and character fusions (such as splicing, copy-move, and removal). The dataset contains 11,142 image sets (a total of 103,028 images) with a wide range of resolutions (width: 68-12,024 pixels, height: 136-20,000 pixels).
- Paper link: [The PS-Battles Dataset - an Image Collection for Image Manipulation Detection](https://arxiv.org/abs/1804.04866)
- APA citation: Heller, S., Rossetto, L., & Schuldt, H. (2018). The ps-battles dataset-an image collection for image manipulation detection. *arXiv preprint arXiv:1804.04866*. [https://arxiv.org/abs/1804.04866](https://arxiv.org/abs/1804.04866)

### 8.2 Download and Usage

- Official GitHub repository: [https://github.com/dbisUnibas/PS-Battles](https://github.com/dbisUnibas/PS-Battles)
- Kaggle: [https://www.kaggle.com/datasets/timocasti/psbattles/data](https://www.kaggle.com/datasets/timocasti/psbattles/data)

  - For Ubuntu and MacOS: See the GitHub repository for details. Run the provided `download.sh` script.
  - For Windows: See the Kaggle link. Place the provided `download.py` script and `Originals.tsv` & `photoshops.tsv` in the same directory, then run the `download.py` script.

## 9 Carvalho (DSO-1)

### 9.1 Basic Information

- Introduction: The DSO-1 dataset contains 200 high-resolution images (2048×1536 pixels), with 100 original unmodified images and 100 forged images. Forgery is achieved by *splicing to add people* (adding one or more people to a source image that already contains one or more people) with adjustments in color and brightness.
- Paper link: [Exposing Digital Image Forgeries by Illumination Color Classification](https://ieeexplore.ieee.org/document/6522874)
- APA citation: De Carvalho, T. J., Riess, C., Angelopoulou, E., Pedrini, H., & de Rezende Rocha, A. (2013). Exposing digital image forgeries by illumination color classification. *IEEE Transactions on Information Forensics and Security*, *8*(7), 1182-1194. [https://doi.org/10.1109/TIFS.2013.2265677](https://doi.org/10.1109/TIFS.2013.2265677)

### 9.2 Download and Usage

- Download link: [http://ic.unicamp.br/~rocha/pub/downloads/2014-tiago-carvalho-thesis/tifs-database.zip](http://ic.unicamp.br/%7Erocha/pub/downloads/2014-tiago-carvalho-thesis/tifs-database.zip) (appears to be problematic and may not succeed)
- Related datasets collection: [https://recodbr.wordpress.com/code-n-data/#porno](https://recodbr.wordpress.com/code-n-data/#porno)

## 10 GRIP Dataset

### 10.1 Basic Information

- Introduction: The dataset is designed to assess the robustness of *copy-move forgery detection* algorithms under complex post-processing interference. It focuses on detecting small-scale tampering regions, containing 80 images and addressing issues of large tampering regions and limited post-processing types in previous datasets. It includes high-resolution images (e.g., 768×1024 pixels) with tampering regions ranging from 4000 pixels (<1%) to 50,000 pixels, categorized into three background complexities: smooth, mixed, and textured. Pixel-level ground-truth masks (0-1 masks) are provided.
- Paper link: [Efficient Dense-Field Copy–Move Forgery Detection](https://ieeexplore.ieee.org/document/7154457)
- APA citation: Cozzolino, D., Poggi, G., & Verdoliva, L. (2015). Efficient dense-field copy–move forgery detection. *IEEE Transactions on Information Forensics and Security*, *10*(11), 2284-2297. [https://doi.org/10.1109/TIFS.2015.2455334](https://doi.org/10.1109/TIFS.2015.2455334)

### 10.2 Download and Usage

- The dataset is provided by the GRIP group at [University Federico II of Naples](http://www.unina.it/): [https://www.grip.unina.it/download/prog/CMFD/](https://www.grip.unina.it/download/prog/CMFD/)

## 11 CoMoFoD

### 11.1 Basic Information

- Introduction: CoMoFoD (Copy-Move Forgery Detection) database, developed by the University of Zagreb, Croatia, is a comprehensive benchmark dataset for *copy-move forgery detection* algorithm evaluation. It systematically integrates various geometric transformations and post-processing operations, providing pixel-level tampering mask annotations to address the limitations of existing datasets in terms of post-processing types and scale. It contains 260 sets of forged images divided into two categories: 200 sets in the small image category (512x512) and 60 sets in the large image category (3000x2000). All forged and original images have undergone different types of post-processing methods, such as JPEG compression, blurring, adding noise, and color reduction.
- Paper link: [CoMoFoD — New database for copy-move forgery detection](https://ieeexplore.ieee.org/document/6658316)
- APA citation: Tralic, D., Zupancic, I., Grgic, S., & Grgic, M. (2013, September). CoMoFoD—New database for copy-move forgery detection. In *Proceedings ELMAR-2013* (pp. 49-54). IEEE.

### 11.2 Download and Usage

- Official dataset website: [https://www.vcl.fer.hr/comofod/download.html](https://www.vcl.fer.hr/comofod/download.html)
  - Small image category database (512x512): 200 sets, download link: [https://www.vcl.fer.hr/comofod/comofod_small.rar](https://www.vcl.fer.hr/comofod/comofod_small.rar)
  - Large image category database (3000x2000): 60 sets, no download link available; contact the authors for access.

## 12 CocoGlide

### 12.1 Basic Information

- Introduction: CocoGlide is a tampering detection dataset built on the COCO 2017 validation set, designed to evaluate the local tampering detection capabilities of modern generative models (e.g., diffusion models). It generates realistic tampering content by combining the GLIDE diffusion model with semantic prompts to replace corresponding regions in the original images (e.g., animals, vehicles), simulating semantic-level local tampering in real-world scenarios. It contains **512 tampered images**, all generated from 256×256 pixel crops of the COCO validation set, filling the gap of *generative tampering* samples in traditional datasets.
- Paper link: [TruFor: Leveraging all-round clues for trustworthy image forgery detection and localization](https://arxiv.org/abs/2212.10957)
- APA citation: Guillaro, F., Cozzolino, D., Sud, A., Dufour, N., & Verdoliva, L. (2023). Trufor: Leveraging all-round clues for trustworthy image forgery detection and localization. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition* (pp. 20606-20615).

### 12.2 Download and Usage

- The dataset is provided by the GRIP group at [University Federico II of Naples](http://www.unina.it/): [https://www.grip.unina.it/download/prog/TruFor/CocoGlide.zip](https://www.grip.unina.it/download/prog/TruFor/CocoGlide.zip)

## 13 tampCOCO

### 13.1 Basic Information

- Introduction: tampCOCO is a tampering detection dataset built on the COCO 2017 dataset, consisting of two parts: SP COCO (cross-image splicing) and CM COCO (intra-image copy-move). Pixel-level binary masks (0-1 masks) are provided.
  - **SP COCO**:
    - Based on COCO images, objects (e.g., people, vehicles) are randomly selected from one image, rotated/scaled, and pasted into a random position in another image.
    - A total of 200,000 forged images, all subjected to JPEG compression (quality factor 60-100) without additional blurring or other post-processing.
  - **CM COCO**:
    - In a single COCO image, selected regions (e.g., objects or backgrounds) are copied and pasted to other locations to create copy-move tampering samples.
    - A total of 600,000 images, with JPEG compression parameters consistent with SP COCO, retaining clear boundaries to support model learning of low-level tampering traces.

- Paper link: [Learning JPEG Compression Artifacts for Image Manipulation Detection and Localization](https://arxiv.org/abs/2108.12947)
- APA citation: Kwon, M. J., Nam, S. H., Yu, I. J., Lee, H. K., & Kim, C. (2022). Learning jpeg compression artifacts for image manipulation detection and localization. *International Journal of Computer Vision*, *130*(8), 1875-1895. [https://arxiv.org/abs/2108.12947](https://arxiv.org/abs/2108.12947)

### 13.2 Download and Usage

- The dataset is fully uploaded to Kaggle: [https://www.kaggle.com/datasets/qsii24/tampcoco](https://www.kaggle.com/datasets/qsii24/tampcoco)
- Due to its large size, the dataset is divided into 13 parts for download. The above link directs to the index for all downloads.

## 14 compRAISE

### 14.1 Basic Information

- Introduction: compRAISE (CM RAISE) is a complex tampering detection dataset built on the **RAISE high-resolution image library** and **COCO instance annotations**. High-resolution natural scene images (resolution range: 2,000×3,008 to 4,928×3,264) are selected from the RAISE dataset (containing 8,156 uncompressed RAW images). Irregularly shaped regions are extracted using random polygon instance masks from COCO 2017 (approximately 1.2 million annotations), ensuring non-rectangular and asymmetric tampering boundaries. A *copy-move strategy* is then applied within individual RAISE images to forge tampering.
- Paper link: [Learning JPEG Compression Artifacts for Image Manipulation Detection and Localization](https://arxiv.org/abs/2108.12947)
- APA citation: Kwon, M. J., Nam, S. H., Yu, I. J., Lee, H. K., & Kim, C. (2022). Learning jpeg compression artifacts for image manipulation detection and localization. *International Journal of Computer Vision*, *130*(8), 1875-1895. [https://arxiv.org/abs/2108.12947](https://arxiv.org/abs/2108.12947)

### 14.2 Download and Usage

- The dataset is fully uploaded to Kaggle: [https://www.kaggle.com/datasets/qsii24/compraise](https://www.kaggle.com/datasets/qsii24/compraise)
- Due to its large size, the dataset is divided into 15 parts for download. The above link directs to the index for all downloads.

## 15 OpenForensics

### 15.1 Basic Information

- Introduction: OpenForensics is the first large-scale dataset for *multi-face forgery detection and segmentation* in complex natural scenes, jointly constructed by the National Institute of Informatics, Japan, the Graduate University for Advanced Studies, and the University of Tokyo. It provides pixel-level fine annotations to support multi-dimensional tasks such as forgery detection, instance segmentation, and forgery boundary identification.
  - Contains 115,325 images with a total of 334,136 faces (an average of 2.9 faces per image), including 160,670 real faces and 173,660 fake faces.
  - Divided into training set (44K+ images), validation set (7K+ images), test development set (18K+ images), and test challenge set (45K+ images).
  - Image scenes cover both indoor (63.7%) and outdoor (36.3%) environments, with highly diverse face poses, ages, genders, and occlusion conditions, including faces of varying sizes from small to large.
  - Fake faces have a resolution of 512×512.
- Paper link: [OpenForensics: Large-Scale Challenging Dataset For Multi-Face Forgery Detection And Segmentation In-The-Wild](https://arxiv.org/abs/2107.14480)
- APA citation: Le, T. N., Nguyen, H. H., Yamagishi, J., & Echizen, I. (2021). Openforensics: Large-scale challenging dataset for multi-face forgery detection and segmentation in-the-wild. In *Proceedings of the IEEE/CVF international conference on computer vision* (pp. 10117-10127).

### 15.2 Download and Usage

- Official download link: [https://zenodo.org/records/5528418](https://zenodo.org/records/5528418)
- The dataset is divided into multiple parts for download. The above link directs to the index for all downloads.
