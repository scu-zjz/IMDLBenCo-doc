# Dataset Preparation

## Important
The dataset-related functionality and interfaces will be managed uniformly by the BenCo CLI in future versions.

Currently, you need to manually manage the corresponding `json` or dataset paths in each working directory to complete the setup.

## Dataset Formats and Characteristics
- IMDL-BenCo implements three different dataset formats, including two basic types: `JsonDataset` and `ManiDataset`, as well as one `BalancedDataset`. Organizing your dataset in any of these formats will allow subsequent models to read it.
  - `ManiDataset`: This follows the organizational structure of the CASIA dataset, suitable for lightweight development scenarios where **no authentic images** are required.
  - `JsonDataset`: This organizes the dataset through a JSON file, especially suitable for scenarios where authentic images are needed.
  - `BalancedDataset`: Mainly designed for the [CAT-Net](https://openaccess.thecvf.com/content/WACV2021/html/Kwon_CAT-Net_Compression_Artifact_Tracing_Network_for_Detection_and_Localization_of_WACV_2021_paper.html) and [TruFor](https://openaccess.thecvf.com/content/CVPR2023/html/Guillaro_TruFor_Leveraging_All-Round_Clues_for_Trustworthy_Image_Forgery_Detection_and_CVPR_2023_paper.html) protocols, and can be ignored if not reproducing these protocols.


Additionally, during testing, it is necessary to input a large amount of datasets simultaneously, so an additional JSON format was defined for inputting large datasets. A sample is provided at the end of this section.

## Specific Format Definitions

1. `JsonDataset`: Provide the path to a JSON file that organizes the images and corresponding masks in the following format:
   ```json
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
   Here, "Negative" represents a completely black mask, indicating a fully authentic image, so no path needs to be provided.

2. `ManiDataset`: Provide the path to a folder containing two subfolders, `Tp` and `Gt`. BenCo will automatically read the images from `Tp` and the corresponding masks from `Gt`, pairing them based on the alphabetical order of file names as returned by `os.listdir()`. Typically, the default CASIA dataset is organized in this format. You can refer to the [sample folder in IML-ViT](https://github.com/SunnyHaze/IML-ViT/tree/main/images/sample_iml_dataset).

3. `BalancedDataset`: Provide the path to a JSON file specifically used to organize the datasets for the [CAT-Net](https://openaccess.thecvf.com/content/WACV2021/html/Kwon_CAT-Net_Compression_Artifact_Tracing_Network_for_Detection_and_Localization_of_WACV_2021_paper.html) and [TruFor](https://openaccess.thecvf.com/content/CVPR2023/html/Guillaro_TruFor_Leveraging_All-Round_Clues_for_Trustworthy_Image_Forgery_Detection_and_CVPR_2023_paper.html) protocols.
   1. Protocol definition: Protocol-CAT uses nine large datasets for training, but in each epoch, only 1,800 images are randomly sampled from each dataset to form a dataset of 16,200 images for training.
   2. JSON format:
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
      This two-dimensional array represents a list of datasets, where each row corresponds to a dataset. The first column specifies the dataset class type as a string, and the second column provides the path to that dataset.

After organizing the required datasets as needed, you can proceed to reproduce models or implement your own models.

## Test Dataset JSON
Specifically, for testing purposes, since a large number of datasets need to be input simultaneously to complete the test, a dedicated `test_dataset.json` has been defined to fulfill this function.

The keys represent the field names used for functionalities like Tensorboard, logging, and other visualization purposes, while the values correspond to the actual paths of the aforementioned datasets.

Example:

```JSON
{
    "Columbia": "/mnt/data0/public_datasets/IML/Columbia.json",
    "NIST16_1024": "/mnt/data0/public_datasets/IML/NIST16_1024",
    "NIST16_cleaned": "/mnt/data0/public_datasets/IML/NIST16_1024_cleaning",
    "coverage": "/mnt/data0/public_datasets/IML/coverage.json",
    "CASIAv1": "/mnt/data0/public_datasets/IML/CASIA1.0",
    "IMD20_1024": "/mnt/data0/public_datasets/IML/IMD_20_1024"
}
```