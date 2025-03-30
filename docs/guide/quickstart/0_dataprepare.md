# Dataset Preparation

## Important
The dataset-related functionality and interfaces will be managed uniformly by the BenCo CLI in future versions.

Currently, you need to manually manage the corresponding `json` or dataset paths in each working directory to complete the setup.

## Dataset Formats and Characteristics

- IMDL-BenCo internally supports three different dataset formats, including two fundamental formats, `JsonDataset` and `ManiDataset`, which are used for reading individual datasets, and one special format, `BalanceDataset`, which manages multiple datasets simultaneously using a specific sampling strategy. Any dataset can be structured in one of three formats for subsequent model training.
  - `ManiDataset` follows the same organization as the CASIA dataset, making it suitable for lightweight development scenarios where **real images are not required**.
  - `JsonDataset` organizes the dataset using a JSON file, making it particularly suitable for scenarios that require real images.
  - `BalancedDataset` manages a dictionary containing multiple `ManiDataset` or `JsonDataset` objects. In each epoch, it **randomly samples n images** (default: 1800) from all the included sub-datasets. As a result, the actual number of images used for training in one epoch is calculated as `number of datasets × n`. However, when the dataset is large enough, the diversity of images across multiple epochs remains high. Additionally, this approach helps prevent the trained model from **overfitting to larger datasets**. `BalancedDataset` is primarily designed for protocols related to [CAT-Net](https://openaccess.thecvf.com/content/WACV2021/html/Kwon_CAT-Net_Compression_Artifact_Tracing_Network_for_Detection_and_Localization_of_WACV_2021_paper.html) and [TruFor](https://openaccess.thecvf.com/content/CVPR2023/html/Guillaro_TruFor_Leveraging_All-Round_Clues_for_Trustworthy_Image_Forgery_Detection_and_CVPR_2023_paper.html). If you are not reproducing these protocols, you do not need to focus on this format.
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
             "/mnt/data0/public_datasets/IML/compRAISE/compRAISE_1024_list.json"
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


Except for the protocol, pre-processing is also needed to speed up the training.

## Preprocessing High-Resolution Images
Some datasets come with very high resolutions by default. For example, the NIST16 dataset and the compRAISE dataset in CAT-Protocol includes images with resolutions as high as 4000x4000. Directly reading these datasets during training can lead to an extremely high I/O load, especially when used as training data. 

Therefore, we highly recommend resizing these images to a smaller size in advance, such as reducing the longer edge to 1024 while maintaining the aspect ratio. Otherwise, the training speed may be significantly slowed down. Please refer to [IMDL-BenCo issue #40](https://github.com/scu-zjz/IMDLBenCo/issues/40).

We provide a multithreaded image resizing code here, which can efficiently convert all images in a directory to the desired resolution using a thread pool:
```python
import os
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

def process_image(filename, directory, output_directory, target_size):
    try:
        with Image.open(os.path.join(directory, filename)) as img:
            width, height = img.size
            print(f'Processing Image: {filename} | Resolution: {width}x{height}')

            # Determine the scaling ratio for the longest side to be 1024
            if max(width, height) > target_size:
                if width > height:
                    new_width = target_size
                    new_height = int((target_size / width) * height)
                else:
                    new_height = target_size
                    new_width = int((target_size / height) * width)

                # Resize the image
                img_resized = img.resize((new_width, new_height), Image.ANTIALIAS)

                # Save the image to the specified folder
                output_path = os.path.join(output_directory, filename)
                img_resized.save(output_path)
                print(f'Resized and saved {filename} to {output_directory} with resolution {new_width}x{new_height}')
            else:
                # If no resizing is needed, directly copy the image to the target folder
                img.save(os.path.join(output_directory, filename))
                print(f'Image {filename} already meets the target size and was saved without resizing.')
            return 1  # Return a success count
    except Exception as e:
        print(f"Cannot process {filename}: {e}")
        return 0  # Return a failure count

def get_image_resolutions_and_resize(directory='.', output_directory='resized_images', target_size=1024):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Get all image files
    image_files = [f for f in os.listdir(directory) if f.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'gif', 'tiff'))]
    
    # Process images using a thread pool
    total_processed = 0
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_image, filename, directory, output_directory, target_size) for filename in image_files]
        
        # Wait for all threads to complete and accumulate the number of processed images
        for future in futures:
            total_processed += future.result()

    # Output the total number of images processed
    print(f"\nTotal number of images processed: {total_processed}")

# Execute the function
get_image_resolutions_and_resize(
    directory="./compRAISE",
    output_directory="./compRAISE1024",
    target_size=1024
)
```

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