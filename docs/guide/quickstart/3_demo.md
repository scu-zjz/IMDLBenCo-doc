# Case Three: Implementing Your Own Model with `benco init`
We believe that the fastest way to learn is "Learn by Doing" (learning by doing), so we use several cases to help users get started quickly.

Overall, IMDL-BenCo helps you quickly complete the development of image tampering detection scientific research projects through command line calls similar to `git` and `conda`. If you have learned front-end technologies such as vue, understanding the design pattern of IMDLBenCo according to vue-cli will be very easy.

Regardless, please refer to [Installation](./install.md) to complete the installation of IMDL-BenCo first.

:::tip Motivation of This Chapter
To ensure that you can flexibly create your own models, this chapter will help you understand the design patterns and interface paradigms of IMDL-BenCo. We will introduce each part step by step according to the usage process.
:::

## Introducing All Design Patterns

### Generating Default Scripts

Under a clean working path, executing the following command line instruction will generate all the scripts needed for **creating your own model and testing**. As a default instruction, omitting the base will also execute the same command.

::: tabs
@tab Full Command
```shell
benco init base
```
@tab Abbreviated Command
```shell
benco init
```
:::

After normal execution, you will see the following files generated under the current path, and their uses are as commented:
```bash
.
├── balanced_dataset.json       # Stores the dataset path organized according to Protocol-CAT
├── mymodel.py                  # The core model implementation
├── README-IMDLBenCo.md         # A simple readme
├── test_datasets.json          # Stores the test dataset path
├── test_mymodel.sh             # The shell script for running tests with parameters
├── test.py                     # The actual Python code for the test script
├── test_robust_mymodel.sh      # The shell script for running robustness tests with parameters
├── test_robust.py              # The actual Python code for robustness tests
├── train_mymodel.sh            # The shell script for running training with parameters
└── train.py                    # The actual Python code for the training script
```

::: warning Special Attention
If the scripts have been generated and some modifications have been made, please be very careful when calling `benco init` for the second time. IMDLBenCo will cover the files one by one after asking, and if you operate incorrectly, it may cause you to lose your existing modifications. Be careful. It is recommended to use git version control to avoid losing existing code due to this operation.
:::


### Model File Design Pattern
IMDLBenCo needs to organize the model files in a certain format to ensure that the entry can align with the `DataLoader` and the exit can align with the subsequent `Evaluator` and `Visualize tools`.

After executing `benco init`, a model with the simplest **single-layer convolution** is generated in `mymodel.py` by default. You can quickly view its content through the [Github link to mymodel.py](https://github.com/scu-zjz/IMDLBenCo/blob/main/IMDLBenCo/statics/base/mymodel.py).

```python
from IMDLBenCo.registry import MODELS
import torch.nn as nn
import torch

@MODELS.register_module()
class MyModel(nn.Module):
    def __init__(self, MyModel_Customized_param:int, pre_trained_weights:str) -> None:
        """
        The parameters of the `__init__` function will be automatically converted into the parameters expected by the argparser in the training and testing scripts by the framework according to their annotated types and variable names. 
        
        In other words, you can directly pass in parameters with the same names and types from the `run.sh` script to initialize the model.
        """
        super().__init__()
        
        # Useless, just an example
        self.MyModel_Customized_param = MyModel_Customized_param
        self.pre_trained_weights = pre_trained_weights

        # A single layer conv2d 
        self.demo_layer = nn.Conv2d(
            in_channels=3,
            out_channels=1,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        # A simple loss
        self.loss_func_a = nn.BCEWithLogitsLoss()
        
    def forward(self, image, mask, label, *args, **kwargs):
        # simple forwarding
        pred_mask = self.demo_layer(image)
        
        # simple loss
        loss_a = self.loss_func_a(pred_mask, mask)
        loss_b = torch.abs(torch.mean(pred_mask - mask))
        combined_loss = loss_a + loss_b
        
        
        pred_label = torch.mean(pred_mask)
        
        inverse_mask = 1 - mask
        
        # ----------Output interface--------------------------------------
        output_dict = {
            # loss for backward
            "backward_loss": combined_loss,
            # predicted mask, will calculate for metrics automatically
            "pred_mask": pred_mask,
            # predicted binaray label, will calculate for metrics automatically
            "pred_label": pred_label,

            # ----values below is for visualization----
            # automatically visualize with the key-value pairs
            "visual_loss": {
                # keys here can be customized by yourself.
                "predict_loss": combined_loss,
                'loss_a' : loss_a,
                "I am loss_b :)": loss_b, 
            },

            "visual_image": {
                # keys here can be customized by yourself.
                # Various intermediate masks, heatmaps, and feature maps can be appropriately converted into RGB or single-channel images for visualization here.
                "pred_mask": pred_mask,
                "reverse_mask" : inverse_mask, 
            }
            # -------------------------------------------------------------
        }
        
        return output_dict
    
    
if __name__ == "__main__":
    print(MODELS)
```


**Introducing the design requirements of IMDLBenCo's model file from top to bottom:**
- Line 5: `@MODELS.register_module()`
  - Registers the model to IMDLBenCo's global registry based on the registration mechanism, making it easy for other scripts to quickly call the class through strings.
  - If you are not familiar with the registration mechanism, a one-sentence explanation is: **It automatically maintains a dictionary mapping from strings to corresponding class mappings**, making it easy to "freely" pass parameters.
  - In actual use, by passing the registered "class name identical string" to the `--model` function of the shell script that starts training, the framework can load the corresponding custom or built-in model. For details, please refer to [this link](https://github.com/scu-zjz/IMDLBenCo/blob/4c6a2937c3cae8d6ff26bf85e9bad0c5ec467468/IMDLBenCo/statics/model_zoo/runs/demo_train_mvss.sh#L10).
- Lines 29 and 37: **Loss functions must be defined in the `__init__()` or `forward()` functions**
- Line 31: Defining the forward function `def forward(self, image, mask, label, *args, **kwargs):`
  - It is necessary to include `*args, **kwargs` required for Python function unpacking to receive unused parameters.
    - If you are not familiar with it, please refer to [Python Official Documentation-4.8.2. Keyword Arguments](https://docs.python.org/3/tutorial/controlflow.html#keyword-arguments), [Python Official Documentation Chinese Version-4.8.2 Keyword Arguments](https://docs.python.org/zh-cn/3/tutorial/controlflow.html#keyword-arguments)
  - The parameter variable names must be exactly the same as the field names contained in the dictionary `data_dict` returned by [`abstract_dataset.py`](https://github.com/scu-zjz/IMDLBenCo/blob/main/IMDLBenCo/datasets/abstract_dataset.py). The default fields are shown in the table below:
    - |Key Name|Meaning|Type|
      |:-:|-|:-:|
      |image|Input original image|Tensor(B,3,H,W)|
      |mask|Mask of the prediction target|Tensor(B,1,H,W)|
      |edge_mask|After [erosion (erosion) and dilation (dilation)](https://docs.opencv.org/3.4/db/df6/tutorial_erosion_dilatation.html) based on the mask, only the boundary is white, which is used for various models that need boundary loss functions. To reduce computational overhead, the corresponding dataloader will only return this key-value pair for the model's `forward()` function when the `--edge_mask_width 7` parameter is passed in the training `shell`, refer to the [shell](https://github.com/scu-zjz/IMDLBenCo/blob/4c6a2937c3cae8d6ff26bf85e9bad0c5ec467468/IMDLBenCo/statics/model_zoo/runs/demo_train_iml_vit.sh#L22) and [model forward function](https://github.com/scu-zjz/IMDLBenCo/blob/4c6a2937c3cae8d6ff26bf85e9bad0c5ec467468/IMDLBenCo/model_zoo/iml_vit/iml_vit.py#L125) of `IML-ViT`.<br>If you do not need the boundary mask to calculate subsequent losses, you do not need to pass it in the shell, nor do you need to prepare a parameter named `edge_mask` in the model's `forward()` function, refer to the [shell](https://github.com/scu-zjz/IMDLBenCo/blob/main/IMDLBenCo/statics/model_zoo/runs/demo_train_object_former.sh) and [model forward function](https://github.com/scu-zjz/IMDLBenCo/blob/4c6a2937c3cae8d6ff26bf85e9bad0c5ec467468/IMDLBenCo/model_zoo/object_former/object_former.py#L285) of `ObjectFormer`.|Tensor(B,1,H,W)|
      |label|Image-level prediction of zero-one labels|Tensor(B,1)|
      |shape|The shape of the image after padding or resizing|Tensor(B,2), two dimensions each with one value, representing H and W respectively|
      |original_shape|The shape of the image when it was first read|Tensor(B,2), two dimensions each with one value, representing H and W respectively|
      |name|The path and filename of the image|str|
      |shape_mask|In the case of padding, only the pixels inside this mask that are 1 are calculated as the final metric, 1 defaults to a square area the same size as the original image|Tensor(B,1,H,W)|
    - For different tasks, you can take these fields as needed and input them into the model for use.
    - In addition, for the Jpeg-related image materials needed by CAT-Net, we have designed the post-processing function `post_func` to generate more content based on the existing fields. At this time, it is also necessary to ensure that the corresponding forward function's fields are aligned. **Custom models with similar needs can also use this pattern to introduce other modal information in the dataloader.** The following is a case of CAT-Net:
      - [Github link to `cat_net_post_function`](https://github.com/scu-zjz/IMDLBenCo/blob/c2d6dc03eab3f33461690d5026b43afdac22f70c/IMDLBenCo/model_zoo/cat_net/cat_net_post_function.py#L7-L10), you can see that it includes two additional fields `DCT_coef` and `q_tables` for the model to input additional modalities
      - [Github link to `cat_net_model`](https://github.com/scu-zjz/IMDLBenCo/blob/c2d6dc03eab3f33461690d5026b43afdac22f70c/IMDLBenCo/model_zoo/cat_net/cat_net.py#L30), the `forward` function's parameter list needs to have corresponding fields to receive the above additional input information.
- Lines 36 to 38: All loss functions must be calculated in the `forward` function
- Lines 45 to 70: The dictionary of output results. <span style="color: red;font-weight: bold;">Very important!</span>, the function of each field in the dictionary is introduced as follows:
  - |Key|Meaning|Type|
    |:-:|:-:|:-:|
    |backward_loss|The loss function directly used for backpropagation|Tensor(1)|
    |pred_mask|The predicted mask, which will be directly used for subsequent metric calculations|Tensor(B,1,H,W)|
    |pred_label|The predicted zero-one label, which will be directly used for subsequent metric calculations|Tensor(B,1)|
    |visual_loss|Pass in the scalars that need to be visualized. You can name any number, any name of keys and pass in the corresponding scalars, which will be automatically visualized according to the Key name later|Dict()|
    |visual_image|Pass in the images, feature maps, various masks that need to be visualized. You can name any number, any name of keys and pass in the corresponding Tensors, which will be automatically visualized according to the Key name later|Dict()|
  - Be sure to organize according to this format to integrate normally into subsequent metric calculations, visualization, etc.

# A Comprehensive Tutorial from Start to Finish