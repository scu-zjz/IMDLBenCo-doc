Here's the translated content:  

---

# Getting Started with Examples  

We believe that the fastest way to learn is through "Learn by Doing," so we provide a few examples to help users quickly get started.  

Overall, IMDL-BenCo facilitates the development of image tampering detection research projects through a command-line interface similar to `git` or `conda`. If you're familiar with frontend technologies like Vue, you'll find it easy to understand IMDL-BenCo's design paradigm by comparing it to `vue-cli`.  

Before anything else, please refer to the [Installation Guide](./install.md) to complete the setup of IMDL-BenCo.  

## Example 0: Quickly Understanding the Design Paradigm  

### Generating Default Scripts  

In a clean working directory, run the following command to generate all the necessary scripts for minimal execution:  

```bash  
benco init base  
```  

Since this is the default command, you can also use the shorter version:  

```bash  
benco init  
```  

Once successfully executed, the following files will be generated in the current directory, with their purposes explained in the comments:  

```bash  
.  
├── balanced_dataset.json       # Stores dataset paths organized according to Protocol-CAT  
├── mymodel.py                  # Core model implementation  
├── README-IMDLBenCo.md         # A simple readme file  
├── test_datasets.json          # Stores paths of test datasets  
├── test_mymodel.sh             # Shell script for running tests with parameters  
├── test.py                     # Actual Python script for testing  
├── test_robust_mymodel.sh      # Shell script for running robustness tests with parameters  
├── test_robust.py              # Actual Python script for robustness testing  
├── train_mymodel.sh            # Shell script for training with parameters  
└── train.py                    # Actual Python script for training  
```  

**Important Note:** If you have already generated scripts and made modifications, be cautious when running `benco init` again. IMDL-BenCo will prompt you before overwriting files, but accidental execution may result in the loss of your changes. It is highly recommended to use Git version control to avoid losing existing code.  

### Model File Design Paradigm  

IMDL-BenCo requires model files to be structured in a specific way to ensure compatibility with `DataLoader` as input and alignment with subsequent `Evaluator` and `Visualize tools` as output.  

After executing `benco init`, the default model—implemented with a simple **single-layer convolution**—is generated in `mymodel.py`. You can quickly review its content via the [mymodel.py link on GitHub](https://github.com/scu-zjz/IMDLBenCo/blob/main/IMDLBenCo/statics/base/mymodel.py).  

```python  
from IMDLBenCo.registry import MODELS  
import torch.nn as nn  
import torch  

@MODELS.register_module()  
class MyModel(nn.Module):  
    def __init__(self, MyModel_Customized_param:int, pre_trained_weights:str) -> None:  
        """  
        The parameters of the `__init__` function will be automatically converted into arguments for the training and testing scripts  
        by the framework based on their annotated types and variable names.  

        In other words, you can directly pass in parameters with the same names and types from the `run.sh` script to initialize the model.  
        """  
        super().__init__()  

        # Useless, just an example  
        self.MyModel_Customized_param = MyModel_Customized_param  
        self.pre_trained_weights = pre_trained_weights  

        # A single layer Conv2D  
        self.demo_layer = nn.Conv2d(  
            in_channels=3,  
            out_channels=1,  
            kernel_size=3,  
            stride=1,  
            padding=1,  
        )  

        # A simple loss function  
        self.loss_func_a = nn.BCEWithLogitsLoss()  

    def forward(self, image, mask, label, *args, **kwargs):  
        # Simple forward pass  
        pred_mask = self.demo_layer(image)  

        # Simple loss computation  
        loss_a = self.loss_func_a(pred_mask, mask)  
        loss_b = torch.abs(torch.mean(pred_mask - mask))  
        combined_loss = loss_a + loss_b  

        pred_label = torch.mean(pred_mask)  
        inverse_mask = 1 - mask  

        # ---------- Output Interface ----------  
        output_dict = {  
            "backward_loss": combined_loss,  # Loss for backward propagation  
            "pred_mask": pred_mask,          # Predicted mask for metric calculation  
            "pred_label": pred_label,        # Predicted binary label for metric calculation  

            # ---- Values for visualization ----  
            "visual_loss": {  
                "predict_loss": combined_loss,  
                'loss_a' : loss_a,  
                "I am loss_b :)": loss_b,  
            },  

            "visual_image": {  
                "pred_mask": pred_mask,  
                "reverse_mask" : inverse_mask,  
            }  
        }  
        return output_dict  

if __name__ == "__main__":  
    print(MODELS)  
```  

### Key Design Principles of IMDLBenCo Model Files  

To ensure proper execution within IMDLBenCo, model files must adhere to the following principles:  

- **Line 5: `@MODELS.register_module()`**  
  - This decorator registers the model into IMDLBenCo's global registry, allowing other scripts to access it by name.  
  - If you're unfamiliar with the registration mechanism, think of it as **automatically maintaining a dictionary that maps strings to corresponding classes**, enabling flexible parameter passing.  

- **Lines 29 & 37: Loss functions must be defined in `__init__()` or `forward()`**  

- **Line 31: Defining the `forward` function**  
  - `def forward(self, image, mask, label, *args, **kwargs):`  
  - Must include `*args, **kwargs` to handle unused parameters.  
  - Parameter names must match the dictionary keys returned by [`abstract_dataset.py`](https://github.com/scu-zjz/IMDLBenCo/blob/main/IMDLBenCo/datasets/abstract_dataset.py), which include:  

    | Key   | Meaning                                      | Type                 |  
    |-------|----------------------------------------------|----------------------|  
    | image | Input raw image                             | Tensor(B,3,H,W)      |  
    | mask  | Target mask for prediction                 | Tensor(B,1,H,W)      |  
    | label | Binary label for image-level prediction    | Tensor(B,1)          |  
    | shape | Shape of the input image                   | Tensor(B,1,1)        |  
    | name  | File path and name of the image            | str                  |  
    | shape_mask | Mask for calculating metrics during padding | Tensor(B,1,H,W) |  

  - Different tasks can selectively use these fields.  
  - For models like CAT-Net that require JPEG-related features, the `post_func` is used to generate additional input fields.  
  - Example from CAT-Net:  
    - [`cat_net_post_function`](https://github.com/scu-zjz/IMDLBenCo/blob/c2d6dc03eab3f33461690d5026b43afdac22f70c/IMDLBenCo/model_zoo/cat_net/cat_net_post_function.py#L7-L10) introduces `DCT_coef` and `q_tables`.  
    - [`cat_net_model`](https://github.com/scu-zjz/IMDLBenCo/blob/c2d6dc03eab3f33461690d5026b43afdac22f70c/IMDLBenCo/model_zoo/cat_net/cat_net.py#L30) ensures `forward` includes these fields.  

- **Lines 36-38: All loss calculations must occur within `forward`**  

- **Lines 45-70: Output dictionary structure**  

    | Key           | Meaning                                  | Type            |  
    |--------------|----------------------------------|----------------|  
    | backward_loss | Loss for backpropagation       | Tensor(1)      |  
    | pred_mask    | Predicted mask for metric evaluation | Tensor(B,1,H,W) |  
    | pred_label   | Predicted binary label          | Tensor(B,1)    |  
    | visual_loss  | Scalars for visualization       | Dict()         |  
    | visual_image | Images, masks, and heatmaps    | Dict()         |  

  **Ensure adherence to this format to integrate seamlessly with evaluation and visualization tools.**  

This concludes the key aspects and best practices for implementing models in IMDLBenCo.