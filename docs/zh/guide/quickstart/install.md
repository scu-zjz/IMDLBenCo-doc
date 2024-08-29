# 安装

**请注意，与大多数Paper提供的代码不同，Benco的仓库本身并不期望通过git clone等方式完成使用，因为时机代码中涉及到大量的工程开发所需的组件。期望的使用方法是直接通过pip install并将其当做Python库使用**

## 普通用户
IMDL-BenCo的安装方式非常简单，目前通过PyPI进行了包管理，直接通过如下指令即可完成
```shell
pip install imdlbenco
```

## 开发者
如果你试图为IMDL-BenCo的**Python Library**在本地开发新功能并贡献到仓库，推荐你先在环境中卸载所有已经安装的IMDL-BenCo，然后克隆您fork过的IMDL-BenCo的仓库后，切换到dev分支获得最新的“开发版本”后，并使用特殊的`pip install -e . `指令完成本地安装，这会使得当前Python环境始终根据本路径下的包所含的脚本执行IMDL-BenCo库的行文，并在更新文件时自动更新相应行为，非常便于调试开发。

```shell
# 卸载已有的 IMDL-BenCo 库
pip uninstall imdlbenco

# 克隆 GitHub 上的 IMDL-BenCo 的fork过的仓库
git clone https://github.com/your_name/IMDL-BenCo.git

# 进入项目目录
cd IMDL-BenCo

# 切换到 dev 分支
git checkout dev

# 使用 `pip install -e .` 进行本地开发安装
pip install -e .

# 验证安装
pip show imdl-benco
```