# 安装

**请注意，与大多数Paper提供的代码不同，Benco的仓库本身并不期望通过git clone等方式完成使用，因为时机代码中涉及到大量的工程开发所需的组件。期望的使用方法是直接通过pip install并将其当做Python库使用**

## 对于普通用户
如果你只希望用IMDL-BenCo复现论文，并构建自己的模型，则IMDL-BenCo的安装方式非常简单，目前通过PyPI进行了包管理，直接通过如下指令即可完成
```shell
pip install imdlbenco
```
如果安装成功，在命令行中运行如下指令可以验证是否安装成功，并且自动检查是否有新版本。（本仓库处于迭代开发中，最好经常检查版本更新）
```shell
benco -v 
```

```shell
benco --version
```
如果正常安装了最新版本，应该可以看到如下内容：
```
IMDLBenCo codebase version: 0.1.21
        Checking for updates...
        Local version:  0.1.21
        PyPI newest version:  0.1.21
You are using the latest version: 0.1.21.
```

像其他的命令行工具一样，运行`benco -h`可以查看帮助引导等内容。

## 对于想为官方仓库贡献的开发者
如果你试图为IMDL-BenCo的**Python Library**在开发新功能并贡献到官方仓库，则需要按照本段完成。推荐你先在环境中卸载所有已经安装的IMDL-BenCo，然后克隆您fork过的IMDL-BenCo的仓库后，切换到dev分支获得最新的“开发版本”后，并使用特殊的`pip install -e . `指令完成本地安装，这会使得当前Python环境始终根据本路径下的包所含的脚本执行IMDL-BenCo库的行文，并在更新文件时自动更新相应行为，非常便于调试开发。

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
pip show imdlbenco
```

如果安装正常，执行`pip list`后应当看到
```
Package                 Version            Editable project location
----------------------- ------------------ ------------------------------------------------------
...
IMDLBenCo               0.1.10             /mnt/data0/xiaochen/workspace/IMDLBenCo_pure/IMDLBenCo
...
```
`Editable Project Location` 这一栏有对应的路径，即代表所有对于该路径下的python脚本修改可以直接生效于该Python环境内部，无需重新安装，非常便于调试。