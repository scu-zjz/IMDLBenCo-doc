# Installation

**Please note that, unlike most code provided in papers, the Benco repository is not intended to be used via methods like `git clone`, as the code involves numerous components required for engineering development. The expected method of usage is through `pip install`, treating it as a Python library.**

## For Regular Users  
If you only wish to use IMDL-BenCo to reproduce the paper and build your own model, the installation process is very simple. Currently, IMDL-BenCo is managed via PyPI, and you can complete the installation by running the following command:

```shell
pip install imdlbenco
```

## For Developers Contributing to the Official Repository  
If you are trying to develop new features for the **IMDL-BenCo Python Library** and contribute to the official repository, follow the instructions in this section. It is recommended to first uninstall any previously installed versions of IMDL-BenCo in your environment. Then, clone your forked repository of IMDL-BenCo, switch to the `dev` branch to get the latest "development version," and use the special command `pip install -e .` to complete the local installation. This will ensure that the current Python environment always executes the IMDL-BenCo library based on the scripts in this directory and automatically updates the corresponding behavior when files are updated, which is highly convenient for debugging and development. 

```shell
# Uninstall any existing IMDL-BenCo library
pip uninstall imdlbenco

# Clone your forked IMDL-BenCo repository from GitHub
git clone https://github.com/your_name/IMDL-BenCo.git

# Enter the project directory
cd IMDL-BenCo

# Switch to the dev branch
git checkout dev

# Perform a local development installation using `pip install -e .`
pip install -e .

# Verify the installation
pip show imdlbenco
```

If the installation is successful, after executing `pip list`, you should see something like this:

```
Package                 Version            Editable project location
----------------------- ------------------ ------------------------------------------------------
...
IMDLBenCo               0.1.10             /mnt/data0/xiaochen/workspace/IMDLBenCo_pure/IMDLBenCo
...
```

The presence of a corresponding path in the `Editable project location` column indicates that any modifications to the Python scripts in this path will take effect directly in the Python environment without the need for reinstallation. This is very convenient for debugging.