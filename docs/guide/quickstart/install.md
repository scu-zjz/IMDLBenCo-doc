# Installation

**Please note that, unlike most code provided in papers, the Benco repository is not intended to be used via methods like `git clone`, as the code involves numerous components required for engineering development. The expected method of usage is through `pip install`, treating it as a Python library.**

## Regular Users
Installing IMDL-BenCo is very straightforward. Currently, it is managed via PyPI, so you can install it with the following command:

```shell
pip install imdlbenco
```

## Developers
If you are attempting to develop new features for the **IMDL-BenCo Python Library** locally and contribute to the repository, it is recommended that you first uninstall any existing IMDL-BenCo installations in your environment. Then, clone your forked IMDL-BenCo repository, switch to the `dev` branch to get the latest "development version," and use the special command `pip install -e .` to complete the local installation. This will ensure that your current Python environment always executes the IMDL-BenCo library scripts based on the files in your local directory, automatically updating behavior when files are modified, making it convenient for debugging and development.

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
pip show imdl-benco
```