# Installation

## For Regular Users

The installation of IMDL-BenCo is very simple. It is currently managed via PyPI, and can be installed with the following command:

```shell
pip install imdlbenco
```

## For Developers

If you are looking to develop new features for IMDL-BenCo locally, it is recommended to first uninstall any existing versions of IMDL-BenCo from your environment. Then, clone the forked IMDL-BenCo repository, switch to the `dev` branch to get the latest "development version", and use the special `pip install -e .` command for local installation. This will ensure that the current Python environment always executes the IMDL-BenCo library scripts from the package in the local path and automatically updates the behavior as you make changes, which is very convenient for debugging and development.

```shell
# Uninstall the existing IMDL-BenCo library
pip uninstall imdlbenco

# Clone your forked IMDL-BenCo repository from GitHub
git clone https://github.com/your_name/IMDL-BenCo.git

# Navigate to the project directory
cd IMDL-BenCo

# Switch to the dev branch
git checkout dev

# Perform a local development installation with `pip install -e .`
pip install -e .

# Verify the installation
pip show imdlbenco
```