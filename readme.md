# Data-Driven Risk-Conscious Thermoelectric Materials Discovery

[![License](https://img.shields.io/github/license/janosh/thermo?label=License)](/license)
[![GitHub Repo Size](https://img.shields.io/github/repo-size/janosh/thermo?label=Repo+Size)](https://github.com/janosh/thermo/graphs/contributors)
[![GitHub last commit](https://img.shields.io/github/last-commit/janosh/thermo?label=Last+Commit)](https://github.com/janosh/thermo/commits)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/janosh/thermo/master.svg)](https://results.pre-commit.ci/latest/github/janosh/thermo/master)

## Project description

The aim is to discover high figure of merit (_zT_ > 1) and sustainable (lead-free and rare earth-free) bulk thermoelectrics using machine learning-guided experimentation. The key advance is going beyond 'big data' which in this domain is unattainable for the foreseeable future since both first principles calculations and experimental synthesis and characterization of bulk thermoelectrics are costly and low throughput. Instead, we move towards so-called 'optimal data' by developing novel algorithms that optimize thermoelectric performance (_zT_) with minimal number of expensive calculations and experiments.

To date there has been no statistically robust approach to simultaneously incorporate experimental and model error into machine learning models in a search space with high opportunity cost and high latency (i.e. large time between prediction and validation).

Consequently, searches have been unable to effectively guide experimentalists in the selection of exploring or exploiting new materials when the validation step is inherently low throughput and resource-intensive, as is the case for synthesizing new bulk functional materials like thermoelectrics. This project aims to implement a holistic pipeline to discover novel thermoelectrics: ML models predict the _zT_ of a large database of structures as well as their own uncertainty for each prediction. Candidate structures are then selected, based on maximizing _zT_ subject to a tolerable level of uncertainty, to proceed to the next stage where expensive experimental synthesis and characterization of high-_zT_ candidates are guided by Bayesian optimization and active machine learning.

## Setup

To check out the code in this repo, reproduce results and start contributing to the project, clone the repo and create a `conda` environment containing all dependencies by running the following command (assumes you have `git` and `conda` installed)

```sh
git clone https://github.com/Lee-Group/thermo \
&& cd thermo \
&& conda env create -f env.yml \
&& pre-commit install
```

## Usage

### Locally

Run any of the files in under [`src/notebooks`](https://github.com/janosh/thermo/tree/master/src/notebooks). The recommended way to work with this project is using [VS Code](https://code.visualstudio.com) along with its [Python extension](https://marketplace.visualstudio.com/items?itemName=ms-python.python). You'll see the results of running those files rendered in [VS Code's interactive window](https://code.visualstudio.com/docs/python/jupyter-support-py) (similar to Jupyter in purpose and functionality).

In VS Code, you'll probably want to add the following setting for local module imports to work and for code changes in imported modules to be auto-reloaded into the active Jupyter session:

```json
"python.dataScience.runStartupCommands": [
  "%load_ext autoreload",
  "%autoreload 2",
  "import sys",
  "sys.path.append('${workspaceFolder}')",
  "sys.path.append('${workspaceFolder}/src')",
],
```

### HPC

To submit a job to [Cambridge University's CSD3](https://www.hpc.cam.ac.uk) HPC facility ([docs](https://docs.hpc.cam.ac.uk/hpc)):

1. Connect via `ssh` using your [CRSid](https://help.uis.cam.ac.uk/new-starters/it-for-students/student-it-services/your-crsid) and password, e.g.

   ```sh
   ssh jr769@login-gpu.hpc.cam.ac.uk
   ```

2. Copy over the directory using `rsync`

   ```sh
   rsync -av --delete --include-from=hpc/rsync.include . jr769@login-gpu.hpc.cam.ac.uk:thermoelectrics
   ```

   See `hpc/rsync.include` for a list of files that will be transferred to your CSD3 home directory. You can also simulate this command before executing it with the `--dry-run` option.

3. To submit a single HPC job, enter

   ```sh
   sbatch hpc/gpu_submit
   ```

   For a job array, first modify the GPU submission script at `./hpc/gpu-submit` and make sure in the section `#! Run options for the application:` you comment out the line below `# single job` and uncomment the line `# array job`. Then again issue the `sbatch` command, this time including the `--array` option. E.g. to submit 16 jobs at once, use

   ```sh
   sbatch --array=0-15 hpc/gpu_submit
   ```

For a more user-friendly experience, you can also [request cluster resources through Jupyter](https://docs.hpc.cam.ac.uk/hpc/software-packages/jupyter.html) by first instantiating a notebook server and then specifying that as a [remote server in VS Code's interactive window](https://code.visualstudio.com/docs/python/jupyter-support#_connect-to-a-remote-jupyter-server).

## Environment

The environment file `env.yml` was generated with `conda env export --no-builds > env.yml`. You can recreate the environment from this file via `conda env create -f env.yml`.

The environment `thermo` was originally created by running the command:

```sh
conda create -n thermo python pip \
  && conda activate thermo \
  && pip install numpy pandas tensorflow tensorflow-probability automatminer scikit-learn scikit-optimize jupyter matplotlib seaborn plotly umap-learn pytest ipykernel
  && conda install pytorch -c pytorch
  && conda install gurobi -c http://conda.anaconda.org/gurobi
```

You can delete the environment with `conda env remove -n thermo`.

To update all packages and reflect new versions in this file, use

```sh
conda update --all \
  && pip list --outdated --format=freeze | grep -v '^\-e' | cut -d = -f 1  | xargs -n1 pip install -U \
  && conda env export --no-builds > env.yml
```
