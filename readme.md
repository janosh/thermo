<h1 align='center'>Data-Driven Risk-Conscious<br />Thermoelectric Materials Discovery</h1>

<h4 align='center'>

[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/janosh/thermo/main.svg)](https://results.pre-commit.ci/latest/github/janosh/thermo/main)
[![This project supports Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg?logo=python&logoColor=white)](https://python.org/downloads)
[![GitHub Repo Size](https://img.shields.io/github/repo-size/janosh/thermo?label=Repo+Size)](https://github.com/janosh/thermo/graphs/contributors)
</h4>

## Project description

The aim is to discover high-figure–of-merit ($zT > 1$) and sustainable (lead-free and rare earth-free) bulk thermoelectrics using machine learning-guided experimentation. The key advance is going beyond 'big data' which in this domain is unattainable for the foreseeable future since both first-principles calculations and experimental synthesis and characterization of bulk thermoelectrics are costly and low throughput. Instead, we move towards so-called 'optimal data' by developing novel algorithms that optimize thermoelectric performance ($zT$) with minimal number of expensive calculations and experiments.

To date, there has been no statistically robust approach to simultaneously incorporate experimental and model error into machine learning models in a search space with high opportunity cost and high latency (i.e. large time between prediction and validation).

Consequently, searches have been unable to effectively guide experimentalists in the selection of exploring or exploiting new materials when the validation step is inherently low throughput and resource-intensive, as is the case for synthesizing new bulk functional materials like thermoelectrics. This project aims to implement a holistic pipeline to discover novel thermoelectrics: ML models predict the $zT$ of a large database of structures as well as their own uncertainty for each prediction. Candidate structures are then selected, based on maximizing $zT$ subject to a tolerable level of uncertainty, to proceed to the next stage where expensive experimental synthesis and characterization of high-$zT$ candidates are guided by Bayesian optimization and active machine learning.

## Setup

To check out the code in this repo, reproduce results and start contributing to the project, clone the repo and create a `conda` environment containing all dependencies by running the following command (assumes you have `git` and `conda` installed)

```sh
git clone https://github.com/janosh/thermo \
&& cd thermo \
&& pip install -r requirements.txt
&& pre-commit install
```

## Usage

Run any of the files in [`src/notebooks`](https://github.com/janosh/thermo/tree/main/notebooks). The recommended way to work with those files is using [VS Code](https://code.visualstudio.com) and its [Python extension](https://marketplace.visualstudio.com/items?itemName=ms-python.python). You'll see the results of running those files [in an interactive window](https://code.visualstudio.com/docs/python/jupyter-support-py) (similar to Jupyter).
