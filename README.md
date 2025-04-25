# HTGNN: Graph Neural Networks for Virtual Sensing in Complex Systems: Addressing Heterogeneous Temporal Dynamics

[![Paper DOI](https://img.shields.io/badge/DOI-10.1016/j.ymssp.2025.112544-blue)](https://doi.org/10.1016/j.ymssp.2025.112544)
[![Zenodo DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14972954.svg)](https://doi.org/10.5281/zenodo.14972954)
[![Zenodo DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14959000.svg)](https://doi.org/10.5281/zenodo.14959000)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/graph-neural-networks-for-virtual-sensing-in/load-virtual-sensing-on-skf-bls-dataset)](https://paperswithcode.com/sota/load-virtual-sensing-on-skf-bls-dataset?p=graph-neural-networks-for-virtual-sensing-in)

This repository contains the official implementation for the paper "Graph Neural Networks for Virtual Sensing in Complex Systems: Addressing Heterogeneous Temporal Dynamics".

**Publication Status:**
Our paper has been published in the **Mechanical Systems and Signal Processing (MSSP)** journal. You can access the official version here:
[https://doi.org/10.1016/j.ymssp.2025.112544](https://doi.org/10.1016/j.ymssp.2025.112544)

## Overview

Current data-driven virtual sensing approaches often struggle in complex industrial systems due to sensors exhibiting diverse temporal dynamics (modalities) and the influence of varying operating conditions. To address these limitations, this project introduces the **Heterogeneous Temporal Graph Neural Network (HTGNN)**, a novel framework specifically designed for virtual sensing in these challenging environments.

HTGNN explicitly models signals with distinct temporal dynamics as separate node types within a graph structure. It also incorporates operating condition context, allowing it to:

* Effectively fuse information from diverse sensor modalities.
* Account for the varying influence of exogenous variables (like load or speed).
* Model both **intra-modality** and **inter-modality** interactions distinctly.
* Capture unique temporal dynamics for each sensor modality under changing conditions using novel **operating condition-aware modules**.

This approach enables more accurate prediction of essential parameters crucial for Prognostics and Health Management (PHM), overcoming limitations of existing methods. To the best of our knowledge, HTGNN represents the first architecture specifically designed to analyze such diverse sensor modalities while accounting for exogenous variables in virtual sensing.

Key contributions validated in our paper include the heterogeneous interaction modeling, the operating condition-aware modules, and a comprehensive ablation study analyzing the impact of each component. Additionally, we contribute two new, publicly available datasets (Bearing and Bridge) specifically designed to advance research in virtual sensing with multimodal sensor data and varying operating conditions (see Datasets section).


## Environment Setup
This project relies on specific dependencies and packages, which are defined in the eff_env.yml file. You can set up the environment using Conda by running the following command:

```bash
conda env create -f env.yml
```

If you want to update the environment, you can run the following command:

```bash 
conda env update --file env.yml  --prune
```

### Installing torch with GPU support
To install PyTorch with CUDA support, use the following command:

```bash 
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=10.2 -c pytorch
```

Depending on your CUDA version, you may need to change the `cudatoolkit` version.
Detailed instructions can be found [here](https://pytorch.org/get-started/previous-versions/).
Note pyTorch 1.12.* binaries do not support CUDA versio above (including) 11.7.


### Installing pytorch-geometric
Follow [PyG 2.2.0 INSTALLATION Guide](https://pytorch-geometric.readthedocs.io/en/2.2.0/notes/installation.html) for detailed instructions.


## Datasets
The datasets used in the paper are publicly available on Zenodo:

- **Bearing Dataset**: Vibration data from a bearing test rig. [Download Link](https://zenodo.org/records/14959001)
- **Bridge Dataset**: Sensor data from a structural health monitoring system on a bridge. [Download Link](https://zenodo.org/records/14972955)

Instructions:

Download the desired dataset(s) from the links above.
Create the dataset directory if it doesn't exist:
Place the downloaded dataset files into the `run/datasets` directory.


## Usage
To train the model, run the following command:

```bash 
python run/main.py --cfg run/configs/[DATASET]/[MODEL].yaml --repeat N
``` 

where:
- `--cfg run/configs/[DATASET]/[MODEL].yaml`: Specifies the path to the configuration file.
- Replace `[DATASET]` with bearing or bridge.
- Replace `[MODEL]` with the model configuration you want to use (e.g., htgnn, timemixer).
- `--repeat N`: Specifies the number of times to repeat the experiment with different random seeds (e.g., --repeat 5).


For any questions or feedback, please open an issue in this repository or contact us directly via email.

## Citation
If you find this code or the datasets useful in your research, please cite our paper using:

```bibtex
@article{zhao2025graph,
  title={Graph neural networks for virtual sensing in complex systems: Addressing heterogeneous temporal dynamics},
  author={Zhao, Mengjie and Taal, Cees and Baggerohr, Stephan and Fink, Olga},
  journal={Mechanical Systems and Signal Processing},
  volume={230},
  pages={112544},
  year={2025},
  publisher={Elsevier}
}
```
