
<p align="center" width="100%">
  <img src='figures/title.png' width="95%">
</p>

[📘Documentation](https://xxx/) |
[🚀Installation](https://github.com/qiqi5757/UniBIP?tab=readme-ov-file#-get-started) |
[👀Visualization](https://github.com/qiqi5757/UniBIP?tab=readme-ov-file#-showcase) |
[🛠️Example](https://github.com/qiqi5757/UniBIP#%EF%B8%8F%EF%B8%8F%EF%B8%8Fexample) |
[🌟Guide](https://github.com/qiqi5757/UniBIP#-quick-start-guide) |

 <p align="center" width="100%">
  <img src='figures/The framework of UniBIP.png' width="100%">
</p>


## 📑 Datasets


| Dataset       | Source data                              | Data Code                                 |
|---------------|------------------------------------------|-------------------------------------------|
| PPI01         | https://doi.org/10.5281/zenodo.7600622.  | https://github.com/zqgao22/HIGH-PPI       |
| PPI02         | https://doi.org/10.5281/zenodo.13752181  | https://github.com/rui-yan/DNE            |
| PPI03         | https://tinyurl.com/networks-HuRI-paper  | https://github.com/kexinhuang12345/SkipGNN|
| DDI01         | https://go.drugbank.com/                 | https://github.com/F-windyy/DGATDDI       |
| DDI02         | https://doi.org/10.5281/zenodo.10016715  | https://github.com/LARS-research/EmerGNN  |
| DDI03         | http://snap.stanford.edu/biodata/        | https://github.com/kexinhuang12345/SkipGNN|
| DTI01         | https://zenodo.org/records/14847966      | https://github.com/CSUBioGroup/DTIAM      |
| DTI02         | https://zenodo.org/records/14847966      | https://github.com/CSUBioGroup/DTIAM      |
| DTI03         | http://snap.stanford.edu/biodata/        | https://github.com/kexinhuang12345/SkipGNN|
| DTA01         | https://zenodo.org/records/14847966      | https://github.com/CSUBioGroup/DTIAM      |
| DTA02         | https://zenodo.org/records/14847966      | https://github.com/CSUBioGroup/DTIAM      |
| circRNA-drug  | https://hanlab.tamhsc.edu/cRic/          | https://github.com/yinboliu-git/MHGTCDA   |
| drug-gene     | https://github.com/wentao228/DGCL        | https://github.com/wentao228/DGCL         |
| gene-disease  | http://www.disgenet.org/                 | https://github.com/kexinhuang12345/SkipGNN|
| miRNA-disease | http://www.cuilab.cn/hmdd                | https://github.com/a1622108/MDA-CF        |

## 🚀🚀🚀Get Started

This guide will help you quickly configure and run the UniBIP project.

#### 📦 Environment Setup


**1. Create Conda Environment:**

In the project's root directory, the `environment.yml` file defines all the necessary dependencies. Please use the following command to create the environment:


```python
conda env create -f environment.yml
```



**2. Activate the Environment:**


```
conda activate UniBIP_env
```

**3. Install the Project Package:**

After activating the environment, run the following command to install the project itself. This will ensure all internal modules and dependencies are correctly linked.

```bash
pip install .
```

##  🛠️🛠️🛠️Example 


The training module is located at `example/main.py`. You can start the training process by running `main.py`. The training module is located at `example/main.py`. You can start the training process by running `main.py`. The dataset is located in the`example/dataset` directory. Given the substantial size of the dataset, a full upload is not feasible. The complete dataset and others can be found at the following URL: [https://figshare.com/s/4f1adea54bde58c8dbc6](https://figshare.com/s/4f1adea54bde58c8dbc6).

```python
python main.py
```
-----
##  🌟🌟🌟 Quick Start Guide
The UniBIP model automatically adapts to the type of graph data you input. For a **homogeneous network**, you can simply set the `mode` parameter to `'I'`, while for a **heterogeneous network**, you'll set it to `'H'`.

Furthermore, you can specify the directionality of the edges using the `undirected` parameter: `undirected=True` indicates **undirected edges** (or bidirectional edges), whereas `undirected=False` indicates **directed edges** (or unidirectional edges).

Notably, the model also features **multi-task** adaptability, **automatically** determining whether to perform **binary qualitative** or **multi-class qualitative** prediction based on the data, without requiring manual configuration. This design greatly simplifies the user experience, allowing you to focus on the task itself rather than on complex model settings.

-----
## 🏆🏆🏆 Showcase
<p align="center">
   <img src='figures/visualizing surface.jpg' width="100%">
<br><br>
<b>Figure 1: </b>a, Presents the surface interaction model of the protein (Q9Y6M4) and the drug (CHEMBL408982). b, The right side zooms in on the core interaction region, presenting the spatial interaction pattern between the drug molecule CHEMBL408982 (methyl carbon) and the protein residue (ASP-300, corresponding to site 300 of the Q9Y6M4 protein). 
</p>



