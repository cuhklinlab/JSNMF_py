# JSNMF_py
JSNMF: Joint analysis of scRNA-seq and epigenomic-seq data via semi-orthogonal nonnegative matrix factorization.

This is the Python implementation of the JSNMF algorithm. Note that this implementation can employ GPU to speed up the code, just 
simply submit the script to the GPU clusters will work.

## 1. Installation
You can use the following command to install JSNMF:
```
pip install JSNMF-py
```

## 2. Usage
The main class `JSNMF` needs to be initialized with at least two [anndata.AnnData](https://anndata.readthedocs.io/en/latest/anndata.AnnData.html#anndata.AnnData) object, for`RNA`, and `ATAC` data, or data from two modalities, respectively. The preprocessed data is stored in `RNA.X` and `ATAC.X`. And the true cell labels should be kept in `RNA.obs['celltype']`. Note that the data preprocessing process is done with R. The default number of maximum epochs to run, i.e. the `max_epochs` parameter, is set as 200. So it is quite simple to initialize a JSNMF model with the following code:
```
from JSNMF.model import JSNMF
test_model = JSNMF(rna,atac)
```
After initializing, run the model is also quite easy: 
```
test_model.run()
```
The result is saved in `test_model.result`, which is a dict, and the major output of JSNMF, the complete graph S, can be get with easy access:
```
S = test_model.result['S']
```
`JSNMF` class also has other methods, you can use the `help` or `?` command for more details explanations of the methods.


## 3. Tutorials
Please refer to [here](https://github.com/cuhklinlab/JSNMF_py/blob/main/Tutorials/jsnmf_tutorial.ipynb) for a simple illustration of the use of JSNMF model, with clustering and visualization results shown. The guide for obtaining the gene-peak association can be found [here](https://github.com/cuhklinlab/JSNMF_py/blob/main/Tutorials/Gene_Assoc_Tutorial.md). And the tutorial of using some extensions or variants of JSNMF model is [here](https://github.com/cuhklinlab/JSNMF_py/blob/main/Tutorials/var_jsnmf_tutorials.md). 
