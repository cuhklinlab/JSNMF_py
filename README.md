# JSNMF
JSNMF: Joint analysis of scRNA-seq and epigenomic-seq data via semi-orthogonal nonnegative matrix factorization.

This is the Python implementation of the JSNMF algorithm. Note that this implementation can employ GPU to speed up the code, just 
simply submit the script to the GPU clusters is okay.

## 1. Installation
You can use the following command to install JSNMF:
```
pip install JSNMF-py
```

## 2. Usage
The main class `JSNMF` needs to be initialized with at least two [annda.AnnData](https://anndata.readthedocs.io/en/latest/anndata.AnnData.html#anndata.AnnData), for`RNA`, and `ATAC` data, or data from two modalities, respectively. The preprocessed data is stored in `RNA.X` and `ATAC.X`. And the true cell labels should be kept in `RNA.obs['celltype']`. The default number of maximum epochs to run, i.e. the `max_epochs` parameter, is set as 200.  





You can refer to [here](https://github.com/cuhklinlab/JSNMF_py/blob/main/Example/demo_jsnmf.ipynb) for a simple illustration of the use of JSNMF.


