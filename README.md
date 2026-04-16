# ZOOM (Zoom from Macro to Micro)

## Decoding molecular and cellular underpinnings of macroscopic spatial brain phenotypes with ZOOM
ZOOM is a Python-based integrative framework for linking macroscopic and mesoscopic spatial brain phenotypes (SBPs) with microscopic molecular, cellular information.

[![python >=3.9](https://img.shields.io/badge/python-%3E%3D3.9-brightgreen)](https://www.python.org/) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18821772.svg)](https://doi.org/10.5281/zenodo.18821772)

![avatar](docs/demonstration.png)

Code used to generate results of this study can be found [here](paper).

## Tutorial
For quick start, you can download [demostration data](https://drive.google.com/file/d/1kQ6hbkaS7PvuaTy3hyCcY3cVNnD_gjvy/view?usp=sharing) and run the following code, which will decode the single-cell underpinnings of Adolescent change in myelination (ΔMT).
快速下载Python包zoom

```
conda create -n zoom_env python=3.9
conda activate zoom_env
cd ZOOM-main
pip install -r requirements.txt
pip install .
```
用于Demonstration的代码
```
import scanpy as sc
import pandas as pd
import numpy as np
import zoom
import os
os.chdir("/.../demo")

# Prepare AHBA expression and SBP data
expression = pd.read_csv("expression_HCPMMP.csv",index_col=0)
SBP, SBP_perm = zoom.prepare.process_SBP(
    SBP="lh.deltaMT.fsLR.32k.func.gii",
    parcellation="lh.HCPMMP1.fsLR.32k.label.gii",
    atlas="fsLR", density="32k", hemi="L",
    n_perm=1000, seed=123
)
# Keep valid regions
SBP = SBP[SBP.index.isin(expression.index)]
SBP_perm = SBP_perm[SBP_perm.index.isin(expression.index)]
# Prepare ZOOM_SC object for analysis
adata = sc.read_h5ad("adata_ctx.h5ad")
zoom_obj = zoom.ZOOM_SC(
    adata=adata,
    expression=expression,
    SBP=SBP,
    SBP_perm=SBP_perm,
    best_comp=9, #This can be tested via zoom, but takes time
    processed = True
)
zoom_obj.cv_PLSR()
zoom_obj.get_gene_contrib(metric="VIP")
# Compute ZOOM single-cell score
zoom_obj.get_SBP_score(
    direction=True,
    gene_size=30,
    fdr_method="group_bh",
    alpha=0.1,
    group="Subcluster"
)
# For visualization
zoom_obj.adata.obs["norm_score"] = zoom_obj.SBP_scores["norm_score"]
zoom_obj.adata.obs["p_fdr0.1"] = zoom_obj.SBP_scores["p_fdr0.1"]
zoom_obj.adata.obs['sig_score'] = zoom_obj.adata.obs['norm_score'].where((zoom_obj.adata.obs['p_fdr0.1']=='True')&(zoom_obj.adata.obs['norm_score']>3), other=np.nan)
sc.pl.embedding(zoom_obj.adata, basis='umap', color='Cluster', size=5)
sc.pl.embedding(zoom_obj.adata, basis='umap', color='norm_score',
                size=5,color_map='magma',frameon=False)
sc.pl.embedding(zoom_obj.adata, basis='umap', color='sig_score', 
                vmin=min(zoom_obj.adata.obs["norm_score"]),
                vmax=max(zoom_obj.adata.obs["norm_score"]),
                na_color='lightgray', size=5, color_map='magma', frameon=False)
```
这在48个核的Linux服务器上大概需要3min左右的运行时间。

Detailed guidance of ZOOM can be found [here](https://zoom-tutorial.readthedocs.io/en/latest/).

## Citation
Our article has not been published yet.
