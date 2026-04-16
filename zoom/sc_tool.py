# -*- coding: utf-8 -*-
"""
Functionality for preprocessing scRNA-seq data.
"""

import scanpy as sc
import anndata as ad
import numpy as np
import pandas as pd
import gseapy as gp
import os
from scipy.stats import ttest_1samp
from statsmodels.stats.multitest import multipletests
from scipy.stats import gmean, rankdata
from multiprocessing import Pool
from tqdm import tqdm
import scipy.sparse as sp
from joblib import Parallel, delayed
import subprocess
from typing import (Tuple, Union, Optional, Dict, Any)
from zoom.data_loader import load_sc, load_df

def preprocess(
    adata: ad.AnnData,
    QC: bool,
    min_genes: int,
    min_cells: int,
    d: int,
    expression: pd.DataFrame,
    DS: Union[os.PathLike, pd.DataFrame],
) -> ad.AnnData:
    
    """
    Calculates the gene expression rank for each cell.

    Parameters
    ----------
    adata : ad.AnnData
        The AnnData object containing expression data.
        - `adata.X` should contain the raw or normalized expression data.
    QC : bool
        If True, filter low-quality genes and cells through scanpy
        - sc.pp.filter_cells(adata, min_genes=min_genes)
        - sc.pp.filter_genes(adata, min_cells=min_cells)
    d : int
        Number of nearest neighbors for each cell.
    expression : pd.DataFrame
        The AHBA gene expression matrix.
    DS : pd.DataFrame
        Gene-level differential stability(DS).

    Returns
    -------
    adata : ad.AnnData
        The AnnData object containing common genes only and
        gene-level differential stability information.
    expression : pd.DataFrame
        The AHBA gene expression containing common genes only.
    """
    
    # Load scRNA-seq data
    # adata = load_sc(adata,flag_sparse=True)
    # Filter low-quality genes and cells
    if QC:
        sc.pp.filter_cells(adata, min_genes=min_genes)
        sc.pp.filter_genes(adata, min_cells=min_cells)
        
    # Find nearest neighbors for each cell
    sc.pp.neighbors(adata, n_pcs=30, n_neighbors=d)
    
    # Keep commnon genes between scRNA-seq data and AHBA
    sc_genes = list(adata.var_names)
    ahba_genes = list(expression.columns)
    common_genes = [g for g in ahba_genes if g in set(sc_genes)]
    expression = expression.loc[:, common_genes]
    DS = load_df(DS)
    DS = DS.loc[common_genes]
    adata = adata[:, common_genes].copy()    
    # Add DS on the AnnData object
    adata.uns['GENE_STATS'] = DS    
    return expression, adata

def _rank_cell(data):
    """Help function for ranking single-cell"""
    return rankdata(data, method="average").astype(np.float32)

def rank_expression(adata: ad.AnnData) -> ad.AnnData:
    
    """
    Calculates the gene expression rank for each cell.

    Parameters
    ----------
    adata : ad.AnnData
        The AnnData object containing expression data.

    Returns
    -------
    adata : ad.AnnData
        The AnnData object with a new layer `adata.layers['rank']`
        containing the gene expression rank matrix.
    """

    expr_mtx = adata.X.toarray()
    n_cells = adata.n_obs

    # Parallel computation
    with Pool() as pool:
        results = list(tqdm(pool.imap(_rank_cell, expr_mtx),
                            total=n_cells, desc="Computing ranks per cell"))    
    adata.layers['rank'] = np.vstack(results)
    return adata

def compute_gss(
    adata: ad.AnnData,
    n_jobs: int
) -> ad.AnnData:
    
    """
    Calculates the gene expression rank for each cell.

    Parameters
    ----------
    adata : ad.AnnData
        The AnnData object containing expression data.
    gss_limit : float
        Allowed maximum GSS value to avoid over-representation.
    n_jobs : int
        Number of cores for parallel computation.    

    Returns
    -------
    adata : ad.AnnData
        The AnnData object with a new layer `adata.layers['gss']`
        containing the gene specificity scores.
        
    References
    ----------
    Song, L., Chen, W., Hou, J., Guo, M. & Yang, J. Spatially 
    resolved mapping of cells associated with human complex traits. 
    Nature 641 932-941 (2025).
    """
    
    # Compute geometric mean of gene ranks
    rank_mtx = adata.layers["rank"].astype(np.float32)
    gm_all = gmean(rank_mtx, axis=0).reshape(-1,1)
    # Compute the proportion of gene expression in cells
    adata_X_bool = adata.X.astype(bool)
    frac_all = (np.asarray(adata_X_bool.sum(axis=0)).flatten()).reshape(-1,1)/adata_X_bool.shape[0]
    
    # Find nearest neighbors for each cell
    D = adata.obsp["distances"]
    indptr = D.indptr.astype(np.int64)
    indices = D.indices.astype(np.int64)
    
    # Help function for computing gene specificity scores
    def compute_gss_i(i, rank_mtx, adata_X_bool, indptr, 
                      indices, gm_all, frac_all):
        # Find nearest neighbors
        start = indptr[i]
        end = indptr[i+1]
        d = end-start-1
        nn_idx = indices[start:end]
        # Compute mean ranks and expression fraction on microdomain
        gm_sub = gmean(rank_mtx[nn_idx,:], axis=0).reshape(-1,1)
        frac_sub = (np.asarray(adata_X_bool[nn_idx].sum(axis=0)).flatten()).reshape(-1,1)/(d+1)
        # Compute gene specificity score
        spcy = gm_sub/gm_all
        # Enforce sparsity and filter outlier
        spcy[frac_sub<frac_all] = 0
        spcy[spcy<1] = 0
        # Exponential power scale
        spcy = np.exp(np.square(spcy))-1    
        return spcy.ravel()
    
    # Parallel computation
    results = Parallel(n_jobs=n_jobs)(
        delayed(compute_gss_i)(i,rank_mtx,adata_X_bool,indptr,indices,gm_all,frac_all) 
        for i in tqdm(range(adata.shape[0]), desc="Computing gene specificity scores (GSS)")
    )
    gss_mtx = np.vstack(results)
    gss_mtx = sp.csr_matrix(gss_mtx)
    adata.layers["gss"] = gss_mtx
    
    # Add GSS statistics to AnnData object
    gss_max = adata.layers["gss"].max(axis=0)
    gss_max = gss_max.toarray().ravel()
    adata.uns['GENE_STATS']["gss_max"] = gss_max
    del adata.layers['rank']
    return adata

def select_ctrl(
    weight_perm: pd.DataFrame,
    sign_perm: pd.DataFrame,
    gene_rep: pd.DataFrame,
    weight_key: str,
    gene_stats: pd.DataFrame,
    qbin_counts: dict,
    direction: bool,
    n_jobs: int
) -> Tuple[dict, dict]:
    
    """
    Calculates the gene expression rank for each cell.

    Parameters
    ----------
    weight_perm : pd.DataFrame
        Gene weights in permutation tests.
    sign_perm : pd.DataFrame
        Signed gene weights in permutation tests.
    gene_rep : pd.DataFrame
        Results for PLS-R, must contain column `weight_key`.
    weight_key: str
        Column name indicating the original gene weights for scoring.
    gene_stat : pd.DataFrame
        Gene statistics of scRNA-seq dataset.
    qbin_counts : dict
        Dictory of gene counts in each bin.
    direction : bool
        If True, find gene set relevant to the positive direction
        of given SBP else negative direction.
    n_jobs : int
        Number of cores used for parallel computation.

    Returns
    -------
    dic_ctrl_list : dict
        Control gene sets selected based on spatial permutation tests.
    dic_ctrl_weight : dict
        Corresponding gene weights of control gene sets.
        
    References
    ----------
    Fulcher, B. D., Arnatkeviciute, A. & Fornito, A. Overcoming false-positive 
    gene-category enrichment in the analysis of spatially resolved transcriptomic 
    brain atlas data. Nat. Commun. 12, 2669 (2021).
    """
    
    # Extract neccessary information
    n_perm = weight_perm.shape[1]
    weight_perm_vals = weight_perm.values
    sign_perm_vals = sign_perm.values
    cond_pos = sign_perm_vals > 0
    cond_neg = sign_perm_vals < 0
    gene_idx = weight_perm.index
    
    # Help function for parallel computation
    def select_ctrl_p(p):
        weight_p = weight_perm_vals[:, p]
        sign_p = sign_perm_vals[:, p]
        # Substitute the pth column with non-permutated weight
        weight_perm_vals_p = weight_perm_vals.copy()
        weight_perm_vals_p[:, p] = gene_rep[weight_key].values
        ge_mtx = weight_perm_vals_p > weight_p[:, None]
        # Compute p-values
        pos_mask = sign_p > 0
        neg_mask = sign_p < 0
        counts = np.zeros(weight_p.shape[0], dtype=int)
        if pos_mask.any():
            counts[pos_mask] = np.sum(ge_mtx[pos_mask,:]&cond_pos[pos_mask,:],axis=1)
        if neg_mask.any():
            counts[neg_mask] = np.sum(ge_mtx[neg_mask,:]&cond_neg[neg_mask,:],axis=1)
        p_perm_p = counts/float(n_perm)

        # Construct data frame for convenience
        gene_rep_p = pd.DataFrame({
            'Sign': sign_p,
            weight_key: weight_p,
            'p_perm_p': p_perm_p
        }, index=gene_idx)

        # Filter and rank genes
        gene_rep_p = gene_rep_p[gene_rep_p['Sign']>0] if direction else gene_rep_p[gene_rep_p['Sign']<0]
        gene_rep_p = gene_rep_p.sort_values(by=['p_perm_p', weight_key], ascending=[True, False])
        
        ctrl_list = []
        for qbin in qbin_counts.keys():
            qbin_genes = list(gene_stats[gene_stats['qbin']==qbin].index)
            gene_rep_p_ = gene_rep_p[gene_rep_p.index.isin(qbin_genes)]
            qbin_ctrl = list(gene_rep_p_.head(qbin_counts[qbin]).index)
            ctrl_list = ctrl_list + qbin_ctrl        
        ctrl_weight = list(gene_rep_p.loc[ctrl_list][weight_key])
        return p, ctrl_list, ctrl_weight

    # Parallel computation
    results = Parallel(n_jobs=n_jobs)(
        delayed(select_ctrl_p)(p) for p in range(n_perm)
    )    
    dic_ctrl_list = {}
    dic_ctrl_weight = {}
    for p, genes, weights in results:
        dic_ctrl_list[p] = genes
        dic_ctrl_weight[p] = weights
    return dic_ctrl_list, dic_ctrl_weight

def _compute_raw_score(
    adata: ad.AnnData, 
    gene_list: list, 
    gene_weight: list, 
    weight_opt: str
) -> Tuple[np.ndarray, np.ndarray]:
    
    """
    Compute raw enrichment scores. And this function is totally adopted from scDRS.

    Parameters
    ----------
    adata : ad.AnnData
        The AnnData object containing expression data.
        - `adata.X` should have underwent gene specificity score normalization.
        - `adata.uns['GENE_STATS']` should be present.
    gene_list : list
        SBP-relevant gene list.
    gene_weight : list
        Gene weights for genes in the gene_list.
    weight_opt : str
        Gene statistics for re-weighting genes, must be present in `adata.uns
        ['GENE_STATS']`. In our study, we use differential stability to down-weight 
        genes with instable spatial expression pattern.

    Returns
    -------
    v_raw_score : np.ndarray
        Raw scores of shape (n_cell,).
    v_score_weight : np.ndarray
        Gene weights of shape (gene_size,).

    References
    ----------
    Zhang, M. J. et al. Polygenic enrichment distinguishes disease associations of 
    individual cells in single-cell RNA-seq data. Nat. Genet. 54, 1572-1580 (2022).
    """
    
    # Re-weight and normalize gene weights
    gene_stats = adata.uns["GENE_STATS"]
    v_score_weight = gene_stats.loc[gene_list, weight_opt].values
    v_score_weight = v_score_weight * np.array(gene_weight)
    v_score_weight = v_score_weight/v_score_weight.sum()

    # Compute raw enrichment scores
    gene_idx = adata.var_names.get_indexer(gene_list)
    gene_idx = gene_idx[gene_idx >= 0]
    v_raw_score = adata.X[:,gene_idx].dot(v_score_weight).reshape([-1])
    return v_raw_score, v_score_weight

def _correct_background(
    v_raw_score: np.ndarray, 
    mat_ctrl_raw_score: np.ndarray, 
    v_var_ratio_c2t: np.ndarray
)-> Tuple[np.ndarray, np.ndarray]:
    
    """
    Cell-wise and gene-wise background correction. And this function is
    totally adopted from scDRS.

    Parameters
    ----
    v_raw_score : np.ndarray
        Raw enrichment scores of shape (n_cell,).
    mat_ctrl_raw_score : np.ndarray
        Raw control enrichment scores of shape (n_cell,n_perm).
    v_var_ratio_c2t : np.ndarray
        Ratio of independent variance between control scores and SBP-relevant score,
        of shape (n_perm).

    Returns
    -------
    v_norm_score : np.ndarray
        Normalized disease score of shape (n_cell,)
    mat_ctrl_norm_score : np.ndarray
        Normalized control scores of shape (n_cell,n_ctrl).
        
    References
    ----------
    Zhang, M. J. et al. Polygenic enrichment distinguishes disease associations of 
    individual cells in single-cell RNA-seq data. Nat. Genet. 54, 1572-1580 (2022).
    """

    # Zero-values are assigned the smallest values at the end
    ind_zero_score = v_raw_score == 0
    ind_zero_ctrl_score = mat_ctrl_raw_score == 0

    # First gene set alignment: mean 0 and same independent variance
    v_raw_score = v_raw_score - v_raw_score.mean()
    mat_ctrl_raw_score = mat_ctrl_raw_score - mat_ctrl_raw_score.mean(axis=0)
    mat_ctrl_raw_score = mat_ctrl_raw_score / np.sqrt(v_var_ratio_c2t)

    # Cell-wise standardization
    v_mean = mat_ctrl_raw_score.mean(axis=1)
    v_std = mat_ctrl_raw_score.std(axis=1)
    v_norm_score = v_raw_score.copy()
    v_norm_score = (v_norm_score - v_mean) / v_std
    mat_ctrl_norm_score = ((mat_ctrl_raw_score.T - v_mean) / v_std).T

    # Second gene set alignment: mean 0
    v_norm_score = v_norm_score - v_norm_score.mean()
    mat_ctrl_norm_score = mat_ctrl_norm_score - mat_ctrl_norm_score.mean(axis=0)

    # Set cells with raw_score=0 to the minimum norm_score value
    norm_score_min = min(v_norm_score.min(), mat_ctrl_norm_score.min())
    v_norm_score[ind_zero_score] = norm_score_min - 1e-3
    mat_ctrl_norm_score[ind_zero_ctrl_score] = norm_score_min
    return v_norm_score, mat_ctrl_norm_score

def _get_p_from_empi_null(
    v_t: np.ndarray, 
    v_t_null: np.ndarray
) -> np.ndarray:
    """
    Compute p-value from empirical null.

    Parameters
    ----
    v_t : np.ndarray
        Observed score of shape (M,).
    v_t_null : np.ndarray
        Null scores of shape (N,).

    Returns
    -------
    v_p: : np.ndarray
        P-value for each element in v_t of shape (M,).
        
    References
    ----------
    Zhang, M. J. et al. Polygenic enrichment distinguishes disease associations of 
    individual cells in single-cell RNA-seq data. Nat. Genet. 54, 1572-1580 (2022).
    """

    v_t = np.array(v_t)
    v_t_null = np.array(v_t_null)

    v_t_null = np.sort(v_t_null)
    v_pos = np.searchsorted(v_t_null, v_t, side="left")
    v_p = (v_t_null.shape[0] - v_pos + 1) / (v_t_null.shape[0] + 1)
    return v_p
    
def score_cell_zoom(
    adata: ad.AnnData,
    gene_rep: pd.DataFrame,
    weight_key: str,
    sign_key: str,
    p_perm_key: str,
    weight_perm: pd.DataFrame,
    sign_perm: pd.DataFrame,
    direction: bool,
    gene_size: int,
    ctrl_match_key: str,
    weight_opt: str,
    n_genebin: int,
    return_ctrl_raw_score: bool,
    return_ctrl_norm_score: bool,
    n_jobs: int
) -> pd.DataFrame:
    
    """
    Calculates single-cell SBP-relevant enrichment scores.

    Parameters
    ----------
    adata : ad.AnnData
        The AnnData object containing expression data.
        - `adata.X` should have underwent gene specificity score normalization.
        - `adata.uns['GENE_STATS']` should be present.
    gene_rep : pd.DataFrame
        Results for PLS-R, must contain following columns:
        - weight_key: Column name indicating the original gene weights for scoring.
        - sign_key: Column name indicating the direction of genes.
        - p_perm_key: Column name indicating the permutation p-values.
    weight_perm : pd.DataFrame
        Gene weights in permutation tests.
    sign_perm : pd.DataFrame
        Signed gene weights in permutation tests.
    direction : bool
        If True, find gene set relevant to the positive direction
        of given SBP else negative direction.
    gene_size : int
        Number of genes used for scoring.
    ctrl_match_key : str
        Gene-level statistic used for matching control and SBP-relevant genes,
        must be present in `adata.uns['GENE_STATS']`.
    weight_opt : str
        Gene-level statistic used for re-weighting SBP-relevant genes,
        must be present in `adata.uns['GENE_STATS']`.
    n_genebin : int
        Number of bins for dividing genes by `ctrl_match_key`.
    return_ctrl_raw_score : bool
        If True, return raw scores for control gene sets.
    return_ctrl_norm_score : bool
        If True, return normalized scores for control gene sets.
    n_jobs : int
        Number of cores used for parallel computation.
    
    Returns
    -------
    df_res : pd.DataFrame
        Results of single-cell SBP-relevant enrichment scores and other statistics.
        - raw_socre: Raw enrichment scores.
        - norm_score: Normalized enrichment scores.
        - p_perm: Single-cell level p-values estimated on spatial permutation tests of the same cell.
        - pval: Pooled p-value estimated on all permutation tests, all cells
        
    References
    ----------
    .. [1] Zhang, M. J. et al. Polygenic enrichment distinguishes disease associations of 
        individual cells in single-cell RNA-seq data. Nat. Genet. 54, 1572-1580 (2022).
    .. [2] Fulcher, B. D., Arnatkeviciute, A. & Fornito, A. Overcoming false-positive 
        gene-category enrichment in the analysis of spatially resolved transcriptomic 
        brain atlas data. Nat. Commun. 12, 2669 (2021).
    """
    
    n_cell, n_gene = adata.shape
    # Select genes most relevant to SBP of interest
    gene_rep_ = gene_rep.copy()
    gene_rep_ = gene_rep_.sort_values(by=[p_perm_key,weight_key], ascending=[True, False])
    gene_rep_ = gene_rep_[gene_rep_[sign_key]>0] if direction else gene_rep_[gene_rep_[sign_key]<0]
    gene_rep_ = gene_rep_.head(gene_size)
    gene_list = list(gene_rep_.index)
    gene_weight = gene_rep_[weight_key].to_numpy()

    # Check preprocessing information
    assert (
        "GENE_STATS" in adata.uns
    ), "adata.uns['GENE_STATS'] not found, run `preprocess` first"

    gene_stats_set_expect = {"DS", "gss_max"}
    gene_stats_set = set(adata.uns["GENE_STATS"])    
    assert (
        len(gene_stats_set_expect - gene_stats_set) == 0
    ), "One of 'DS', 'gss_max' not found in adata.uns['GENE_STATS'], run `preprocess` and `compute_gss` first"
    
    # Check if ctrl_match_key and weight_opt is in GENE_STATS
    assert ctrl_match_key in adata.uns["GENE_STATS"], (
        "ctrl_match_key=%s not found in adata.uns['GENE_STATS']"
        % ctrl_match_key
    )
    assert weight_opt in adata.uns["GENE_STATS"], (
        "ctrl_match_key=%s not found in adata.uns['GENE_STATS']"
        % weight_opt
    )
    
    # Divide genes into bins according to `ctrl_match_key`
    gene_stats = adata.uns["GENE_STATS"].copy()
    gene_stats["qbin"] = pd.qcut(
        gene_stats[ctrl_match_key], q=n_genebin, labels=False, duplicates="drop"
    )    
    gene_stats_SBP = gene_stats.loc[gene_list]
    qbin_counts = dict(gene_stats_SBP["qbin"].value_counts())
    
    if gene_weight is not None:
        gene_weight = list(gene_weight)
    else:
        gene_weight = [1]*len(gene_list)

    # Select control gene sets
    dic_ctrl_list, dic_ctrl_weight = select_ctrl(
        weight_perm, sign_perm, gene_rep, weight_key,
        gene_stats, qbin_counts, direction, n_jobs
    )

    # Compute raw scores
    v_raw_score, v_score_weight = _compute_raw_score(
        adata, gene_list, gene_weight, weight_opt
    )
    
    # Compute control scores with parallel computation
    n_perm = weight_perm.shape[1]
    mat_ctrl_raw_score = np.zeros([n_cell, n_perm])
    mat_ctrl_weight = np.zeros([len(gene_list), n_perm])    
    results = Parallel(n_jobs=n_jobs)(
        delayed(_compute_raw_score)(adata, dic_ctrl_list[p], dic_ctrl_weight[p], weight_opt) 
        for p in tqdm(range(n_perm),desc="Computing control scores")
    )
    # Extract results
    for p, (v_ctrl_raw_score, v_ctrl_weight) in enumerate(results):
        mat_ctrl_raw_score[:, p] = v_ctrl_raw_score
        mat_ctrl_weight[:, p] = v_ctrl_weight

    # Compute ratio of independent variance
    v_var_ratio_c2t = np.ones(n_perm)
    for p in range(n_perm):
        v_var_ratio_c2t[p] = (mat_ctrl_weight[:,p]**2).sum()
    v_var_ratio_c2t /= (v_score_weight**2).sum()
    # Compute normalized scores
    v_norm_score, mat_ctrl_norm_score = _correct_background(
        v_raw_score,
        mat_ctrl_raw_score,
        v_var_ratio_c2t,
    )

    # Get p-values
    p_perm = (1+(mat_ctrl_norm_score.T>=v_norm_score).sum(axis=0))/(1+n_perm)
    pooled_p = _get_p_from_empi_null(v_norm_score,mat_ctrl_norm_score.flatten())

    # Return result
    dic_res = {
        "raw_score": v_raw_score,
        "norm_score": v_norm_score,
        "p_perm": p_perm,
        "pval": pooled_p
    }
    if return_ctrl_raw_score:
        for i in range(n_perm):
            dic_res["ctrl_raw_score_%d" % i] = mat_ctrl_raw_score[:,i]
    if return_ctrl_norm_score:
        for i in range(n_perm):
            dic_res["ctrl_norm_score_%d" % i] = mat_ctrl_norm_score[:,i]
    df_res = pd.DataFrame(index=adata.obs.index, data=dic_res, dtype=np.float32)
    return df_res

def group_bh(
    adata: ad.AnnData,
    df_res: pd.DataFrame, 
    pval: str, 
    group: str, 
    alpha: float
) -> pd.DataFrame:
    
    """
    Perform group Benjamini–Hochberg FDR correction.

    Parameters
    ----------
    adata : ad.AnnData
        The AnnData object containing expression data.
    df_res : pd.DataFrame
        Results of single-cell SBP-relevant enrichment scores and other statistics.
    pval : str
        Column name indicating the cell-level p-values, must be present in `df_res`.
    group : str
        Column name indicating the cell groups, based on which p-values are adjusted, 
        must be present in `adata.obs`.
    alpha : float
        Significance level for group Benjamini–Hochberg FDR correction.

    Returns
    -------
    df_res: pd.DataFrame
        Results with adjusted p-values, stored as `p_adj`.
        
    References
    ----------
    Hu, J. X., Zhao, H. & Zhou, H. H. False discovery rate control with groups. 
    J. Am. Stat. Assoc. 105, 1215-1227 (2010).
    """
    
    # Initialization
    N = df_res.shape[0]
    df_res[group] = adata.obs[group]
    pi_0 = sum(df_res[pval]>alpha)/N
    # Get cell counts for each group
    group_counts = dict(df_res[group].value_counts())
    group_counts_sig = dict(df_res[df_res[pval]<alpha][group].value_counts())    
    
    # Compute weighted p-values for each group
    result_pw = pd.Series(index=df_res.index, dtype=float, name='p_w')
    for g, df_grp in df_res.groupby(group, observed=False):
        g_count = group_counts[g]
        g_count_sig = group_counts_sig[g]
        if g_count_sig==0:
            w_g = 10000
        else:
            w_g = (g_count - g_count_sig)/g_count_sig        
        p_wg = (df_grp[pval].values)*w_g
        result_pw.loc[df_grp.index] = p_wg
    
    # Group Benjamini–Hochberg FDR correction
    df_res['p_w'] = result_pw
    df_res = df_res.sort_values(by='p_w')
    df_res['p_rank'] = range(1,N+1)
    df_res['p_adj'] = (df_res['p_w']*N*(1-pi_0))/df_res['p_rank']
    df_res["p_adj"] = df_res["p_adj"][::-1].cummin()[::-1]
    df_res[f'p_fdr{alpha}'] = np.where(df_res['p_adj']<alpha, 'True', 'False')
    df_res = df_res.reindex(adata.obs.index)
    return df_res
    
def downstream_DEG(
    adata: ad.AnnData,
    df_res: pd.DataFrame,
    alpha: float,
    min_score: float,
    group: str,
    rank_method: str,
    max_iter: int
) -> ad.AnnData:
    
    """
    Calculates differentially expressed genes (DEG) for cell groups 
    containing cells significantly relevant to SBP.

    Parameters
    ----------
    adata : ad.AnnData
        The AnnData object containing expression data.
        - `adata.X` should have underwent gene specificity score normalization.
    df_res : pd.DataFrame
        Results of single-cell SBP-relevant enrichment scores and other statistics.
    alpha : float
        Significance level indicating significant cell subpopulations.
    min_score : float
        Minimum enrichment score for significant cell subpopulations.
    group : str
        Column name indicating the cell groups, based on which significant cells and
        non-significant cells are compared, must be present in `adata.obs`.
    rank_method : {'t-test','t-test_overestim_var','wilcoxon','logreg'}
        Methods how scanpy calculate differentially expressed genes.
    max_iter : int
        Maximum iteration for DEG calculation, only used for `logreg`.
    
    Returns
    -------
    adata : ad.AnnData
        The AnnData object with `adata.uns['DEG']`, where the DEG results for each
        cell group are stored.
    """
    
    # If no significant cell is found, skip this function
    if sum(df_res[f'p_fdr{alpha}']=='True')==0:
        print("No significant cell is found.")
    else:
        adata.uns['DEG'] = {}
        # Find pre-defined cell groups containing significant cells
        adata.obs['norm_score'] = df_res['norm_score']
        adata.obs[f'p_fdr{alpha}'] = df_res[f'p_fdr{alpha}']
        adata.obs["Sig"] = np.where(
            (adata.obs["norm_score"]>min_score)&(adata.obs[f"p_fdr{alpha}"]=="True"),
            "Significant", "Non-significant"
        )
        sig_groups = list(adata.obs[adata.obs['Sig']=='Significant'][group].unique())
        # Compute differentially expressed genes for each selected group
        for g in sig_groups:
            adata_sub = adata[adata.obs[group]==g].copy()
            if sum(adata_sub.obs["Sig"]=="Significant") < 2:
                continue
            if rank_method == 'logreg':
                sc.tl.rank_genes_groups(
                    adata_sub, 
                    groupby='Sig', 
                    method=rank_method, 
                    max_iter=max_iter
                )
            else:
                sc.tl.rank_genes_groups(
                    adata_sub, 
                    groupby='Sig', 
                    method=rank_method
                )
            gene_rnk = adata_sub.uns["rank_genes_groups"]["names"]
            gene_rnk = np.array([t[0] for t in gene_rnk])
            gene_rnk = pd.DataFrame(adata_sub.uns["rank_genes_groups"]["scores"], index=gene_rnk)
            adata.uns['DEG'][g] = gene_rnk            
    return adata

def downstream_region_enrich(
    adata: ad.AnnData,
    df_res: pd.DataFrame,
    alpha: float,
    min_score: float,
    group: int,
    region_col: str,
    batch_col: str,
    dataset: list,
    indvd_col: Optional[str]=None,
) -> ad.AnnData:
    
    """
    Computes region enrichment for cell groups containing significant cells based on
    predefined statistical results.

    Parameters
    ----------
    adata : ad.AnnData
        The AnnData object containing expression data.
        - `adata.X` should have underwent gene specificity score normalization.
    df_res : pd.DataFrame
        Results of single-cell SBP-relevant enrichment scores and other statistics.
    alpha : float
        Significance level indicating significant cell subpopulations.
    min_score : float
        Minimum enrichment score for significant cell subpopulations.
    group : str
        Column name indicating the cell groups, based on which significant cells and
        non-significant cells are compared, must be present in `adata.obs`.
    region_col : str
        Column name indicating the brain region the cell came from. must be present 
        in `adata.obs`.
    batch_col : str
        Column name indicating the biological batch the cell came from. must be 
        present in `adata.obs`.
    dataset : list
        List of subset of dataset to perform region enrichment analysis on.
    indvd_col : {None, str}, optional
        Column name indicating biological replicates, must be present in `adata.obs`. 
        If provided, region enrichment score will be presented as mean and standard 
        error across biological replicates.
    
    Returns
    -------
    adata : ad.AnnData
        The AnnData object with `adata.uns['Region Enrichment']`, where the region 
        enrichment analysis results for each cell group are stored.
    
    References
    ----------
    Yang, L. et al. Projection-TAGs enable multiplex projection tracing and 
    multi-modal profiling of projection neurons. Nat. Commun. 16, 5557 (2025).
    """
    
    # If no significant cell is found, skip this function
    if sum(df_res[f'p_fdr{alpha}']=='True')==0:
        print("No significant cell is found.")
    else:
        adata.uns['Region Enrichment'] = {}
        # Find pre-defined cell groups containing significant cells
        adata.obs['norm_score'] = df_res['norm_score']
        adata.obs[f'p_fdr{alpha}'] = df_res[f'p_fdr{alpha}']
        adata.obs["Sig"] = np.where(
            (adata.obs["norm_score"]>min_score)&(adata.obs[f"p_fdr{alpha}"]=="True"),
            "Significant", "Non-significant"
        )
        sig_groups = list(adata.obs[adata.obs['Sig']=='Significant'][group].unique())
        # Compute region enrichment for each selected group
        for g in sig_groups:
            region_enrich_g = {}
            df_g = adata.obs[(adata.obs[group]==g) &
                             (adata.obs[batch_col].isin(dataset))]
            if indvd_col is None:
                counts_all = dict(df_g[region_col].value_counts())
                prop_all = {k:v/sum(counts_all.values()) for k,v in counts_all.items()}
                counts_sig = dict(df_g[df_g['Sig']=='Significant'][region_col].value_counts())
                prop_sig = {k:v/sum(counts_sig.values()) for k,v in counts_sig.items()}
            
                for r in counts_all.keys():
                    if prop_all[r] != 0:
                        enrich = prop_sig[r]/prop_all[r]
                        region_enrich_g[r] = enrich
                region_enrich_g = pd.DataFrame(
                    region_enrich_g.values(),
                    region_enrich_g.keys(),
                    columns=['Region Enrichment']
                )
                
            else:
                region_enrich_g = {}
                indvds = list(df_g[indvd_col].unique())
                # Compute region enrichment for all individuals, respectively
                for i in indvds:
                    region_enrich_g_i = {}
                    df_g_i = df_g[df_g[indvd_col]==i]
                    counts_all = dict(df_g_i[region_col].value_counts())
                    prop_all = {k:v/sum(counts_all.values()) for k,v in counts_all.items()}
                    counts_sig = dict(df_g_i[df_g_i['Sig']=='Significant'][region_col].value_counts())
                    prop_sig = {k:v/sum(counts_sig.values()) for k,v in counts_sig.items()}
                    for r in counts_all.keys():
                        if prop_all[r] != 0:
                            enrich = prop_sig[r]/prop_all[r]
                            region_enrich_g_i[r] = enrich
                    region_enrich_g_i = pd.DataFrame(
                        region_enrich_g_i.values(),
                        region_enrich_g_i.keys(),
                        columns=['Region Enrichment']
                    )
                    region_enrich_g[i] = region_enrich_g_i
                # Concatenate results from different individuals
                common_idx = set.intersection(*[set(df.index) for df in region_enrich_g.values()])
                common_idx = sorted(common_idx)
                dfs = [df.loc[common_idx] for df in region_enrich_g.values()]
                region_enrich_g = pd.concat(dfs, axis=1)
                # Compute mean and standard error across individuals
                region_enrich_g["Mean"] = region_enrich_g.mean(axis=1)
                region_enrich_g["Std"] = region_enrich_g.std(axis=1)
                n = region_enrich_g.index[region_enrich_g.eq("Mean").any(axis=1)][0]
                region_enrich_g.iloc[:n, :].columns = [f"Indvd{i+1}" for i in range(n)]
                
                # Apply one-sample T-test
                def row_ttest(row):
                    values = row[[f"Indvd{i+1}" for i in range(n)]].values
                    values = np.log(values) + 1e-6
                    t_stat, p_two_tailed = ttest_1samp(values, popmean=0)
                    if t_stat > 0:
                        p_one_tailed = p_two_tailed/2
                    else:
                        p_one_tailed = 1 - p_two_tailed/2
                    return p_one_tailed
                region_enrich_g["p-value"] = region_enrich_g.apply(row_ttest, axis=1)
                _, region_enrich_g["p-fdr"], _, _ = multipletests(region_enrich_g["p-value"], method="fdr_bh")
                region_enrich_g = region_enrich_g.reindex(order)
            adata.uns['Region Enrichment'][g] = region_enrich_g
    return adata
            
def run_hdWGCNA_py(
    R: str,
    pwd: os.PathLike,
    seurat_rds: os.PathLike,
    celltype_col: str,
    batch_col: str,
    reduction: str,
    default: bool,
    PARAMS: Optional[Dict[str, Any]]=None
) -> None:
    
    """
    Run hdWGCNA and module preservation pipeline.

    Parameters
    ----------
    R : str
        R version, load as `module load {R}` on terminal.
    pwd : os.PathLike
        File path containing the Seurat object of scRNA-seq data.
    seurat_rds : os.PathLike
        File name of the Seurat object of scRNA-seq data, must end with `.rds`.
    celltype_col : str
        Pre-defined cell type column in seurat_obj@meta.data.
    batch_col : str
        Pre-defined batch effect column in seurat_obj@meta.data.
    reduction : str
        The dimensionality reduction to perform KNN on.
    default : bool
        If True, run this pipeline with default setting, `PARAMS` must be
        provided else.
    PARAMS : dict
        Specify customed arguments, must contain following arguments:
        - k: Number of nearest neighbors to aggregate (default=50).
        - max_shared: The maximum number of cells to be shared across two meta-cells (default=15).
        - min_cells: The minimum number of cells in a particular group to construct meta-cells (default=100).
        - min_metacell: Minimum number of meta-cells a group must include (default=100).
        - deepSplit: The sensitivity of the dynamic tree cut algorithm in hierarchical clustering (default=4).
        - pamStage: Whether the Partitioning Around Medoids (PAM) refinement step is applied (default=FALSE).
        - detectCutHeight: The cut height threshold used during the module detection (default=0.995).
        - minModuleSize: The minimum number of genes required for a cluster to be considered a module (default=50).
        - mergeCutHeight: The threshold for merging modules based on the dissimilarity of their eigengenes (default=0.2).
        - n_permutations: Number of permutations for the module preservation test (default=500).
        - seed: Random seed for reproducibility (default=0).

    Returns
    -------
    None. Results files will be saved during running process.
        
    References
    ----------
    [1] Morabito, S., Reese, F., Rahimzadeh, N., Miyoshi, E. & Swarup, V. hdWGCNA 
        identifies co-expression networks in high-dimensional transcriptomics data. 
        Cell Rep. Methods 3, 100498 (2023).
    [2] Langfelder, P., Luo, R., Oldham, M. C. & Horvath, S. Is my network module 
        preserved and reproducible? PloS Comput. Biol. 7, e1001057 (2011).
        
    Notes
    -----
    To enable users to perform all operations exclusively through the Python interface 
    when conditions permit, we introduce this function. We also acknowledge that running 
    R code from Python provides no practical advantage; if further parameter configuration 
    is required, we recommend using the R code for this part directly.
    """
    
    # Fetch R code files (development stage)
    R_pwd = "/slurm/home/yrd/liaolab/nieshuyang/ZOOM/R"
    Metacell_R = f"{R_pwd}/Metacell.R"
    run_hdWGCNA_R = f"{R_pwd}/run_hdWGCNA.R"
    ModulePreservation_R = f"{R_pwd}/ModulePreservation.R"
    
    # Start up a shell to run command lines
    shell = subprocess.Popen(["/bin/bash"], stdin=subprocess.PIPE, text=True)
    
    # Run hdWGCNA pipeline and module preservation statistics on R with default setting.
    if default:
        shell.stdin.write(f"""
        module load {R}
        Rscript {Metacell_R} {pwd} {seurat_rds} {celltype_col} {batch_col} {reduction} \n
        Rscript {run_hdWGCNA_R} {pwd} {seurat_rds} {celltype_col} {batch_col} \n
        Rscript {ModulePreservation_R} {pwd} {seurat_rds} {celltype_col} {batch_col} \n
        """)
        shell.stdin.flush()
    else:
        k = PARAMS['k']
        max_shared = PARAMS['max_shared']
        min_cells = PARAMS['min_cells']
        min_metacell = PARAMS['min_metacell']
        deepSplit = PARAMS['deepSplit']
        pamStage = PARAMS['pamStage']
        detectCutHeight = PARAMS['detectCutHeight']
        minModuleSize = PARAMS['minModuleSize']
        mergeCutHeight = PARAMS['mergeCutHeight']
        n_permutations = PARAMS['n_permutations']
        seed = PARAMS['seed']
        
        shell.stdin.write(f"""
        module load {R}
        Rscript {Metacell_R} {pwd} {seurat_rds} {celltype_col} {batch_col} {reduction} {k} {max_shared} {min_cells} {seed} \n
        Rscript {run_hdWGCNA_R} {pwd} {seurat_rds} {celltype_col} {batch_col} {min_metacell} {deepSplit} {pamStage} {detectCutHeight} {minModuleSize} {mergeCutHeight} {seed} \n
        Rscript {ModulePreservation_R} {pwd} {seurat_rds} {celltype_col} {batch_col} {n_permutations} {seed} \n
        """)
        shell.stdin.flush()
    
    # Run the shell
    shell.stdin.close()
    shell.wait()
    return None

def gsea_perm(
    gene_rep: pd.DataFrame,
    weight_key: str,
    sign_key: str,
    weight_perm: pd.DataFrame,
    sign_perm: pd.DataFrame,
    gene_sets: dict,
    min_size: int,
    max_size: int,
    one_sided: bool,
    n_jobs: int
) -> pd.DataFrame:
    
    """
    Permform GSEA based on spatial permutation test.

    Parameters
    ----------
    gene_rep : pd.DataFrame
        Results for PLS-R, must contain following columns:
        - weight_key: Column name indicating the original gene weights for scoring.
        - sign_key: Column name indicating the direction of genes.
    weight_perm : pd.DataFrame
        Gene weights in permutation tests.
    sign_perm : pd.DataFrame
        Signed gene weights in permutation tests.
    direction : bool
        If True, find gene set relevant to the positive direction
        of given SBP else negative direction.
    gene_sets : dict
        Gene sets for enrichment analysis, must be organized as {'Term1': [Gene1, Gene2,...],...}
    min_size & max_size : int,
        Minimum and maximum size of target gene set to be included in GSEA analysis
    one_sided: bool:
        If True, infer statistical significance via one-sided p-values. Else, use two-sided p-values.
    n_jobs : int
        Number of cores used for parallel computation.

    Returns
    -------
    gsea_res : pd.DataFrame
        Results of spatial permutation test based GSEA.
        - index: Gene terms in `gene_sets`.
        - ES: Raw GSEA enrichment scores.
        - NES: Normalized GSEA enrichment scores.
        - p_perm: P-values inferred from spatial permutation test.
        
    References
    ----------
    [1] Fulcher, B. D., Arnatkeviciute, A. & Fornito, A. Overcoming false-positive 
        gene-category enrichment in the analysis of spatially resolved transcriptomic 
        brain atlas data. Nat. Commun. 12, 2669 (2021).
    [2] Martins, D. et al. Imaging transcriptomics: convergent cellular, 
        transcriptomic, and molecular neuroimaging signatures in the healthy adult human 
        brain. Cell Rep. 37, 110173 (2021).
    [3] Fang, Z., Liu, X. & Peltz, G. GSEApy: a comprehensive package for performing 
        gene set enrichment analysis in Python. Bioinformatics 39, btac757 (2023).
    
    Notes
    -----
    Though we developed this function to link SBP-derived gene signatures to cell type-
    specific gene co-expression modules, this function can be equally applied to other
    biologically meaningful gene sets like GO, KEGG pathways, and so on.
    """
    
    # Rank genes by `Weight` for GSEA
    df = pd.DataFrame({'Sign': gene_rep[sign_key], 
                       'Weight': gene_rep[weight_key]}, 
                       index=gene_rep.index)
    df.loc[df['Sign']<0, 'Weight'] = df.loc[df['Sign']<0, 'Weight']*-1
    df = df.sort_values(by = 'Weight', ascending=False)
    df_rnk = pd.DataFrame(df['Weight'])
    
    # GSEA analysis on original gene ranking
    gsea_res = gp.prerank(
        rnk=df_rnk, gene_sets=gene_sets,
        threads=-1, min_size=min_size, 
        max_size=max_size, permutation_num=0, 
        outdir=None, verbose=False
    )
    # Dataframe format of results
    gsea_res = gsea_res.res2d
    gsea_res = gsea_res.set_index('Term')
    
    # Function for parallel permutation test
    def gsea_perm_p(p, sign_perm, weight_perm, 
                    gene_sets, min_size, max_size):
        df_p = pd.DataFrame({'Sign': sign_perm.iloc[:,p], 
                             'Weight': weight_perm.iloc[:,p]}, 
                             index=gene_rep.index)
        df_p.loc[df_p['Sign']<0, 'Weight'] = df_p.loc[df_p['Sign']<0, 'Weight']*-1
        df_p = df_p.sort_values(ascending=False, by = 'Weight')
        df_rnk_p = pd.DataFrame(df_p['Weight'])
        gsea_res_p = gp.prerank(
            rnk=df_rnk_p, gene_sets=gene_sets,
            threads=-1, min_size=min_size, 
            max_size=max_size, permutation_num=0, 
            outdir=None, verbose=False
        )
        gsea_res_p = gsea_res_p.res2d
        gsea_res_p = gsea_res_p.set_index('Term')
        gsea_res_p = gsea_res_p.reindex(gsea_res.index)
        return gsea_res_p['ES'].values
    
    # Parallel computation
    n_perm = weight_perm.shape[1]
    results = Parallel(n_jobs=n_jobs)(
        delayed(gsea_perm_p)(p, sign_perm, weight_perm, gene_sets, min_size, max_size)
        for p in range(n_perm)
    )
    GSEA_perm = pd.concat([pd.Series(res) for res in results], axis=1)
    GSEA_perm.index = gsea_res.index
    
    # Compute normalized enrichment score and p-value
    for term in gsea_res.index:
        ES = gsea_res.loc[term]['ES']
        ES_perm = GSEA_perm.loc[term]
        if ES > 0:
            if one_sided:
                count = sum(ES_perm>ES)
            else:
                count = sum(abs(ES_perm)>abs(ES))
            NES = ES/np.mean(ES_perm[ES_perm>0])
        elif ES < 0:
            if one_sided:
                count = sum(ES_perm<ES)
            else:
                count = sum(abs(ES_perm)>abs(ES))
            NES = -1*(ES/np.mean(ES_perm[ES_perm<0]))
        gsea_res.loc[term, 'NES'] = NES
        gsea_res.loc[term, 'p_perm'] = count/n_perm
    gsea_res = gsea_res.sort_values(by='p_perm')
    return gsea_res
