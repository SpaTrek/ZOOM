# -*- coding: utf-8 -*-
"""
Main class used for quick analyses in ``zoom``.

- ZOOM: Implementation of cross-validation PLS-R and traditional imaging-transcriptomics 
  paradigm and spatial permutation test-based GSEA.

- ZOOM_SC: Calculation of single-cell enrichment scores of SBP-relevant gene sets and 
  single-cell level statistical significance.
"""

import numpy as np
import pandas as pd
import anndata as ad
import os
from statsmodels.stats.multitest import multipletests
from typing import Optional, Union, Dict, List
from zoom.data_loader import load_sc, load_df
import zoom.pls_tool as pls
import zoom.sc_tool as sct

class ZOOM:
    
    """
    The ZOOM class serves as fundation for this package.It implements a framework for 
    partial least squares regression (PLS-R) applied to anatomically comprehensive 
    transcriptomics and spatial brain phenotypic data. 

    Specifically, this class provides:
    
    - Initialization of input data (expression, SBP, and permutation-based SBP).
    - Configuration of PLS-R parameters such as component number, cross-validation 
      folds, random seed, and parallelization settings.
    - Methods for cross-validation of PLS-R models to determine optimal latent 
      components, assess prediction accuracy, and evaluate statistical significance 
      via spatial permutation tests.
    - Procedures for quantifying gene-level contributions to prediction performance 
      using multiple metrics (PLS1 weights, regression coefficients, and variable 
      importance in projection), with spatial permutation strategies to infer 
      statistical significance.
    - Relate SBP-relevant genes to pre-defined gene set, such as GO, KEGG pathways,
      gene co-expression modules, and so on.
    
    Attributes
    ----------
    expression : pd.DataFrame
        Gene expression matrix (e.g., AHBA data).
    SBP : pd.DataFrame
        Spatial brain phenotype data.
    SBP_perm : pd.DataFrame
        Permuted SBP data for null distribution estimation.
    best_comp : int (default is None)
        Optimal number of PLS-R components. If None, determined via cross-validation.
    __cv : int
        Number of cross-validation folds.
    __seed : int
        Random seed for reproducibility.
    __n_jobs : int
        Number of parallel jobs for computation.
    PLS_performance : dict
        Stores prediction outputs and correlation scores.
    PLS_r : float
        Median correlation between predicted and observed SBP.
    PLS_Q2 : float
        Cross-validated predictive accuracy statistic.
    PLS_p_perm : float
        Permutation-based p-value for prediction performance.
    PLS_report : pd.DataFrame
        Gene-level statistical report including weights, signs, and permutation p-values.
    weight_perm : pd.DataFrame
        Permuted gene weights for null distribution.
    sign_perm : pd.DataFrame
        Permuted gene signs for null distribution.
    gsea_res : pd.DataFrame or dict
        Results of spatial permutation test based GSEA.
        - index: Gene terms in `gene_sets`.
        - ES: Raw GSEA enrichment scores.
        - NES: Normalized GSEA enrichment scores.
        - p_perm: P-values inferred from spatial permutation test.
        If multiple gene sets are provided, return a dict of such data frames.

    Notes
    -----
    This class is designed for integrative neurogenomics, enabling rigorous evaluation 
    of gene–SBP associations while accounting for spatial autocorrelation. Its functionality 
    is also inherited by the following subclasses; however, if users intend only to perform 
    imaging-transcriptomics analysis without extending to single-cell transcriptomic datasets, 
    it is recommended to employ this class.
    
    References
    ----------
    .. [1] Wang, Y. et al. Spatio-molecular profiles shape the human 
        cerebellar hierarchy along the sensorimotor-association axis. 
        Cell Rep. 43, 113770 (2024).
    .. [2] Whitaker, K. J., Vértes, P. E., Romero-Garcia, R & Bullmore, E. T.
        Adolescence is associated with genomically patterned consolidation 
        of the hubs of the human brain connectome. Proc. Natl Acad. Sci. 
        USA 113, 9105-9110 (2016).
    .. [3] Mahieu, B., Qannari, E. M. & Jaillais, B. Extension and 
        significance testing of Variable Importance in Projection (VIP) 
        indices in Partial Least Squares regression and Principal 
        Components Analysis. Chemom. Intell. Lab. Syst. 242, 104986 (2023).
    .. [4] Fulcher, B. D., Arnatkeviciute, A. & Fornito, A. Overcoming false-
        positive gene-category enrichment in the analysis of spatially resolved 
        transcriptomic brain atlas data. Nat. Commun. 12, 2669 (2021)
    .. [5] Martins, D. et al. Imaging transcriptomics: convergent cellular, 
        transcriptomic, and molecular neuroimaging signatures in the healthy 
        adult human brain. Cell Rep. 37, 110173 (2021).
    """
    
    def __init__(
        self,
        expression: Union[os.PathLike, pd.DataFrame],
        SBP: Union[os.PathLike, pd.DataFrame],
        SBP_perm: Union[os.PathLike, pd.DataFrame],
        best_comp: Optional[Union[int, None]] = None,
        #gene_sets: Optional[Dict[str, Union[Dict[str, Union[str, List[str]]], List[str]]]] = None,
        cv: int = 10,
        seed: int = 123,
        n_jobs: int = -1
    ) -> None:
        
        """
        Initializer of class ZOOM.
        """
        
        # Load AHBA and SBP
        self.expression = load_df(expression)
        self.SBP = load_df(SBP)
        self.SBP_perm = load_df(SBP_perm)
        #self.gene_sets = gene_sets
        # Initialize PLS-R parameters
        self.best_comp = best_comp
        self.__cv = cv
        self.__seed = seed
        self.__n_jobs = n_jobs
        # Initialize PLS-R results
        self.PLS_performance = None
        self.PLS_r = None
        self.PLS_Q2 = None
        self.PLS_p_perm = None
        self.PLS_report = None
        self.weight_perm = None
        self.sign_perm = None
        #self.gsea_res = None
    
    def cv_PLSR(
        self,
        ncomps: Union[List[int], np.ndarray] = range(1, 16),
        repeats_cv: int = 30,
        repeats_pred: int = 101
    ) -> None:
        
        """
        Class method for the implementation of cross-validation (CV) partial
        least squares regression (PLS-R). This function undergoes 3 stages:
        - Evaluate optimal component number if it is not previously provided.
        - Evaluate PLS-R prediction performance under optimal parameter.
        - Infer statistical significance of prediction performance through 
          spatial permutation test.

        Parameters
        ----------
        ncomps: {List[int], np.ndarray}, optional
            Optimal component number candidates.
        repeats_cv : int
            How many times should optimal component evaluation be run?
        repeats_pred : int
            How many times should model performance evaluation be run?

        Returns
        -------
        None
            Results of this function are stored on `self.PLS_performance`,
            `self.PLS_r`, `self.PLS_Q2` and `self.PLS_p_perm`.
        
        References
        ----------
        Wang, Y. et al. Spatio-molecular profiles shape the human 
        cerebellar hierarchy along the sensorimotor-association axis. 
        Cell Rep. 43, 113770 (2024).
        """
        
        # Evaluate optimal component number if it is not provided
        if self.best_comp is None:
            self.best_comp = pls.run_component_eval(
                self.expression, self.SBP, ncomps,
                self.__cv, self.__seed, repeats_cv
            )
        
        # Evaluate PLS-R prediction performance under optimal parameter
        preds_rep, scores = pls.model_eval(
            self.expression, self.SBP, 
            self.best_comp, self.__cv, 
            repeats_pred, self.__seed, 
            self.__n_jobs
        )
        self.PLS_performance = {
            'Predictions': preds_rep, 
            'Correlations': scores
        }
        self.PLS_r = np.median(scores)
        self.PLS_Q2 = pls.get_Q2(self.SBP, preds_rep)
        
        # Infer statistical significance of prediction performance through spatial permutation test
        self.PLS_p_perm, _ = pls.pls_perm(
            self.expression, self.SBP, 
            self.SBP_perm, self.best_comp, 
            scores, self.__cv, 
            self.__seed, self.__n_jobs
        )
        return None
        
    def get_gene_contrib(
        self,
        metric: str = "VIP",
        n_boot: Optional[Union[int, None]] = 1000,
        one_sided: bool = "True"
    ) -> None:
        
        """
        Compute gene-level contribution to PLS-R prediction and infer gene-
        level statistical significance against spatial autocorrelation.

        Parameters
        ----------
        metric : str
            The statistical metric to be used. Must be one of {"PLS1", "RC", "VIP"}.
            - PLS1: PLS1 weights.
            - RC: Regression coefficient.
            - VIP: Variable importance in projection.
        n_boot : int or None, default=1000
            Number of bootstrap iterations to perform if the `metric` is `PLS1`.
        one_sided : bool
            If True, infer statistical significance via one-sided
            p-values. Else, use two-sided p-values.

        Returns
        -------
        None
            Results of this function are stored on `self.PLS_report`,
            `self.weight_perm` and `self.sign_perm`.
        
        References
        ----------
        [1] Whitaker, K. J., Vértes, P. E., Romero-Garcia, R & Bullmore, E. T.
            Adolescence is associated with genomically patterned consolidation 
            of the hubs of the human brain connectome. Proc. Natl Acad. Sci. 
            USA 113, 9105-9110 (2016).
        [2] Wang, Y. et al. Spatio-molecular profiles shape the human 
            cerebellar hierarchy along the sensorimotor-association axis. 
            Cell Rep. 43, 113770 (2024).
        [3] Mahieu, B., Qannari, E. M. & Jaillais, B. Extension and 
            significance testing of Variable Importance in Projection (VIP) 
            indices in Partial Least Squares regression and Principal 
            Components Analysis. Chemom. Intell. Lab. Syst. 242, 104986 (2023).
        """
        
        metric_cand = ["PLS1","RC","VIP"]
        if metric not in metric_cand:
            raise ValueError(
                f"The argument `metric` must be one of {metric_cand}. "
                f"However, the provided value `{metric}` is invalid. "
            )
        
        if metric == 'PLS1':
            # Calculate PLS1 weights through bootstrap strategy
            gene_rep, PLS1_perm = pls.pls1_perm(
                self.expression, self.SBP, 
                self.SBP_perm, self.best_comp,
                n_boot, one_sided, 
                self.__seed, self.__n_jobs
            )
            # Get report for each gene
            self.PLS_report = pd.DataFrame({
                "p_perm": gene_rep["p_perm"],
                "Weight": gene_rep["PLS1"].abs(),
                "Sign": np.where(gene_rep["PLS1"]>0,1,-1)
            }, index = gene_rep.index)
            self.weight_perm = PLS1_perm.abs()
            self.sign_perm = PLS1_perm.map(lambda x: 1 if x>0 else -1)
        
        # Calculate regression coefficient (RC) 
        # or variable importance in projection (VIP)
        else:
            gene_rep, RC_perm, VIP_perm = pls.vip_perm(
                self.expression, self.SBP, self.SBP_perm, 
                self.best_comp, one_sided, self.__n_jobs
            )
            # Get regression coefficient (RC)
            if metric == 'RC':
                self.PLS_report = pd.DataFrame({
                    "p_perm": gene_rep["p_perm_rc"],
                    "Weight": gene_rep["RC"].abs(),
                    "Sign": np.where(gene_rep["RC"]>0,1,-1)
                }, index = gene_rep.index)
                self.weight_perm = RC_perm.abs()
                self.sign_perm = RC_perm.map(lambda x: 1 if x>0 else -1)
            elif metric == 'VIP':
                self.PLS_report = pd.DataFrame({
                    "p_perm": gene_rep["p_perm_vip"],
                    "Weight": gene_rep["VIP"].abs(),
                    "Sign": np.where(gene_rep["RC"]>0,1,-1)
                }, index = gene_rep.index)
                self.weight_perm = VIP_perm.abs()
                self.sign_perm = RC_perm.map(lambda x: 1 if x>0 else -1)
        return None
    
    def GSEA(
        self,
        gene_sets: Dict[str, Union[Dict[str, List[str]], List[str]]],
        min_size: int = 30,
        max_size: int = 500,
        one_sided: bool = True
    ) -> None:
        
        """
        Implementation of spatial permutation test-based GSEA.

        Parameters
        ----------
        gene_sets: dict
            Gene set for enrichment analysis, must be organized as {'Term1': [Gene1, Gene2,...],...},
            or a dict of the above gene set.
        min_size & max_size : int,
            Minimum and maximum size of target gene set to be included in GSEA analysis
        one_sided : bool
            If True, infer statistical significance via one-sided
            p-values. Else, use two-sided p-values.

        Returns
        -------
        None
            Results of this function are stored on `self.gsea_res`.
        
        References
        ----------
        [1] Fulcher, B. D., Arnatkeviciute, A. & Fornito, A. Overcoming false-
            positive gene-category enrichment in the analysis of spatially resolved 
            transcriptomic brain atlas data. Nat. Commun. 12, 2669 (2021)
        [2] Martins, D. et al. Imaging transcriptomics: convergent cellular, 
            transcriptomic, and molecular neuroimaging signatures in the healthy 
            adult human brain. Cell Rep. 37, 110173 (2021).
        """
        
        # Determine data type of given gene sets
        value_types = {type(v) for v in gene_sets.values()}
        
        # Perform spatial permutation test-based GSEA on single Dict
        if value_types == {list}:
            gsea_res = sct.gsea_perm(
                self.PLS_report, "Weight","Sign", 
                self.weight_perm, self.sign_perm, 
                gene_sets, min_size, max_size, 
                one_sided, self.__n_jobs
            )
        # Perform spatial permutation test-based GSEA on multiple Dicts
        elif value_types == {dict}:
            gsea_res = {}
            for key, gs in gene_sets.items():
                gsea_res[key] = self.gsea_res = sct.gsea_perm(
                    self.PLS_report, "Weight", "Sign", 
                    self.weight_perm, self.sign_perm, gs, 
                    min_size, max_size, one_sided, self.__n_jobs
                )
        return gsea_res

class ZOOM_SC(ZOOM):
    
    """
    The ZOOM_SC class extends traditional imaging-transcriptomics paradigm (ZOOM
    class) by link SBP-relevant gene sets with single-cell RNA sequencing dataset. 

    Specifically, this class provides:
    - Preprocess scRNA-seq for this analysis during initialization.
    - Calculate single-cell enrichment score of SBP-relevant gene sets and infer
      statistical significance at single-cell resolution.
    - Downstream analyses based on single-cell enrichment scores:
      - Differential expressed gene analysis.
      - Region enrichment analysis.
    
    Attributes
    ----------
    adata : str or AnnData
        Path to .h5ad file or AnnData object, where scRNA-seq dataset is stored.
    SBP_scores : pd.DataFrame
        Single-cell enrichment scores of SBP-relevant gene sets.
    DS : pd.DataFrame
        Gene-level differential stability(DS).
    processed : bool
        If False, perprocess scRNA-seq data, skip preprocess pipeline else.
    QC : bool
        If True, filter low-quality genes and cells through scanpy
        - sc.pp.filter_cells(adata, min_genes=min_genes)
        - sc.pp.filter_genes(adata, min_cells=min_cells)
    d : int
        Number of nearest neighbors for each cell.
    gss_limit : int
        Allowed maximum GSS value to avoid over-representation.
    
    References
    ----------
    [1] Zhang, M. J. et al. Polygenic enrichment distinguishes disease associations 
        of individual cells in single-cell RNA-seq data. Nat. Genet. 54, 1572-1580 (2022).
    [2] Song, L., Chen, W., Hou, J., Guo, M. & Yang, J. Spatially resolved mapping 
        of cells associated with human complex traits. Nature 641 932-941 (2025).
    [3] Fulcher, B. D., Arnatkeviciute, A. & Fornito, A. Overcoming false-positive 
        gene-category enrichment in the analysis of spatially resolved transcriptomic 
        brain atlas data. Nat. Commun. 12, 2669 (2021).
        
    This class also learned from references in class `ZOOM`. If you use functions in class
    `ZOOM`, please also cite them.
    """
    
    def __init__(
        self,
        adata: Union[os.PathLike, ad.AnnData],
        expression: Union[os.PathLike, pd.DataFrame],
        SBP: Union[os.PathLike, pd.DataFrame],
        SBP_perm: Union[os.PathLike, pd.DataFrame],
        best_comp: Optional[Union[int, None]] = None,
        #gene_sets: Optional[Dict[str, Union[Dict[str, Union[str, List[str]]], List[str]]]] = None,
        cv: int = 10,
        processed: bool = False,
        DS: Optional[Union[os.PathLike, pd.DataFrame]] = None,
        QC: bool = True,
        min_genes: int = 250,
        min_cells: int = 50,
        d: int = 50,
        gss_limit: int = 200,
        seed: int = 123,
        n_jobs: int = -1
    ) -> None:
        
        """
        Initializer of class ZOOM_SC, a subclass of ZOOM.
        
        If the AnnData object (scRNA-seq data) has not been preprocessed yet, the
        built-in preprocess line will strat:
        - Filter low-quality cells and genes with
          - sc.pp.filter_cells(adata, min_genes=min_genes)
          - sc.pp.filter_genes(adata, min_cells=min_cells)
        - Find `d` nearest neighbors for each cell.
        - Keep commnon genes between scRNA-seq data and AHBA and add `DS` on `adata`.
        - Calculate gene specificity scores.
        
        Reference
        ---------
        Song, L., Chen, W., Hou, J., Guo, M. & Yang, J. Spatially resolved mapping 
        of cells associated with human complex traits. Nature 641 932-941 (2025).
        """
        
        # Inherit main parameters from ZOOM
        super().__init__(
            expression, SBP, 
            SBP_perm, best_comp, 
            cv, seed, n_jobs
        )
        # Load scRNA-seq data
        self.adata = load_sc(adata,flag_sparse=True)
        # Initialize parameters and results for this stage
        self.SBP_score = None
        # Preprocess AHBA and scRNA-seq data
        if not processed:
            self.expression, self.adata = sct.preprocess(
                self.adata, QC, min_genes, 
                min_cells, d, expression, DS,
            )
        if "gss" not in self.adata.layers:
            self.adata = sct.rank_expression(self.adata)
            self.adata = sct.compute_gss(
                self.adata, gss_limit, self.__n_jobs
            )
        self.adata.X = self.adata.layers['gss']
        del self.adata.layers['gss']
    
    def get_SBP_score(
        self,
        direction: bool,
        gene_size: int,
        ctrl_match_key: str = "gss_max",
        weight_opt: str = "DS",
        n_genebin: int = 20,
        return_ctrl_raw_score: bool = False,
        return_ctrl_norm_score: bool = False,
        fdr_method: str = "fdr_bh",
        pval: str = "pval",
        alpha: float = 0.1,
        group: Optional[str] = None
    ) -> None:
        
        """
        Calculate single-cell SBP-relevant enrichment scores and infer 
        statistical significance.

        Parameters
        ----------
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
        fdr_method : str
            Method for multiple testing correction.
        pval : str
            Column name indicating the cell-level p-values, must be present 
            in `self.SBP_scores`.
        alpha : float
            Significance level for multiple testing correction.
        group : str, optional
            Column name indicating the cell groups, based on which p-values are adjusted, 
            must be present in `self.adata.obs`, only used for `group_bh`.

        Returns
        -------
        None
            Results stored in `self.SBP_scores`.
            
        References
        ----------
        [1] Zhang, M. J. et al. Polygenic enrichment distinguishes disease associations 
            of individual cells in single-cell RNA-seq data. Nat. Genet. 54, 1572-1580 (2022).
        [2] Fulcher, B. D., Arnatkeviciute, A. & Fornito, A. Overcoming false-positive 
            gene-category enrichment in the analysis of spatially resolved transcriptomic 
            brain atlas data. Nat. Commun. 12, 2669 (2021).
        [3] Hu, J. X., Zhao, H. & Zhou, H. H. False discovery rate control with groups. 
            J. Am. Stat. Assoc. 105, 1215-1227 (2010).
        """
        
        # Calculate single-cell SBP-relevant enrichment scores
        self.SBP_scores = sct.score_cell_zoom(
            self.adata, self.PLS_report,
            "Weight", "Sign", "p_perm",
            self.weight_perm, self.sign_perm,
            direction, gene_size, ctrl_match_key, 
            weight_opt, n_genebin, return_ctrl_raw_score, 
            return_ctrl_norm_score, self._ZOOM__n_jobs
        )
        
        # Infer statistical significance
        fdr_exist = ["bonferroni","sidak","holm-sidak","holm","simes-hochberg",
                     "hommel","fdr_bh","fdr_by","fdr_tsbh","fdr_tsbky"]
        if fdr_method in fdr_exist:
            _, p_adj, _, _ = multipletests(
                self.SBP_scores[pval], 
                alpha=alpha, 
                method=fdr_method
            )
            self.SBP_scores['p_adj'] = p_adj
            self.SBP_scores[f'p_fdr{alpha}'] = np.where(self.SBP_scores['p_adj']<alpha, 'True', 'False')
        elif fdr_method == 'group_bh':
            self.SBP_scores = sct.group_bh(
                self.adata, self.SBP_scores, 
                pval, group, alpha
            )
        else:
            fdr_exist = fdr_exist + ["group_bh"]
            raise ValueError(
                f"The argument `fdr_method` must be one of {fdr_exist}. "
                f"However, the provided value `{fdr_method}` is invalid. "
            )
        return None
    
    def downstream_ans(
        self,
        flag_DEG: bool = True,
        alpha: float = 0.1,
        min_score: float = 3.0,
        group: Optional[str] = None,
        rank_method: str = "logreg",
        max_iter: int = 10000,
        flag_region: bool = True,
        region_col: Optional[str] = None,
        batch_col: Optional[str] = None,
        dataset: Optional[List] = None,
        indvd_col: Optional[str] = None
    ) -> None:
        
        """
        Perform downstream analyses based on single-cell enrichment scores.

        Parameters
        ----------
        flag_DEG : bool
            Whether to perform differential expression analysis.
        alpha : float
            Significance threshold for enrichment scores.
        min_score : float
            Minimum enrichment score threshold.
        group : str
            Column name of grouping variable for DEG analysis, must be present
            in `self.adata.obs`.
        rank_method : str
            Ranking method for DEG analysis (default: "logreg").
        max_iter : int
            Maximum iterations for DEG ranking.
        flag_region : bool
            Whether to perform region enrichment analysis, must be present in 
            `self.adata.obs`.
        region_col : str, optional
            Column specifying region identity, must be present in `self.adata.obs`.
        batch_col : str, optional
            Column specifying batch identity, must be present in `self.adata.obs`.
        dataset : list, optional
            Dataset identifiers for region enrichment, must be present in 
            `self.adata.obs[batch_col]`.
        indvd_col : str, optional
            Column specifying individual identity, must be present in `self.adata.obs`.

        Returns
        -------
        None
            Updates `self.adata`.
            
        References
        ----------
        [1] Wolf, F. A., Angerer, P. & Theis, F. J. SCANPY: large-scale single-cell 
            gene expression data analysis. Genome Biol. 19, 15 (2018).
        [2] Yang, L. et al. Projection-TAGs enable multiplex projection tracing and 
            multi-modal profiling of projection neurons. Nat. Commun. 16, 5557 (2025).
        """
        
        # DEG analysis between significnat and non-significant cells in each `group`
        if flag_DEG:
            self.adata = sct.downstream_DEG(
                self.adata, self.SBP_scores,
                alpha, min_score, group,
                rank_method, max_iter
            )
        # Region enrichment analysis between significnat and non-significant cells in each `group`
        if flag_region:
            self.adata = sct.downstream_region_enrich(
                self.adata, self.SBP_scores,
                alpha, min_score, group,
                region_col, batch_col,
                dataset, indvd_col,
            )
        return None
