# -*- coding: utf-8 -*-
"""
Functionality for perparing neccessary data for ZOOM.
- Regional AHBA processing pipeline optimized for cortical samples.
- Regional SBP and spatial permutation test.
- scRNA-seq preprocess and calculate gene specificity scores.
"""

import numpy as np
import pandas as pd
import nibabel as nib
import abagen
import os
from abagen import io
from abagen.samples_ import _get_struct
from abagen.correct import keep_stable_genes
from neuromaps.datasets import fetch_atlas
from neuromaps.images import load_data
from neuromaps.nulls.spins import (gen_spinsamples, get_parcel_centroids,
                                   load_spins, spin_data, spin_parcels)
from typing import Union, Tuple
from zoom.data_loader import (get_vertex_num, load_gii, load_parc)

def abagen_ctx(
    atlas: Union[nib.Nifti1Image, Tuple, dict],
    atlas_info: Union[pd.DataFrame, os.PathLike],
    data_dir: os.PathLike,
    donors_threshold: int,
    gene_stability_threshold: float
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    
    """
    AHBA processing pipeline optimized for cortical samples.

    Parameters
    ----------
    atlas : nibabel.Nifti1Image, tuple or dict
        Brain atlas specification. Can be:
        - A parcellation image in MNI space.
        - A tuple of GIFTI images in fsaverage5 space.
        - a dictionary where keys are donor IDs and values are parcellation 
          images (or surfaces) in the native space of each donor.
    atlas_info : pd.DataFrame or os.PathLike
        Filepath to or pre-loaded dataframe containing information about atlas,
        see `abagen` document.
    data_dir : os.PathLike
        Directory where expression data should be downloaded (if it does not 
        already exist) / loaded.
    donors_threshold : int
        Minimum number of donors required for a region to be included.
    gene_stability_threshold : float
        Threshold for filtering genes based on differential stability across donors.

    Returns
    -------
    expression : pd.DataFrame
        Processed regional AHBA expression.
    stability : pd.DataFrame
        Gene-level differential stability.

    References
    ----------
    [1] Markello, R. D., Arnatkeviciute, A., Poline, J. B., Fulcher, B. D., 
        Fornito, A. & Misic, B. Standardizing workflows in imaging transcriptomics 
        with the abagen toolbox. Elife 10, e72129 (2021).
    [2] Dear, R. et al. Cortical gene expression architecture links healthy 
        neurodevelopment to the imaging, transcriptomics and genetics of autism and 
        schizophrenia. Nat. Neurosci. 27, 1075-1086 (2024).
    """
    
    # Patch `drop_mismatch_samples` function to keep only left cortical samples before probe selection
    def drop_mismatch_samples_and_filter(annotation, ontology):
        annotation = io.read_annotation(annotation)
        ontology = io.read_ontology(ontology).set_index('id')
        sid = np.asarray(annotation['structure_id'])
        # get hemisphere and structure path
        hemisphere = np.asarray(ontology.loc[sid, 'hemisphere']
                                        .replace({np.nan: 'B'}))
        structure = np.asarray(ontology.loc[sid, 'structure_id_path']
                                       .apply(_get_struct))
        # add hemisphere + brain "structure" designation to annotation data and
        # only keep samples with consistent hemisphere + MNI coordinate designation
        annot = annotation.assign(hemisphere=hemisphere, structure=structure) \
                          .query('(hemisphere == "L" & mni_x < 0) '
                                 '| (hemisphere == "R" & mni_x > 0) '
                                 '| (hemisphere == "B" & mni_x == 0)',
                                 engine='python')
        annot = annot.copy().query("structure == 'cortex'")
        annot = annot.copy().query("hemisphere == 'L'")
        return annot

    ### Replace original function in abagen package with patched function ###
    abagen.samples_.drop_mismatch_samples = drop_mismatch_samples_and_filter
    
    # Get donor-wised regional AHBA expression with modified abagen
    expression, counts, report = abagen.allen.get_expression_data(
        atlas = atlas, atlas_info = atlas_info, data_dir = data_dir,
        n_proc = 6, return_donors = True, return_counts = True, 
        return_report = True, verbose = 1, lr_mirror = 'rightleft'
    )
    expression = list(expression.values())
    
    # Find regions with at least 1 sample from at least `donors_threshold` donors
    region_filter = (counts > 0).sum(axis=1) >= donors_threshold
    # Filter all donor expression matrices for those regions
    expression = [e.loc[region_filter, :] for e in expression]
    
    # Remember gene labels to use as index
    gene_labels = expression[0].columns
    # Filter all donor expression matrices for stable genes
    expression, stability = keep_stable_genes(
        expression, threshold = gene_stability_threshold,
        return_stability = True
    )
    stability = pd.Series(stability, index=gene_labels)
    
    # Aggregate final expression matrix
    expression = pd.concat(expression).groupby('label').mean()
    stability = pd.DataFrame(stability, columns=["DS"])
    return expression, stability

def fetch_medial_wall(
    atlas: str,
    density: str,
    hemi: str
) -> list:
    
    """
    Fetch indices of medial wall vertices.

    Parameters
    ----------
    atlas: {'fsaverage', 'fSLR', 'civet'} optional,
        Name of surface atlas on which `parcellation` is defined.
    density : str, optional
        Density of surface mesh on which `parcellation` is defined. Must becompatible 
        with specified `atlas`.
    hemi: {'L','lh','left','both'} optional,
        Hemisphere used to perform downstream analyses.

    Returns
    -------
    medial_wall : list
        Indices of medial wall vertices.
    """
    
    vertex_num = get_vertex_num(atlas, density, hemi='L')
    # For development stage
    pwd = "/slurm/home/yrd/liaolab/nieshuyang/ZOOM/data/medial_wall"
    
    # Fetch medial wall indices
    if hemi in ['L','lh','left']:        
        filename = f"{pwd}/lh.medial_wall.{atlas}.{density}.txt"
        medial_wall = list(np.loadtxt(filename).astype(np.int16))
    elif hemi=='both':
        lh_filename = f"{pwd}/lh.medial_wall.{atlas}.{density}.txt"
        rh_filename = f"{pwd}/rh.medial_wall.{atlas}.{density}.txt"
        lh_medial_wall = list(np.loadtxt(lh_filename).astype(np.int16))
        rh_medial_wall = list(np.loadtxt(rh_filename).astype(np.int16))
        rh_medial_wall = list(map(lambda x: x+vertex_num,rh_medial_wall))
        medial_wall = lh_medial_wall+rh_medial_wall
    return medial_wall
    
def process_SBP(
    SBP: Union[os.PathLike, tuple, nib.gifti.gifti.GiftiImage],
    parcellation: Union[os.PathLike, tuple, nib.gifti.gifti.GiftiImage],
    atlas: str, density: str, hemi: str, n_perm: int, seed: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    
    """
    Assign SBP image to parcellation image and perform spatial permutation test.

    Parameters
    ----------
    SBP : nibabel.Nifti1Image, tuple or dict
        Brain atlas specification. Can be:
        - A parcellation image in MNI space.
        - A tuple of GIFTI images in fsaverage5 space.
        - a dictionary where keys are donor IDs and values are parcellation 
          images (or surfaces) in the native space of each donor.
    parcellation : path_like str, tuple, nib.gifti.gifti.GiftiImage
        Recognized parcellation image
        - If hemi in ['L','lh','left']:
            - path_like str: Filepaths to .gii or .annot
            - nib.gifti.gifti.GiftiImage
        - Otherwise, use both hemisphere:
            - tuple: must be organized as L/R pair
                - (str, str): Filepaths to .gii or .annot
                - (nib.gifti.gifti.GiftiImage, nib.gifti.gifti.GiftiImage)
    atlas: {'fsaverage', 'fSLR', 'civet'} optional,
        Name of surface atlas on which `parcellation` is defined.
    density : str, optional
        Density of surface mesh on which `parcellation` is defined. Must becompatible 
        with specified `atlas`.
    hemi: {'L','lh','left','both'} optional,
        Hemisphere used to perform downstream analyses.
    n_perm : int
        Number of permuation test to perform.
    seed : int
        Random seed of spatial permutation test.

    Returns
    -------
    df_SBP_parc : pd.DataFrame
        SBP values in parcellation labels.
    df_surr_parc : pd.DataFrame
        Permutated SBP values in parcellation labels.

    References
    ----------
    [1] Markello, R. D. et al. Neuromaps: structural and functional interpretation 
        of brain maps. Nat. Methods 19, 1472-1479 (2022)..
    [2] Alexander-Bloch, A. F. et al. On testing for spatial correspondence between 
        maps of human brain structure and function. Neuroimage 178, 540-551 (2018).
    """
    
    # Fetch medial wall and vertex number
    medial_wall = fetch_medial_wall(atlas, density, hemi)
    vertex_num = get_vertex_num(atlas, density, hemi)
    
    # Generate surrogates utilizing Alexander-Bloch's spatial permutation test
    surfaces = fetch_atlas(atlas, density)['sphere']
    coords, hemis = get_parcel_centroids(surfaces, method='surface')
    coords = coords[:vertex_num, :]
    coords = np.delete(coords, medial_wall, axis=0)
    hemis = hemis[:vertex_num]
    hemis = np.delete(hemis, medial_wall)
    spins = gen_spinsamples(coords, hemis, n_rotate=n_perm, seed=seed)
    spins = load_spins(spins)
    
    # Load SBP and parcellation
    SBP_data = load_gii(SBP, atlas, density, hemi)
    SBP_data[medial_wall] = np.full((len(medial_wall),), np.nan)
    SBP_data_valid = np.delete(SBP_data, medial_wall)
    parc_data = load_parc(parcellation, atlas, density, hemi)
    parc_labels = list(np.unique(parc_data).astype(np.int16))
    
    # Assign SBP into parcellation image
    SBP_parc = np.full((len(parc_labels),), np.nan)
    for i in range(len(parc_labels)):
        label = parc_labels[i]
        label_idx = np.where(parc_data == label)[0]
        label_values = SBP_data[label_idx]
        SBP_parc[i] = np.mean(label_values)
    df_SBP_parc = pd.DataFrame(SBP_parc, index=parc_labels, 
                               columns=['SBP'])
    
    # Project permutated SBP onto standard surface
    surrogates = load_data(SBP_data_valid)[spins]
    surr_surf = np.full((vertex_num, n_perm), np.nan)
    for i in range(n_perm):
        surr_surf[:,i][~np.isnan(SBP_data)] = surrogates[:,i]
        
    # Assign permutated SBP into parcellation image
    surr_parc = np.full((len(parc_labels), n_perm), np.nan)
    for p in range(n_perm):
        for i in range(len(parc_labels)):
            label = parc_labels[i]
            label_idx = np.where(parc_data == label)[0]
            label_values = surr_surf[label_idx, p]
            surr_parc[i, p] = np.mean(label_values)
    df_surr_parc = pd.DataFrame(surr_parc, index=parc_labels, 
                                columns=[f'Perm{i+1}' for i in range(n_perm)])
    return df_SBP_parc, df_surr_parc
