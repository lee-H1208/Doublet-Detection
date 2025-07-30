import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scanpy as sc
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

def remove_least_dense_cells(adata, percentile_threshold=12, verbose=0):
    adata.obs['density_outlier'] = False
    
    density_threshold = np.percentile(adata.obs['local_density'].dropna(), percentile_threshold)
    density_outliers = (adata.obs['local_density'] <= density_threshold) & (~adata.obs['local_density'].isna())
    adata.obs['density_outlier'] = density_outliers
    
    cells_to_keep = ~adata.obs['density_outlier']
    adata_filtered = adata[cells_to_keep, :].copy()
    raw_adata = adata.raw.to_adata()
    adata_filtered.raw = raw_adata[cells_to_keep, :]    
    
    if verbose > 0: 
        print(f"Removing cells with density <= {density_threshold:.4f}")
        print(f"Original: {adata.n_obs} cells")
        print(f"Filtered: {adata_filtered.n_obs} cells")
        print(f"Removed: {adata.n_obs - adata_filtered.n_obs} cells ({100 * (adata.n_obs - adata_filtered.n_obs) / adata.n_obs:.1f}%)")
        print(f"Cells marked as density outliers: {adata.obs['density_outlier'].sum()}")

        print("Generating PCA and UMAP comparison plots...")
        _, axs = plt.subplots(2, 2, figsize=(10, 8))

        sc.pl.pca(adata, color='density_outlier', show=False, ax=axs[0, 0], title="PCA: Original")
        sc.pl.pca(adata_filtered, show=False, ax=axs[0, 1], title="PCA: Filtered")

        sc.pl.umap(adata, color='density_outlier', show=False, ax=axs[1, 0], title="UMAP: Original")
        sc.pl.umap(adata_filtered, show=False, ax=axs[1, 1], title="UMAP: Filtered")

        plt.tight_layout()
        plt.show()

    return adata

def isolate_density_outliers(adata, cluster_key='kmeans', verbose=0):
    n_cells = adata.n_obs
    expected_dblr = int((n_cells / 1e5) * 100)

    print('Isolating density outliers...')
    adata.obs['local_density'] = np.nan 

    coords = adata.obsm["X_pca"][:, :2]  

    coords = StandardScaler().fit_transform(coords)
    coords_df = pd.DataFrame(coords, index=adata.obs_names)

    for cluster in adata.obs[cluster_key].unique():
        cell_ids = adata.obs[adata.obs[cluster_key] == cluster].index
        cluster_coords = coords_df.loc[cell_ids].values
        
        if len(cluster_coords) >= 3:
            nbrs = NearestNeighbors(n_neighbors=max(2, int(len(cluster_coords) * 0.02))).fit(cluster_coords)
            distances, indices = nbrs.kneighbors(cluster_coords)
            
            local_density = 1 / np.mean(distances[:, 1:], axis=1)
            adata.obs.loc[cell_ids, 'local_density'] = local_density

    adata.obs['local_density']

    adata_clean = remove_least_dense_cells(adata, percentile_threshold=expected_dblr, verbose=verbose)
    print('Finished!')

    return adata_clean