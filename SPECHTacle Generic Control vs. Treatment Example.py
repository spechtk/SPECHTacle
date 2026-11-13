# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 12:27:35 2026

@author: spechtkm

Control vs. Treatment SPECHTacle Example
"""

import os
import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wot.ot import OTModel
from scipy.stats import entropy

# ----------------------------
# USER SETTINGS
# ----------------------------
''' P ''' # Directory Folder Name
fig_dir_name = "WOT_Fate_Entropy_Maps_Name_Date"  # CHANGE FOLDER NAME PER DATASET
fig_dir = os.path.join(os.getcwd(), fig_dir_name)
os.makedirs(fig_dir, exist_ok=True)

''' P ''' # File Input
adata_file = r"G:\User\Sample_file.h5ad" # Actual Data Input (Loaded as .h5ad file)

# Reads data from .h5ad file to an AnnData object called adata
adata = sc.read_h5ad(adata_file)
                     
# ----------------------------
# If not already preprocessed/normalized, normalize and scale data (If data is preprocessed skip to PCA)
# ----------------------------
# Normalize & log-transform
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
adata.raw = adata.copy()

# Scale the data (optional but often recommended before PCA)
sc.pp.scale(adata, max_value=10)

# ----------------------------
# PCA and Finding Neighbors
# ----------------------------
# Principal Component Analysis (PCA)
sc.tl.pca(adata, svd_solver='arpack')

# Compute neighbors graph
sc.pp.neighbors(adata, n_neighbors=15, n_pcs=15)

# UMAP embedding
sc.tl.umap(adata)

# ----------------------------
# Leiden clustering
# ----------------------------
# Choose a resolution to be used for the WOT analysis (higher resolution = more clusters)
''' P ''' # Leiden Resolution
sc.tl.leiden(
    adata,
    resolution=0.3, # adjust here
    key_added='leiden_wot',
    flavor='igraph',
    directed=False,
    n_iterations=2
)
sc.pl.umap(adata, color='leiden_wot', show=False)
plt.savefig(os.path.join(fig_dir, 'umap_leiden_wot.png'), bbox_inches='tight')
plt.close()


# ----------------------------
# Assign timepoints (If timepoint as a numerical value is already a metadata column, this can be skipped)
# ----------------------------
#Timepoints as Numbers
''' P ''' # Timepoint Variables
t0 = 0.0  # 'Conrol'
t1 = 1.0  # 'Treatment'

print("OBS COLUMNS:", adata.obs.columns)

# Auto-detect sample column
for candidate in ['Sample', 'sample', 'orig.ident', 'batch', 'grp']:
    if candidate in adata.obs.columns:
        sample_col = candidate
        break
else:
    raise ValueError("No sample column found in adata.obs")

print(f"Using '{sample_col}' as sample column")

#Samples as they appear in sample column
''' P ''' # Sample Variables
S1  = 'C1' # Sample 1 (Control Sample #1)
S2  = 'C2'
S3  = 'C3'
S4  = 'C4'
S5  = 'T1' # (Treatment Sample #1)
S6  = 'T2'
S7  = 'T3'
S8  = 'T4'

''' P ''' # Sample Timepoint Matchups
sample_to_timepoint = {
    S1: t0, S2: t0, S3: t0, S4: t0, 
    S5: t1, S6: t1, S7: t1, S8: t1,
} # Assign samples to numbered time value

def assign_timepoint(x):
    x = str(x)
    for sample, tp in sample_to_timepoint.items():
        if sample in x:
            return tp
    return np.nan

adata.obs['timepoint'] = adata.obs[sample_col].apply(assign_timepoint)

print(adata.obs['timepoint'].value_counts())


adata.obs['str_timepoint'] = adata.obs['timepoint'].astype(str)

sc.pl.umap(adata, color = 'str_timepoint', show=False)
plt.savefig(os.path.join(fig_dir, 'umap_timepoint.png'), bbox_inches='tight')
plt.close()

# ----------------------------
# WOT: create OTModel and compute transport maps
# ----------------------------

adata.obs['wot_cluster'] = adata.obs['leiden_wot']

# Build and Set Perameters for Optimal Transport (OT) Model 
    # Epsilon: entropic regularization
    # Lambda1: growth regularization of starting point
    # Lambda2: growth regularization of terminal point
    # tolerance: Convergence Threshold
    # max_iter: Maximum Iterations
''' P ''' # WOT Parameters
ot_model = OTModel(
    adata,
    day_field='timepoint',
    epsilon=0.2,
    lambda1=0.5,
    lambda2=5.0,
    tolerance=1e-5,
    max_iter=50
)

# Transport Maps 
tm_t0_t1_adata = ot_model.compute_transport_map(t0, t1)


# Transport Map to Matrix
tm_t0_t1 = tm_t0_t1_adata.X


# Assign each day's cells their own variables
t0_cells = adata.obs_names[adata.obs['timepoint'] == t0].intersection(tm_t0_t1_adata.obs_names)
t1_cells = adata.obs_names[adata.obs['timepoint'] == t1].intersection(tm_t0_t1_adata.var_names)


# ----------------------------
# Aggregate transport to clusters
# ----------------------------
# Store each days cell's cluster value
t0_clusters = adata.obs.loc[t0_cells, 'wot_cluster']
t1_clusters = adata.obs.loc[t1_cells, 'wot_cluster']


# Turn Transport Map into a clustered data frames
# Groupby using .T to avoid future warnings
tm_t0_t1_df = pd.DataFrame(tm_t0_t1, index=t0_cells, columns=t1_cells)
tm_t0_t1_clustered = tm_t0_t1_df.groupby(t0_clusters, observed=False).sum().T.groupby(t1_clusters, observed=False).sum().T

# Normalize per source cell to sum to 1
tm_t0_t1_norm = tm_t0_t1 / tm_t0_t1.sum(axis=1, keepdims=True)


# Normalize transport matrices to get fractions per source cluster
tm_t0_t1_cluster_norm = tm_t0_t1_clustered.div(tm_t0_t1_clustered.sum(axis=1), axis=0)


# ----------------------------
# Fate entropy per cell
# ----------------------------
# Compute and store Shannon entropy scores. Plot cells on UMAP colored by score
# For color scale define custom, non-uniform boundaries
import matplotlib.pyplot as plt
import matplotlib.colors as colors
bounds = [0, 2, 4, 6, 7, 8, 9, 10]
cmap = plt.get_cmap('magma_r')
norm = colors.BoundaryNorm(bounds, cmap.N)

cell_entropy1 = entropy(tm_t0_t1_norm.T + 1e-9)  # avoid log(0)
adata.obs.loc[t0_cells, 'fate_entropy1'] = cell_entropy1
ax = sc.pl.umap(adata, color='fate_entropy1', cmap='magma_r', show=False, colorbar_loc=None) # Color UMAP by fate_entropy score
ax.collections[0].set_norm(norm)
plt.colorbar(ax.collections[0], ax=ax, ticks=bounds)
plt.title("t0 → t1 Fate Entropy Map")
plt.savefig(os.path.join(fig_dir, 'fate_entropy_t0_t1_umap.png'), bbox_inches='tight')
plt.close()


# ----------------------------
# Overaly of Entropy and Leiden Clusters
# ----------------------------
import matplotlib.pyplot as plt

X = adata.obsm["X_umap"]
x = X[:, 0]
y = X[:, 1]

leiden = adata.obs["leiden_wot"].astype("category")
fate_entropy = adata.obs["fate_entropy1"]

# Get the colors used for leiden_wot
leiden_colors = adata.uns["leiden_wot_colors"]  # list of hex strings

fig, ax = plt.subplots(figsize=(6, 5))

# 1) Points (e.g., fate_entropy)
sc = ax.scatter(
    x,
    y,
    c=fate_entropy,
    cmap="magma_r",
    s=2,
    alpha=0.9,
    zorder=2,
)
cbar = plt.colorbar(sc, ax=ax)
cbar.set_label("fate_entropy1")

# 2) Hulls using the SAME colors as the Leiden plot
from scipy.stats import gaussian_kde

for i, cat in enumerate(leiden.cat.categories):

    mask = (leiden == cat).values
    points = X[mask]

    if len(points) < 30:
        continue

    color = leiden_colors[i]

    # Estimate density
    ''' P ''' # Bandwidth
    kde = gaussian_kde(points.T, bw_method=0.1)

    # Grid covering just this cluster
    xmin, ymin = points.min(axis=0) - 0.5
    xmax, ymax = points.max(axis=0) + 0.5

    xx, yy = np.meshgrid(
        np.linspace(xmin, xmax, 150),
        np.linspace(ymin, ymax, 150)
    )

    grid = np.vstack([xx.ravel(), yy.ravel()])
    zz = kde(grid).reshape(xx.shape)

    # Draw only the outer contour
    ax.contour(
        xx,
        yy,
        zz,
        levels=[zz.max()*0.25],
        colors=[color],
        linewidths=2,
        zorder=5,
    )

ax.set_xlabel("UMAP1")
ax.set_ylabel("UMAP2")
ax.set_title("UMAP: fate_entropy with Leiden-colored outlines")
plt.savefig(os.path.join(fig_dir, 'entropy_with_cluster_oultines_leiden.png'), bbox_inches='tight')
plt.close()

# ----------------------------
# Overaly of Entropy and Celltype Clusters
# ----------------------------
import scanpy as sc
''' P ''' # Celltype Column
CT_col =  "celltype"

if CT_col in adata.obs.columns:
    sc.pl.umap(adata, color=CT_col,show=False)
    plt.savefig(os.path.join(fig_dir, 'umap_celltype.png'), bbox_inches='tight')
    plt.close()

    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import scanpy as sc
    from scipy.stats import gaussian_kde
    from matplotlib.lines import Line2D

    X = adata.obsm["X_umap"]
    x = X[:, 0]
    y = X[:, 1]

    celltype = adata.obs[CT_col].astype("category")
    fate_entropy = adata.obs["fate_entropy1"]

    # Colors matching the celltype annotation
    celltype_colors = adata.uns[f"{CT_col}_colors"]

    fig, ax = plt.subplots(figsize=(8, 5.5))

    # Scatter plot colored by fate_entropy
    scat = ax.scatter(
        x,
        y,
        c=fate_entropy,
        cmap="magma_r",
        s=2,
        alpha=0.9,
        zorder=2,
    )

    # Single colorbar only
    cbar = plt.colorbar(scat, ax=ax, pad=0.02, fraction=0.05)
    cbar.set_label("fate_entropy1", labelpad=8)

    # Draw contours and build matching legend handles
    legend_handles = []

    for i, cat in enumerate(celltype.cat.categories):
        mask = (celltype == cat).values
        points = X[mask]

        if len(points) < 30:
            continue

        color = celltype_colors[i]

        ''' P ''' # Bandwidth
        kde = gaussian_kde(points.T, bw_method=0.1)

        xmin, ymin = points.min(axis=0) - 0.5
        xmax, ymax = points.max(axis=0) + 0.5

        xx, yy = np.meshgrid(
            np.linspace(xmin, xmax, 150),
            np.linspace(ymin, ymax, 150)
        )

        grid = np.vstack([xx.ravel(), yy.ravel()])
        zz = kde(grid).reshape(xx.shape)

        ax.contour(
            xx,
            yy,
            zz,
            levels=[zz.max() * 0.25],
            colors=[color],
            linewidths=2,
            zorder=5,
        )

        legend_handles.append(
            Line2D(
                [0], [0],
                color=color,
                lw=2,
                label=cat
            )
        )

    # Axes formatting
    ax.set_title("UMAP: fate_entropy with Celltype-colored outlines")
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")

    # Make room on the right for colorbar + legend
    plt.subplots_adjust(right=0.78)

    # Clean legend placement
    ax.legend(
        handles=legend_handles,
        title="Celltype",
        loc="center left",
        bbox_to_anchor=(1.2, 0.5),
        frameon=False,
        fontsize=10,
        title_fontsize=12,
        ncol=1,
        borderaxespad=0.0,
        handlelength=1.8,
        handletextpad=0.5,
        labelspacing=0.35,
    )

    plt.savefig(os.path.join(fig_dir, "umap_fate_entropy_celltype_broad_outlines.png"), bbox_inches="tight", dpi=300)
    plt.close()
else:
    print(f"\n[WARNING] No celltype column found in adata.obs: '{CT_col}'. Skipping celltype plots.\n")

