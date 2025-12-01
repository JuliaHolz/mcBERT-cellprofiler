from glob import glob

import pandas as pd
import scanpy as sc

"""Script to extract the most highly variable genes across multiple dataset. Currently, to save memory each h5ad file is processed individually and
are not concatenated. This means that the highly variable genes are not calculated across all datasets but rather within each dataset.

NOTE: Since we process them individually, we cannot gurantee to get x (e.g., 1000) most highly variable genes.
Therefore, greedily increases top_genes_per_file until we get 1,000 genes.
"""

DATASET_PATH="/home/jholz/fraenkel_rotation/cellpainting_data/standardized_originalfeat.h5ad"
feat_path="/home/jholz/fraenkel_rotation/mcBERT-cellprofiler/generated/feat.csv"
adata = sc.read_h5ad(DATASET_PATH, chunk_size=20000)

feat = adata.var

df = pd.DataFrame({"genes": adata.var.index.tolist()})
print(df.shape)
# SAVE GENES Here
df.to_csv(feat_path, index=False)
