from glob import glob

import pandas as pd
import scanpy as sc

"""Creates CSV of feature names
"""
if __name__ == "__main__":
    import argparse
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", type=str, help="path to input (h5ad data)")
    parser.add_argument("-o", type=str, help="output location for csv of feature names")
    args = parser.parse_args()
    adata = sc.read_h5ad(args.i, chunk_size=20000)
    df = pd.DataFrame({"genes": adata.var.index.tolist()})
    # SAVE GENES Here
    df.to_csv(args.o, index=False)
