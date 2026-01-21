from glob import glob

import numpy as np
import pandas as pd
import scanpy as sc
from pathlib import Path





def process_donor(donor,treatments):
    global data #, file

    print(f"Processing: {donor}")
    donor_h5 = data[data.obs[DONOR_COLUMN] == donor]
    for treatment in treatments:
        save_path = SAVE_PATH + f"/{treatment}/DONOR_{donor}.h5ad"
        sc.write(save_path, donor_h5[donor_h5.obs["treatment"] == treatment])
        print(f"Saved to {save_path}")

if __name__ == "__main__":
    import argparse
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", type=str, help="path to input (h5ad data)")
    parser.add_argument("-o", type=str, help="output folder (separated h5ad files)")
    parser.add_argument("-p", type=str, help="name of column to predict/disease column (genotype for our data)")
    args = parser.parse_args()


    INPUT_H5AD=args.i
    SAVE_PATH = args.o
    DONOR_COLUMN = "donor_id"
    DISEASE_COLUMN = "disease"
    print(f"Processing: {INPUT_H5AD}")
    data = sc.read_h5ad(INPUT_H5AD, chunk_size=20000)
    #data.X = data.raw.X
    #file = INPUT_H5AD.raw.to_adata().to_memory()
    data.obs[DONOR_COLUMN] = data.obs["line"].astype(str)
    data.obs[DISEASE_COLUMN] = data.obs["genotype"]
    #sc.pp.normalize_total(data, target_sum=1, inplace=True)
    donors = data.obs[DONOR_COLUMN].unique()
    treatments = data.obs["treatment"].unique()
    #make separate folders for each treatment
    for treatment in treatments:
        directory_path = Path(SAVE_PATH+"/"+treatment)
        directory_path.mkdir(parents=True, exist_ok=True) 
    for donor in donors:
        process_donor(donor, treatments)