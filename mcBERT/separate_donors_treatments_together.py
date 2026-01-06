from glob import glob

import numpy as np
import pandas as pd
import scanpy as sc
from pathlib import Path


INPUT_H5AD="/home/jholz/fraenkel_rotation/mcBERT-cellprofiler/h5ad_files/standardized_originalfeat.h5ad"
#SELECTED_GENES = np.array(pd.read_csv(".csv")["genes"])
SAVE_PATH = r"/home/jholz/fraenkel_rotation/mcBERT-cellprofiler/h5ad_files/"
DATASET_TAG = "standardized_originalfeat"
SAVE_PATH = SAVE_PATH+DATASET_TAG
DONOR_COLUMN = "donor_id"
DISEASE_COLUMN = "disease"



def process_donor(donor):
    global data #, file

    print(f"Processing: {donor}")
    donor_h5 = data[data.obs[DONOR_COLUMN] == donor]
    print(donor_h5)
    save_path = SAVE_PATH + f"/all_perturb/{DATASET_TAG}_DONOR_{donor}.h5ad"
    sc.write(save_path, donor_h5)
    print(f"Saved to {save_path}")



print(f"Processing: {INPUT_H5AD}")
data = sc.read_h5ad(INPUT_H5AD, chunk_size=20000)
#data.X = data.raw.X
#file = INPUT_H5AD.raw.to_adata().to_memory()
data.obs[DONOR_COLUMN] = data.obs["line"].astype(str)
data.obs[DISEASE_COLUMN] = data.obs["genotype"]
#sc.pp.normalize_total(data, target_sum=1, inplace=True)
donors = data.obs[DONOR_COLUMN].unique()
#treatments = data.obs["treatment"].unique()
#make separate folders for each treatment
#for treatment in treatments:
#    directory_path = Path(SAVE_PATH+"/"+treatment)
#    directory_path.mkdir(parents=True, exist_ok=True) 
directory_path = Path(SAVE_PATH+"/"+"all_perturb")
directory_path.mkdir(parents=True, exist_ok=True) 
for donor in donors:
    process_donor(donor)