import pathlib

# ignore mix type warnings from pandas
import warnings

import pandas as pd

from pycytominer import annotate, feature_select, normalize

# pycytominer imports
from pycytominer.cyto_utils.cells import SingleCells
import anndata as ad



if __name__ == "__main__":
    import argparse
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", type=str, help="path to input (csv cellprofiler data)")
    parser.add_argument("-o", type=str, help="output location")
    parser.add_argument('--fs',action='store_true', help="Use pycytominer's feature selection to reduce features")
    parser.add_argument("-s", type=str, help="standardization method: standardize, spherize, robustize or MAD_robustize (uses corresponding pycytominer standardization method)")

    args = parser.parse_args()

    df = pd.read_csv(args.i)
    
    #image features = not the first Unnamed:0 column -- which seems to be a cell individual label
    #and not the last ten which are metadata
    features=df.columns.tolist()[1:-10]
    #metadata features
    meta_feat = [df.columns.tolist()[0]] + df.columns.tolist()[-10:]


    standardization = args.s #standardize, spherize, robustize, MAD_robustze

    if(args.fs):
        print("selecting features")
        df = feature_select(profiles=df,features=features)
        selected_features = df.columns[1:-10].tolist()
        meta_features = [df.columns[0]]+df.columns[-10:].tolist()

    numeric_meta_features=['plate_name', 'well_name', 'site_name']

    nonnumeric_meta_features=['Unnamed: 0','detailed_type','training_type', 
                              'genotype', 'group_of_interest','sex','line', 'treatment']

    #drop the numeric meta features since pycytominer will put them in vars of anndata 
    #we want them in obs. (This is an unfortunate bug in pycytominer)
    no_numeric_meta = df.drop(columns=numeric_meta_features)

    feat_to_normalize = selected_features if args.fs else features

    normalized = normalize(no_numeric_meta,features=feat_to_normalize,meta_features=nonnumeric_meta_features,method=standardization,output_file=args.o, output_type="anndata_h5ad")
    adata = ad.read_h5ad(args.o)

    #re-add our numeric metadata features and re-save the adata
    adata.obs["plate_name"] = df["plate_name"].values
    adata.obs["site_name"] = df["site_name"].values
    adata.obs["well_name"] = df["well_name"].values

    adata.write_h5ad(filename=args.o)





    

