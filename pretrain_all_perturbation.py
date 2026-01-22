import omegaconf
from data2vec.trainer_all_perturb import mcRNA_Trainer_allperturb

if __name__ == "__main__":
    import argparse
    import warnings

    warnings.filterwarnings("ignore", category=UserWarning)
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="path to yaml config file")
    parser.add_argument("-i", type=str, help="location of split h5ad files" )
    parser.add_argument("-o", type=str, help="output folder" )

    args = parser.parse_args()
    cfg_path = args.config
    cfg = omegaconf.OmegaConf.load(cfg_path)
    H5AD_LOC = args.i + "/*.h5ad"
    print("H5AD LOC", H5AD_LOC)
    cfg.H5AD_FILES = H5AD_LOC
    cfg.HIGHLY_VAR_GENES_PATH = args.o+"/feat.csv"
    cfg.train.log_dir = args.o + '/pretrain_all_perturb/logs' 
    cfg.train.checkpoints_dir = args.o + '/pretrain_all_perturb/checkpoints' 
    trainer = mcRNA_Trainer_allperturb(cfg)
    trainer.train()
