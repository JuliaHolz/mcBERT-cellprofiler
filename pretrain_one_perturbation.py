import omegaconf
from data2vec.trainer import mcRNA_Trainer

if __name__ == "__main__":
    import argparse
    import warnings

    warnings.filterwarnings("ignore", category=UserWarning)
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="path to yaml config file")
    parser.add_argument("--perturb", type=str, help="perturbation name (autophagy, DMSO, H202, KPT or tunicamycin)")
    parser.add_argument("-i", type=str, help="location of split h5ad files" )
    parser.add_argument("-o", type=str, help="output folder" )

    args = parser.parse_args()
    perturbation = args.perturb
    cfg_path = args.config
    cfg = omegaconf.OmegaConf.load(cfg_path)
    H5AD_LOC = args.i + "/*.h5ad"
    cfg.H5AD_FILES = H5AD_LOC
    cfg.HIGHLY_VAR_GENES_PATH = args.o+"/feat.csv"
    cfg.train.log_dir = args.o + '/pretrain_'+perturbation+'_only/logs' 
    cfg.train.checkpoints_dir = args.o + '/pretrain_'+perturbation+'_only/checkpoints' 
    trainer = mcRNA_Trainer(cfg)
    trainer.train()
