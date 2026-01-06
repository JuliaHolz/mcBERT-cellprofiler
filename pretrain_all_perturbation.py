import omegaconf
from data2vec.trainer_all_perturb import mcRNA_Trainer_allperturb

if __name__ == "__main__":
    import argparse
    import warnings

    warnings.filterwarnings("ignore", category=UserWarning)
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="path to yaml config file")
    
    args = parser.parse_args()
    cfg_path = args.config
    perturbation = "all_perturb"
    cfg = omegaconf.OmegaConf.load(cfg_path)
    H5AD_LOC = "/home/jholz/fraenkel_rotation/mcBERT-cellprofiler/h5ad_files/standardized_originalfeat/" + perturbation + "/*.h5ad"
    cfg.H5AD_FILES = H5AD_LOC
    cfg.train.log_dir = '/home/jholz/fraenkel_rotation/mcBERT-cellprofiler/0utputs/pretrain_'+perturbation+'_only/logs' 
    cfg.train.checkpoints_dir = '/home/jholz/fraenkel_rotation/mcBERT-cellprofiler/0utputs/pretrain_'+perturbation+'_only/checkpoints' 
    trainer = mcRNA_Trainer_allperturb(cfg)
    trainer.train()
