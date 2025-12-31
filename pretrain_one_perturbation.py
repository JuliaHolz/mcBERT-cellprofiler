import omegaconf
from data2vec.trainer import mcRNA_Trainer

if __name__ == "__main__":
    import argparse
    import warnings

    warnings.filterwarnings("ignore", category=UserWarning)
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="path to yaml config file")
    parser.add_argument("--perturb", type=str, help="perturbation name (autophagy, DMSO, H202, KPT or tunicamycin)")
    
    args = parser.parse_args()
    perturbation = args.perturb
    cfg_path = args.config
    cfg = omegaconf.OmegaConf.load(cfg_path)
    H5AD_LOC = "/home/jholz/fraenkel_rotation/mcBERT-cellprofiler/h5ad_files/standardized_originalfeat/" + perturbation + "/*.h5ad"
    cfg.H5AD_FILES = H5AD_LOC
    cfg.train.log_dir = '/home/jholz/fraenkel_rotation/mcBERT-cellprofiler/0utputs/pretrain_'+perturbation+'_only/logs' 
    cfg.train.checkpoints_dir = '/home/jholz/fraenkel_rotation/mcBERT-cellprofiler/0utputs/pretrain_'+perturbation+'_only/checkpoints' 
    trainer = mcRNA_Trainer(cfg)
    trainer.train()
