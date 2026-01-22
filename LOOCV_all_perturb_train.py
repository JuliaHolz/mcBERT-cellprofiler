import os
from argparse import ArgumentParser
from glob import glob

import numpy as np
import torch
from mcBERT.utils.clustering_utils import get_plot_as_img
from mcBERT.utils.clustering_utils import get_plot_as_img_largerpts

from mcBERT.utils.loocv_utils import get_knn_prediction

from mcBERT.utils.metrics import (
    calc_silhouette_score,
    cosine_similarity_patient_embeddings,
)
from mcBERT.utils.mixed_patient_level_dataset import Mixed_patient_level_dataset
from mcBERT.utils.utils import get_scRNA_model, prepare_dataset, set_seeds
from omegaconf import OmegaConf
from pytorch_metric_learning import losses
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from sklearn.neighbors import KNeighborsClassifier

import csv
set_seeds(42)

"""
After pre-processing all datasets and saving each donor individually, the model can be trained.
Previous Pre-Training is recommended.
"""


# Config file for training
parser = ArgumentParser()
parser.add_argument(
    "--config",
    type=str,
    help="path to yaml config file for fine-tuning training",
)

parser.add_argument(
    "-p",
    type=str,
    help="pretraining checkpoint location",
)

parser.add_argument(
    "-o",
    type=str,
    help="output folder",
)

parser.add_argument(
    "-i",
    type=str,
    help="location of h5ad files",
)


args = parser.parse_args()
PRETRAIN_FOLDER=args.p
FINETUNE_FOLDER =args.o+ "/finetune_all_perturb"
H5AD_LOC = args.i+"/*.h5ad"

PRETRAIN_CHK = PRETRAIN_FOLDER+"/checkpoints/50.pt"

if not os.path.exists(FINETUNE_FOLDER):
    os.mkdir(FINETUNE_FOLDER)
cfg = OmegaConf.load(args.config)
cfg.model.pre_train_ckpt = PRETRAIN_CHK
cfg.H5AD_FILES = H5AD_LOC
cfg.HIGHLY_VAR_GENES_PATH = args.o + "/feat.csv"

print("finetune config", cfg)
cfg_save_file = FINETUNE_FOLDER +"/cfg.yaml"



with open(cfg_save_file, "w") as fp:
    OmegaConf.save(config=cfg, f=fp.name)
   

# Load files and prepare dataset
files = glob(H5AD_LOC)
if cfg.train.exclude_dataset != "":
    files = [file for file in files if cfg.train.exclude_dataset not in file]
df = prepare_dataset(files, multiprocess=True)
if "exclude_diseases" in cfg.train:
    df = df[~df["disease"].isin(cfg.train.exclude_diseases)]
#print("df head")
#print(df.head())
#print("df shape", df.shape)
#print(df["donor_id"].iloc[0])
#print(df["donor_id"].iloc[1])
#print(df["donor_id"].iloc[2])

# Drop all patients which disease only has one patient
df = df.groupby("disease").filter(lambda x: len(x) > 1)
#the only one this filters out is sALS without NuP defects (line CS2XWC)
lines = np.unique(df["donor_id"])
for held_out in lines:
    print("TRAINING MODEL WITH THIS LINE HELD OUT: ", held_out)
    LINE_FOLDER = args.o+"/finetune_all_perturb/"+ held_out
    if not os.path.exists(LINE_FOLDER):
        os.mkdir(LINE_FOLDER)
    LOG_DIR = LINE_FOLDER + "/logs"
    CHECKPOINT_DIR = LINE_FOLDER+"/checkpoints"
    PREDICTION_FILE = LINE_FOLDER + "/predictions.csv"
    #UMAP_PREDICTION_FILE = LINE_FOLDER + "/umap_predictions.csv"
    pred_file = open(PREDICTION_FILE, 'w')
    #umap_pred_file = open(UMAP_PREDICTION_FILE, 'w')

    if not os.path.exists(CHECKPOINT_DIR):
        os.mkdir(CHECKPOINT_DIR)
    if not os.path.exists(LOG_DIR):
        os.mkdir(LOG_DIR)

    if cfg.train.no_test_dataset:
        #df_use, df_test = train_test_split(
        #    df, test_size=0.2, stratify=df["disease"], random_state=42
        #)  # Note: df_test not used during fine-tuning! Only for later testing
        #df_train, df_val = train_test_split(
        #    df_use, test_size=0.125, stratify=df_use["disease"], random_state=42
        #)
        df_train = df[df['donor_id'] != held_out]
        #print("train df", df_train)
        df_val = df[df['donor_id'] == held_out]
        #print("val df", df_val)

    else:
        df_train, df_val = train_test_split(
            df, test_size=0.2, stratify=df["disease"], random_state=42
        )

    df_train.reset_index(inplace=True)
    df_val.reset_index(inplace=True)
    print(
        f"Using {len(df_train)} patients for training and {len(df_val)} patients for validation representing {len(df['disease'].unique())} unique disease"
    )
    print("Training diseases: ", df_train["disease"].unique())

    ds_train = Mixed_patient_level_dataset(
        df_train,
        select_gene_path=cfg.HIGHLY_VAR_GENES_PATH,
        inference=False,
        n_cells=2000,
        oversampling=cfg.train.oversampling,
    )
    ds_val = Mixed_patient_level_dataset(
        df_val,
        select_gene_path=cfg.HIGHLY_VAR_GENES_PATH,
        inference=False,
        n_cells=800,
        oversampling=cfg.val.oversampling,
    )

    dataloader_train = DataLoader(
        ds_train,
        batch_size=cfg.train.batch_size,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        shuffle=True,
    )
    dataloader_val = DataLoader(
        ds_val,
        batch_size=cfg.train.eval_batch_size,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        shuffle=False,
    )

    model = get_scRNA_model(cfg).cuda()
    train_loss = losses.SupConLoss(temperature=0.1)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.optimizer.lr, weight_decay=cfg.optimizer.weight_decay
    )

    writer = SummaryWriter(LOG_DIR)

    best_loss = np.inf
    best_mean_cos_dist = 2
    best_mean_cos_same_dist = 2
    best_mean_cos_diff_dist = 2
    best_silhouette_score_all = -1
    best_silhouette_score_val = -1


    ##################
    # START OF TRAINING LOOP
    ##################
    for epoch in range(0, cfg.train.num_epochs + 1):
        running_loss = 0

        model.train()
        tqdm_loader_train = tqdm(dataloader_train, total=len(dataloader_train))

        # training loop
        for i, batch in enumerate(tqdm_loader_train):
            # prof.step()
            tqdm_loader_train.set_description(f"Epoch {epoch}, loss: {running_loss:.4f}")
            optimizer.zero_grad()
            x = batch[0].cuda()
            labels = batch[1].cuda()
            x = model(x)
            loss = train_loss(x, labels.float())

            loss.backward()
            optimizer.step()

            running_loss += loss.item() / len(tqdm_loader_train)

        model.eval()

        # Calculate embeddings for all training samples again for later cosine similarity calculation
        tqdm_loader_train = tqdm(dataloader_train, total=len(dataloader_train))
        train_embeddings = torch.zeros((len(ds_train), cfg.model["embed_dim"])).cuda()
        train_diseases = []
        with torch.no_grad():
            for i, batch in enumerate(tqdm_loader_train):
                x = batch[0].cuda()
                labels = batch[1].cuda()
                label_names = batch[2]
                x = model(x)

                train_embeddings[
                    i * dataloader_train.batch_size : i * dataloader_train.batch_size
                    + len(labels),
                    :,
                ] = x.detach().cpu()
                train_diseases += label_names

        # Calculate embeddings for all validation samples
        # validation loop
        val_running_loss = 0
        tqdm_loader_val = tqdm(dataloader_val, total=len(dataloader_val))
        val_embeddings = torch.zeros((len(ds_val), cfg.model["embed_dim"])).cuda()
        val_diseases = []
        with torch.no_grad():
            for i, batch in enumerate(tqdm_loader_val):
                tqdm_loader_val.set_description(
                    f"Epoch {epoch}, val_loss: {val_running_loss:.4f}"
                )
                x = batch[0].cuda()
                labels = batch[1].cuda()
                label_names = batch[2]
                x = model(x)

                val_embeddings[
                    i * dataloader_val.batch_size : i * dataloader_val.batch_size
                    + len(labels),
                    :,
                ] = x.detach().cpu()
                val_diseases += label_names

                loss = train_loss(x, labels.float())

                val_running_loss += loss.item() / len(tqdm_loader_val)

        # calculate cosine similarity of all validation vs training embeddings
        labels_train = np.array(train_diseases)
        labels_val = np.array(val_diseases)

        mean_same_cosine_dist, mean_diff_cosine_dist = cosine_similarity_patient_embeddings(
            train_embeddings, val_embeddings, labels_train, labels_val
        )
        mean_cosine_dist = 0.5 * mean_same_cosine_dist + 0.5 * mean_diff_cosine_dist

        # calculate Silhouette Scores
        #silhouette_score_val = calc_silhouette_score(val_embeddings, labels_val)
        silhouette_score_train = calc_silhouette_score(train_embeddings, labels_train)
        silhouette_score_all = calc_silhouette_score(
            torch.cat([train_embeddings, val_embeddings], dim=0),
            np.concatenate([labels_train, labels_val]),
        )

        # Tensorboard logging
        writer.add_scalar("Loss/train", running_loss, epoch)
        writer.add_scalar("Loss/val", val_running_loss, epoch)
        writer.add_scalar("Learning_rate", optimizer.param_groups[0]["lr"], epoch)
        writer.add_scalar("Weight_decay", optimizer.param_groups[0]["weight_decay"], epoch)
        writer.add_scalar(
            "Mean Cosine Distance between val and train samples", mean_cosine_dist, epoch
        )
        writer.add_scalar("mCosDist same classes", mean_same_cosine_dist, epoch)
        writer.add_scalar("mCosDist diff classes", mean_diff_cosine_dist, epoch)
        #writer.add_scalar("Silhouette Score Validation", silhouette_score_val, epoch)
        writer.add_scalar("Silhouette Score Train", silhouette_score_train, epoch)
        writer.add_scalar("Silhouette Score All", silhouette_score_all, epoch)

        # UMAP plot for Tensorboard
        if epoch % cfg.train.umap_frequency == 0:
            scatter_image = get_plot_as_img(
                np.array(train_embeddings.cpu()),
                np.array(val_embeddings.cpu()),
                labels_train,
                labels_val,
            )
            figure_location =LOG_DIR + "UMAP_epoch" + str(epoch)
            scatter_image.savefig(figure_location)
            writer.add_figure("UMAP Plot", scatter_image, epoch)
            scatter_image_l = get_plot_as_img_largerpts(
                np.array(train_embeddings.cpu()),
                np.array(val_embeddings.cpu()),
                labels_train,
                labels_val,
            )
            figure_location =LOG_DIR + "UMAP_epoch" + str(epoch) + "larger"
            scatter_image_l.savefig(figure_location)

            #add prediction for LOOCV
        #if cfg.train.knn_frequency!=0 and epoch % cfg.train.umap_frequency == 0:
        prediction, right, tie = get_knn_prediction(
            np.array(train_embeddings.cpu()),
            np.array(val_embeddings.cpu()),
            labels_train,
            labels_val,
        )
        if cfg.train.knn_frequency!=0: # and epoch % cfg.train.knn_frequency== 0:
            print(f"Predicted Genotype {prediction} for line {held_out}. Correct (0 no, 1 yes)? {right}, tie? {tie}")
        writer.add_scalar("Left Out Prediction Correct", right)
        csv_string =f"{epoch},{prediction},{right},{tie}\n" 
        pred_file.write(csv_string)

            

        # Save model checkpoint based on different criteria
        if epoch % cfg.train.save_ckpt_freq == 0:
            torch.save(model.state_dict(), CHECKPOINT_DIR + f"/{epoch}.pt")

        if val_running_loss < best_loss:
            best_loss = val_running_loss
            torch.save(model.state_dict(),CHECKPOINT_DIR + "/val_best_loss.pt")

        if mean_cosine_dist < best_mean_cos_dist:
            best_mean_cos_dist = mean_cosine_dist
            torch.save(model.state_dict(), CHECKPOINT_DIR + "/best.pt")

        if mean_same_cosine_dist < best_mean_cos_same_dist:
            best_mean_cos_same_dist = mean_same_cosine_dist
            torch.save(model.state_dict(), CHECKPOINT_DIR + "/best_same.pt")

        if mean_diff_cosine_dist < best_mean_cos_diff_dist:
            best_mean_cos_diff_dist = mean_diff_cosine_dist
            torch.save(model.state_dict(), CHECKPOINT_DIR + "/best_diff.pt")

        if silhouette_score_all > best_silhouette_score_all:
            best_silhouette_score_all = silhouette_score_all
            torch.save(
                model.state_dict(),
                CHECKPOINT_DIR + "/best_silhouette_all.pt",
            )
    pred_file.close()
    #umap_pred_file.close()
        #if silhouette_score_val > best_silhouette_score_val:
        #    best_silhouette_score_val = silhouette_score_val
        #    torch.save(
        #        model.state_dict(),
        #        cfg.train.checkpoints_dir + "/best_silhouette_val.pt",
        #    )
