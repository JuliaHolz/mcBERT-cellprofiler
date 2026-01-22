#!/bin/bash

#go through whole training process for a combined model (not splitting by perturbation)
#output folder
base_out="/home/jholz/fraenkel_rotation/testing_mcbert_final"
#csv containing cellprofiler data
input_csv="/home/jholz/fraenkel_rotation/cellpainting_data/101625_up_to_1K.csv"
#where to put the h5ad file after processing
h5ad_out="$base_out/standardized_originalfeat.h5ad"
source /home/jholz/.bashrc

#convert our csv file to a h5ad file (comment out these lines if you've already run this step for another model in this folder)
conda activate pyctominer
python ./format_csv_to_h5ad.py -i $input_csv -o $h5ad_out -s standardize

conda activate mcBERT

#create a list of features/"HVGs" (comment out these lines if you've already run this step for another model in this folder)
#this is a file mcBERT expects to exist but is less relevant/could probably be made obsolete since we use all our features (not just highly variable ones)

feat_loc="$base_out/feat.csv"
python mcBERT/extract_features.py -i $h5ad_out -o $feat_loc 

#split the features by perturbation and by line/donor into separate files
#if you've run for one perturbation already you can get rid of this step
output_split_h5ad="$base_out/h5ad_files/standardized_originalfeat"
python mcBERT/separate_donors_treatments_together.py -i $h5ad_out -o $output_split_h5ad

#unsupervised pretraining step
python pretrain_all_perturbation.py  --config configs/configs_pretraining/pretrain_all_perturb.yaml -i "$output_split_h5ad/all_perturb" -o $base_out 


#supervised finetuning step, starting off from where pretraining left off
python LOOCV_all_perturb_train.py --config configs/configs_finetuning/FineTune_all_perturb.yaml -p "$base_out/pretrain_all_perturb" -o "$base_out" -i "$output_split_h5ad/all_perturb"