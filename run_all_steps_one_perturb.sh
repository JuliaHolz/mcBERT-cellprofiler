#!/bin/bash

base_out="/home/jholz/fraenkel_rotation/testing_mcbert_final"
input_csv="/home/jholz/fraenkel_rotation/cellpainting_data/101625_up_to_1K.csv"
perturbation="KPT"
h5ad_out="$base_out/standardized_originalfeat.h5ad"
source /home/jholz/.bashrc

#convert our csv file to a h5ad file (comment out these lines if you've already run this step for another model in this folder)
conda activate pycytominer
python ./format_csv_to_h5ad.py -i $input_csv -o $h5ad_out -s standardize

conda activate mcBERT

#create a list of features/"HVGs" (comment out these lines if you've already run this step for another model in this folder)
#this is a file mcBERT expects to exist but is less relevant/could probably be made obsolete since we use all our features (not just highly variable ones)
feat_loc="$base_out/feat.csv"
python mcBERT/extract_features.py -i $h5ad_out -o $feat_loc 

#split the features by perturbation and by line/donor into separate files
#if you've run for one perturbation already you can get rid of this step
output_split_h5ad="$base_out/h5ad_files/standardized_originalfeat"
python mcBERT/separate_treatments.py -i $h5ad_out -o $output_split_h5ad

#unsupervised pretraining step
python pretrain_one_perturbation.py  --perturb $perturbation --config configs/configs_pretraining/single_perturbation_cellprofiler.yaml -i "$output_split_h5ad/$perturbation" -o $base_out 


#supervised finetuning step (starts off from chkpt50 of pretraining)
python LOOCV_train.py --config configs/configs_finetuning/FineTune_one_perturb.yaml --perturb $perturbation -p "$base_out/pretrain_${perturbation}_only" -o "$base_out" -i "$output_split_h5ad/$perturbation"