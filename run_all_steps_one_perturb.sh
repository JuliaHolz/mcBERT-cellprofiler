#!/bin/bash

base_out="/home/jholz/fraenkel_rotation/testing_mcbert_final"
input_csv="/home/jholz/fraenkel_rotation/cellpainting_data/101625_up_to_1K.csv"
perturbation="KPT"
h5ad_out="$base_out/standardized_originalfeat.h5ad"
source /home/jholz/.bashrc

conda activate pyctominer

python ./format_csv_to_h5ad.py -i $input_csv -o $h5ad_out -s standardize

conda activate mcBERT

feat_loc="$base_out/feat.csv"
python mcBERT/extract_features.py -i $h5ad_out -o $feat_loc 


output_split_h5ad="$base_out/h5ad_files/standardized_originalfeat"
python mcBERT/separate_treatments.py -i $h5ad_out -o $output_split_h5ad


python pretrain_one_perturbation.py  --perturb $perturbation --config configs/configs_pretraining/single_perturbation_cellprofiler.yaml -i "$output_split_h5ad/$perturbation" -o $base_out 



python /home/jholz/fraenkel_rotation/mcBERT-cellprofiler/LOOCV_train.py --config configs/configs_finetuning/FineTune_one_perturb.yaml --perturb $perturbation -p "$base_out/pretrain_${perturbation}_only" -o "$base_out" -i "$output_split_h5ad/$perturbation"