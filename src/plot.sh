#!/bin/bash
PYTHON=/opt/anaconda/envs/jupyter/bin/python

# $PYTHON -m pip install -r requirements.txt

model_names=(
    "/media/data/models/bert-base-cased"
    "deepset/bert-base-cased-squad2"
    "/media/data/models/roberta-base"
    "deepset/roberta-base-squad2"
    "/media/data/models/flan-t5-base"
    "sjrhuschlee/flan-t5-base-squad2"
)
datasets=(
    "jinaai/negation-dataset"
    "LAMA_negated/ConceptNet"
    "LAMA_negated/Squad"
)

for model_name in "${model_names[@]}"
do
    for dataset in "${datasets[@]}"
    do
        if [[ $model_name == *"flan-t5"* ]]; then
            target="encoder_attentions"
        else
            target="attentions"
        fi
        model_name_base=$(basename $model_name)
        dataset_name="/media/data/thielen/ba/negation_datasets/$model_name_base/$target/$dataset/train/32"
        echo "Running: $dataset_name"
        $PYTHON ./plotting.py \
            --model_name $model_name \
            --dataset_name $dataset_name \
            --comparators comparators.euclidean_distance comparators.cosine_distance
    done
done