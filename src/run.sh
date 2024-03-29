#!/bin/bash
PYTHON=/opt/anaconda/envs/jupyter/bin/python

for model_name in \
    "/media/data/models/bert-base-cased" \
    "deepset/bert-base-cased-squad2" \
    "roberta-base" \
    "deepset/roberta-base-squad2"
do
    export MODEL_NAME=$model_name
    export ATTENTION_TYPE="attentions"
    export CUDA_DEVICE_ORDER=PCI_BUS_ID
    export CUDA_VISIBLE_DEVICES=4,5

    CUDA_DEVICE_ORDER=PCI_BUS_ID \
    CUDA_VISIBLE_DEVICES=4,5 \
    TOKENIZERS_PARALLELISM=true \
        $PYTHON ./collect_attentions.py \
        --model $model_name \
        --model_attention_attribute "attentions" \
        --output_directory /media/data/thielen/ba/negation_datasets \
        --split train \
        --max_sequence_length 32 \
        --dataset_path jinaai/negation-dataset \
        --dataset_name "" \
        --overwrite && \
    CUDA_DEVICE_ORDER=PCI_BUS_ID \
    CUDA_VISIBLE_DEVICES=4,5 \
    TOKENIZERS_PARALLELISM=true \
        $PYTHON ./collect_attentions.py \
        --model $model_name \
        --model_attention_attribute "attentions" \
        --output_directory /media/data/thielen/ba/negation_datasets \
        --split train \
        --max_sequence_length 32 \
        --custom_dataset helpers \
        --overwrite && \
    $PYTHON analyze.py
done

for model_name in \
    "/media/data/models/flan-t5-base" \
    "sjrhuschlee/flan-t5-base-squad2"
do
    for attention_type in \
        "encoder_attentions" \
        # "decoder_attentions"
    do
        export MODEL_NAME=$model_name
        export ATTENTION_TYPE=$attention_type

        CUDA_DEVICE_ORDER=PCI_BUS_ID \
        CUDA_VISIBLE_DEVICES=4,5 \
        TOKENIZERS_PARALLELISM=true \
            $PYTHON ./collect_attentions.py \
                --model $model_name \
                --model_attention_attribute $attention_type \
                --output_directory /media/data/thielen/ba/negation_datasets \
                --split train \
                --max_sequence_length 32 \
                --dataset_path jinaai/negation-dataset \
                --dataset_name "" \
                --overwrite \
                --dummy_target && \
        $PYTHON analyze.py
    done
done
