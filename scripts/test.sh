#!/bin/bash

cd ..


DATA=Code/Dataset

TRAINER=IVLP
DATASET=$1
CFG=vit_b16_c2_ep5_batch4_2+2ctx
SEED=$2


if [ -d "$DIR" ]; then
    echo "Results are available in ${DIR}. Skip this job"
else
    echo "Run this job and save the output to ${DIR}"

    python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir output/DA_fetus_cls/IVLP/TAU0.8_U0.1_epo1/seed1/test \
    --model-dir output/DA_fetus_cls/IVLP/TAU0.8_U0.1_epo1/seed1 \
    --eval-only

fi
