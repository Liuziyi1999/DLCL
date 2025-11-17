#!/bin/bash

cd ..


DATA=Code/Dataset

TRAINER=IVLP
DATASET=$1
CFG=vit_b16
NAME=$2 # job name
TAU=$3 # pseudo label threshold
U=$4 #
SEED=$5




DIR=output/${DATASET}/${TRAINER}/TAU${TAU}_U${U}_${NAME}/seed${SEED}



if [ -d "$DIR" ]; then
    echo "Results are available in ${DIR}."
else
    echo "Run this job and save the output to ${DIR}"

    python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \

fi

wait
