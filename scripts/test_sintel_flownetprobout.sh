#!/bin/bash


# meta
CHECKPOINT="$PWD/../output/,,.checkpoint.ckpt"  # set this to your trained model file
SINTEL_HOME=/fastdata/MPI-Sintel-complete/
MODEL=FlowNetProbOut

# validate clean configuration
PREFIX="validate-sintel-clean"
TIME=$(date +"%Y%m%d-%H%M%S")
SAVE_PATH="$PWD/../output/$MODEL-$TIME-$PREFIX"
python ../main.py \
--batch_size=8 \
--checkpoint=$CHECKPOINT \
--evaluation=True \
--model=$MODEL \
--num_workers=4 \
--proctitle="$MODEL" \
--save=$SAVE_PATH \
--loss=MultiScaleLaplacian \
--loss_with_llh=True \
--validation_dataset=SintelTrainingCleanFull  \
--validation_dataset_root=$SINTEL_HOME \
--validation_keys="[epe]"

# validate final configuration
PREFIX="validate-sintel-final"
SAVE_PATH="$PWD/../output/$MODEL-$TIME-$PREFIX"
python ../main.py \
--batch_size=8 \
--checkpoint=$CHECKPOINT \
--evaluation=True \
--loss=MultiScaleLaplacian \
--loss_with_llh=True \
--model=$MODEL \
--num_workers=4 \
--proctitle="$MODEL" \
--save=$SAVE_PATH \
--validation_dataset=SintelTrainingFinalFull  \
--validation_dataset_root=$SINTEL_HOME \
--validation_keys="[epe]"