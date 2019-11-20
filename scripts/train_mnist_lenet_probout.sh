#!/bin/bash

# meta
CHECKPOINT=None
MNIST_HOME=/fastdata/mnist
MODEL=LeNetProbOut
PREFIX="train-mnist"
TIME=$(date +"%Y%m%d-%H%M%S")
SAVE_PATH="$PWD/../output/$MODEL-$TIME-$PREFIX"


# training configuration
python ../main.py \
--batch_size=64 \
--checkpoint=$CHECKPOINT \
--model=$MODEL \
--optimizer=Adam \
--optimizer_lr=1e-3 \
--proctitle=$MODEL \
--save="$SAVE_PATH" \
--total_epochs=20 \
--training_dataset=MnistTrain \
--training_dataset_root=$MNIST_HOME \
--training_key=total_loss \
--loss=DirichletProbOutLoss \
--loss_label_smoothing=1e-5 \
--validation_dataset=MnistValid \
--validation_dataset_root=$MNIST_HOME \
--validation_keys="[top1]" \
--validation_keys_minimize="[False]"
