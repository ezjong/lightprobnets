#!/bin/bash

# meta
CHECKPOINT="$PWD/../output/../checkpoint.ckpt"  # set this to your trained model file
CIFAR_HOME=/fastdata/cifar
MODEL=AllConvNetADF
TIME=$(date +"%Y%m%d-%H%M%S")
PREFIX="test-cifar-FGSM-ADF-with-Dirichlet"
SAVE_PATH="$PWD/../output/$MODEL-$TIME-$PREFIX"

# model and checkpoint
python ../main.py \
--batch_size=128 \
--checkpoint=$CHECKPOINT \
--evaluation=True \
--attack=FGSM \
--attack_epsilon=0.01 \
--attack_key=total_loss \
--model=$MODEL \
--num_workers=12 \
--proctitle="$MODEL" \
--save="$SAVE_PATH" \
--loss=DirichletProbOutLoss \
--loss_random_off_targets=False \
--loss_label_smoothing=1e-2 \
--validation_dataset=Cifar10Valid \
--validation_dataset_root=$CIFAR_HOME \
--validation_keys="[top1,xe]" \
--validation_keys_minimize="[False,True]"

# model and checkpoint
python ../main.py \
--batch_size=128 \
--checkpoint=$CHECKPOINT \
--evaluation=True \
--attack=FGSM \
--attack_epsilon=0.05 \
--attack_key=total_loss \
--model=$MODEL \
--num_workers=12 \
--proctitle="$MODEL" \
--save="$SAVE_PATH" \
--loss=DirichletProbOutLoss \
--loss_random_off_targets=False \
--loss_label_smoothing=1e-2 \
--validation_dataset=Cifar10Valid \
--validation_dataset_root=$CIFAR_HOME \
--validation_keys="[top1,xe]" \
--validation_keys_minimize="[False,True]"


# model and checkpoint
python ../main.py \
--batch_size=128 \
--checkpoint=$CHECKPOINT \
--evaluation=True \
--attack=FGSM \
--attack_epsilon=0.1 \
--attack_key=total_loss \
--model=$MODEL \
--num_workers=12 \
--proctitle="$MODEL" \
--save="$SAVE_PATH" \
--loss=DirichletProbOutLoss \
--loss_random_off_targets=False \
--loss_label_smoothing=1e-2 \
--validation_dataset=Cifar10Valid \
--validation_dataset_root=$CIFAR_HOME \
--validation_keys="[top1,xe]" \
--validation_keys_minimize="[False,True]"