#!/bin/bash

# meta
CHECKPOINT=None
CIFAR_HOME=/fastdata/cifar
MODEL=AllConvNetADF
TIME=$(date +"%Y%m%d-%H%M%S")
PREFIX="train-cifar-ADF-with-Dirichlet"
SAVE_PATH="$PWD/../output/$MODEL-$TIME-$PREFIX"

# training configuration
python ../main.py \
--batch_size=128 \
--checkpoint=$CHECKPOINT \
--loss=DirichletProbOutLoss \
--loss_random_off_targets=False \
--loss_label_smoothing=1e-2 \
--loss_mult=False \
--lr_scheduler=MultiStepLR \
--lr_scheduler_gamma=0.1 \
--lr_scheduler_milestones="[150, 200, 250]" \
--model=$MODEL \
--num_workers=12 \
--optimizer=SGD \
--optimizer_momentum=0.9 \
--optimizer_weight_decay=1e-6 \
--optimizer_lr=0.01 \
--optimizer_group="{'params': '*vr*', 'lr': 0.01}" \
--optimizer_nesterov=True \
--proctitle="$MODEL" \
--save="$SAVE_PATH" \
--total_epochs=300 \
--training_dataset=Cifar10Train \
--training_dataset_root=$CIFAR_HOME \
--training_key=total_loss \
--validation_dataset=Cifar10Valid \
--validation_dataset_root=$CIFAR_HOME \
--validation_keys="[top1,xe]" \
--validation_keys_minimize="[False,True]"
