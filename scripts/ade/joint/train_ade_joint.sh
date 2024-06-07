#!/bin/bash

GPU=0,1
SAVEDIR='save_ade_JOINT-4'

TASKSETTING='overlap'  # or 'disjoint'
TASKNAME='joint'
LR=0.0001
POSWEIGHT=5
EPOCHS=60
BATCH_SIZE=8
LR_SCHEDULER='WarmupPolyLR'
OPTIMIZER='AdamW'

NAME='DKDSeg'
python train_ade.py -c configs/config_ade.json -d ${GPU} --multiprocessing_distributed --save_dir ${SAVEDIR} \
--name ${NAME} --task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 0 --lr ${LR} \
--pos_weight ${POSWEIGHT} --epochs ${EPOCHS} --batch_size ${BATCH_SIZE} \
--lr_scheduler ${LR_SCHEDULER} --optimizer ${OPTIMIZER}