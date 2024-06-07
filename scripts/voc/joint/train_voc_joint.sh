#!/bin/bash

GPU=0,1
SAVEDIR='save_voc_JOINT_1'

TASKSETTING='overlap'  # or 'disjoint'
TASKNAME='joint'
LR=0.0001
POSWEIGHT=2
EPOCHS=30
BATCH_SIZE=18
LR_SCHEDULER='PolyLR'
OPTIMIZER='AdamW'

NAME='DKDSeg'
python train_voc.py -c configs/config_voc.json -d ${GPU} --multiprocessing_distributed --save_dir ${SAVEDIR} \
--name ${NAME} --task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 0 --lr ${LR} \
--pos_weight ${POSWEIGHT} --epochs ${EPOCHS} --batch_size ${BATCH_SIZE} \
--lr_scheduler ${LR_SCHEDULER} --optimizer ${OPTIMIZER}